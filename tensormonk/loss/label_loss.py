""" TensorMONK's :: loss :: LabelLoss """

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .utils import one_hot_idx, hard_negative_mask
from typing import Union


class LabelLoss(nn.Module):
    r"""
    Labels loss function for detection tasks.

    Args:
        method (str): Various loss functions for detection tasks.

            options: "ce_with_negative_mining" | "focal" | "focal_dynamic" |
                     "focal_with_negative_mining"
            default: "ce_with_negative_mining"

        focal_alpha (float/Tensor): Alpha for focal loss. Actual focal loss
            implementation requires alpha as a tensor of length n_labels that
            contains class imbalance.

            default: 0.1

        focal_gamma (float): Gamma for focal loss.

            default: 2.

        pos_to_neg_ratio (float): Ratio of positives to negatives for hard
            negative mining.

            default: 1.0 / 3.0

        reduction (str): The reduction applied to the output.

            options = None | "mean" | "sum"
            default = "mean"

    """
    def __init__(self,
                 method: str = "ce_with_negative_mining",
                 focal_alpha: Union[float, Tensor] = 0.1,
                 focal_gamma: float = 2.,
                 pos_to_neg_ratio: float = 1 / 3.,
                 reduction: str = "mean"):

        super(LabelLoss, self).__init__()

        METHODS = ("ce_with_negative_mining",
                   "focal",
                   "focal_dynamic",
                   "focal_with_negative_mining")

        # checks
        if not isinstance(method, str):
            raise TypeError("LabelLoss: method must be str: "
                            "{}".format(type(method).__name__))
        self._method = method.lower()
        if self._method not in METHODS:
            raise ValueError("LabelLoss :: method != " +
                             "/".join(METHODS) +
                             " : {}".format(self._method))

        if not isinstance(focal_alpha, (float, Tensor)):
            raise TypeError("LabelLoss: focal_alpha must be float/"
                            "torch.Tensor: "
                            "{}".format(type(focal_alpha).__name__))
        if not isinstance(focal_gamma, float):
            raise TypeError("LabelLoss: focal_gamma must be float"
                            "{}".format(type(focal_gamma).__name__))
        if isinstance(focal_alpha, Tensor):
            self.register_buffer("_focal_alpha", focal_alpha)
        else:
            self._focal_alpha = focal_alpha
        self._focal_gamma = focal_gamma
        if not isinstance(pos_to_neg_ratio, float):
            raise TypeError("LabelLoss: pos_to_neg_ratio must be float: "
                            "{}".format(type(pos_to_neg_ratio).__name__))
        self._pos_to_neg_ratio = pos_to_neg_ratio
        if reduction is not None:
            if not isinstance(reduction, str):
                raise TypeError("LabelLoss: reduction must be str: "
                                "{}".format(type(reduction).__name__))
            self._reduction = reduction.lower()
            if self._reduction not in ("sum", "mean", "none"):
                raise ValueError("LabelLoss :: reduction != " +
                                 "/".join(('"sum"', '"mean"', '"none"',
                                           "None")) +
                                 " : {}".format(self._reduction))
        else:
            self._reduction = None
        self.tensor_size = 1,

    def forward(self, predictions: Tensor, targets: Tensor):
        if "focal" in self._method:
            loss = self._focal_loss(predictions, targets)
        else:
            loss = self._ce_with_negative_mining(predictions, targets)

        if self._reduction == "mean":
            return loss.mean()
        elif self._reduction == "sum":
            return loss.sum()
        else:
            return loss

    def _focal_loss(self, predictions: Tensor, targets: Tensor):
        # ns = predictions.shape[0]
        predictions, targets = predictions.squeeze(), targets.squeeze()
        _negative_mining = "_negative_mining" in self._method
        _dynamic_focal = "dynamic" in self._method
        alpha, gamma = self._focal_alpha, self._focal_gamma

        if predictions.shape == targets.shape:
            # 2-class binary problem
            if _dynamic_focal and not _negative_mining:
                # computes alpha per (sample / batch)
                if targets.ndim == 2:
                    alpha = targets.gt(0).sum(-1, True)
                else:
                    alpha = targets.gt(0).sum()
                alpha = alpha.float().clamp(1e-6).to(targets.device)
                alpha = alpha / targets.eq(0).float().sum(-1, True)

            p = torch.sigmoid(predictions)
            if _negative_mining and targets.ndim == 2:
                mask = hard_negative_mask(p, targets, self._pos_to_neg_ratio)
                mask = mask.view(-1)
                # mask samples for focal loss
                predictions = predictions.view(-1)[mask]
                targets = targets.view(-1)[mask]
                p = p.view(-1)[mask]
                # update alpha given pos_to_neg_ratio
                alpha = targets.gt(0).sum().float() / targets.numel()

            bce = F.binary_cross_entropy_with_logits(
                predictions, targets.float(), reduction="none")

            pt = p * targets.float() + (1 - p) * (1 - targets.float())
            alpha = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha * bce * ((1 - pt) ** gamma)
        else:
            # N-class (N >= 2)
            n_labels = predictions.shape[-1]
            if _dynamic_focal and not _negative_mining:
                # TODO: How to handle unknwon class alpha
                alpha = [targets.eq(i).sum() for i in range(n_labels)]
                torch.rand(10).gt(0.5).sum()
                if targets.ndim == 2:
                    alpha = torch.stack([targets.eq(i).sum(1)
                                         for i in range(n_labels)], 1)
                    alpha = alpha.unsqueeze(1)
                else:
                    alpha = torch.stack([targets.eq(i).sum()
                                         for i in range(n_labels)])
                    alpha = alpha.unsqueeze(0)
                alpha = alpha.float().clamp(1e-6).to(targets.device)
                alpha = alpha / alpha.sum(-1, True)

            if _negative_mining and targets.ndim == 2:
                mask = hard_negative_mask(predictions, targets,
                                          self._pos_to_neg_ratio)
                mask = mask.view(-1)
                # mask samples for focal loss
                predictions = predictions.view(-1, n_labels)[mask]
                targets = targets.view(-1)[mask]
                # update alpha given pos_to_neg_ratio
                alpha = torch.stack([targets.eq(i).sum()
                                     for i in range(n_labels)])
                alpha = alpha.unsqueeze(0)
                alpha = alpha.float().clamp(1e-6).to(targets.device)
                alpha = alpha / alpha.sum(-1, True)

            p = predictions.softmax(-1)
            genuine_idx = one_hot_idx(targets.view(-1), n_labels)
            p = p.view(-1)
            pt_1st_term = p.mul(0)
            pt_1st_term[genuine_idx] = p[genuine_idx]
            pt_2nd_term = 1 - p
            pt_2nd_term[genuine_idx] = 0
            pt = (pt_1st_term + pt_2nd_term).view(*predictions.shape)
            if isinstance(alpha, Tensor):
                # alpha is Tensor with per label balance
                if alpha.ndim == 1:
                    alpha = alpha.view(*(([1] * targets.ndim) + [n_labels]))
            loss = (- alpha * (1 - pt).pow(gamma) * pt.log()).sum(-1)
        return loss

    def _ce_with_negative_mining(self, predictions: Tensor, targets: Tensor):
        predictions, targets = predictions.squeeze(), targets.squeeze()

        if predictions.shape == targets.shape:
            # 2-class & binary cross entropy
            predictions = torch.sigmoid(predictions)
            mask = hard_negative_mask(predictions, targets,
                                      self._pos_to_neg_ratio)
            mask = mask.view(-1)
            predictions = predictions.view(-1)
            return F.binary_cross_entropy(
                predictions[mask], targets.view(-1)[mask].float(),
                reduction="none")
        else:
            # N-class (N >= 2) & cross entropy
            mask = hard_negative_mask(predictions, targets,
                                      self._pos_to_neg_ratio)
            mask = mask.view(-1)
            predictions = predictions.view(-1, predictions.size(-1))
            return F.cross_entropy(
                predictions[mask], targets.view(-1)[mask],
                reduction="none")

    def __repr__(self):
        output = "LabelLoss: method={}".format(self._method)
        if "mining" in self._method:
            output += ", pos_to_neg_ratio={}".format(self._pos_to_neg_ratio)
        if "focal" in self._method:
            output += ", alpha={}".format(
                "Tensor" if isinstance(self._focal_alpha, Tensor) else
                self._focal_alpha)
            output += ", gamma={}".format(self._focal_gamma)
        return output


# p1, p2 = torch.randn(32, 100), torch.randn(32, 100, 2)
# t1 = torch.rand(32, 100).ge(0.9).long()
# test = LabelLoss("ce_with_negative_mining")
# test(p1, t1), test(p2, t1)
# test = LabelLoss("focal")
# test(p1, t1), test(p2, t1)
# test = LabelLoss("focal", focal_alpha=torch.Tensor([0.9, 0.1]))
# test(p2, t1)
# test = LabelLoss("focal_dynamic")
# test(p1, t1), test(p2, t1)
# test = LabelLoss("focal_with_negative_mining", pos_to_neg_ratio=0.001)
# test(p1, t1), test(p2, t1)
