""" TensorMONK's :: loss :: Categorical """

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .center_function import CenterFunction
from .utils import compute_n_embedding, compute_top15, one_hot_idx
from ..utils import Measures
from typing import Type, Union
import warnings


class Categorical(nn.Module):
    r""" Categorical with weight's to convert embedding to n_labels
    categorical responses.
    Args:
        tensor_size (int/list/tuple, required)
            Shape of tensor in (None/any integer >0, channels, height, width)
            or (None/any integer >0, in_features) or in_features

        n_labels (int, required)
            Number of labels

        loss_type (str, default="entr")
            "entr" / "smax"
                log_softmax + negative log likelihood
            "tsmax" / "taylor_smax"
                taylor series + log_softmax + negative log likelihood
                (https://arxiv.org/pdf/1511.05042.pdf)
            "aaml" / "angular_margin"
                additive angular margin loss (ArcFace)
                (https://arxiv.org/pdf/1801.07698.pdf  eq-3)
            "lmcl" / "large_margin"
                large margin cosine loss (CosFace)
                (https://arxiv.org/pdf/1801.09414.pdf  eq-4)
            "lmgm" / "gaussian_mixture"
                large margin gaussian mixture loss
                (https://arxiv.org/pdf/1803.02988.pdf  eq-17)
            "snnl"
                soft nearest neighbor loss
                (https://arxiv.org/pdf/1902.01889.pdf  eq-1)

        measure (str, default="dot")
            Options = "cosine" / "dot" / "euclidean".
            Large angular margin/large margin cosine loss only use "cosine".
            Gaussian mixture lossonly use "cosine" / "euclidean".

        add_center (bool, default=False)
            Adds center loss to final loss -
            https://ydwen.github.io/papers/WenECCV16.pdf

        center_alpha (float, default = 0.01)
            Alpha for center loss.

        center_scale (float, default=0.5)
            Scale for center loss.

        add_focal (bool, default=False)
            Enables focal loss - https://arxiv.org/pdf/1708.02002.pdf

        focal_alpha (float/Tensor, default=0.5)
            Alpha for focal loss. Actual focal loss implementation requires
            alpha as a tensor of length n_labels that contains class imbalance.

        focal_gamma (float, default=2)
            Gamma for focal loss, default = 2.

        add_hard_negative (bool, default=False)
            Enables hard negative mining

        hard_negative_p (float, default=0.2)
            Probability of hard negatives retained.

        lmgm_alpha (float, default=0.01)
            Alpha in eq-17.

        lmgm_coefficient (float, default=0.1)
            lambda in eq-17

        snnl_measure (str, default="euclidean")
            Squared euclidean or cosine, when cosine the score are subtracted
            by 1

        snnl_alpha (float, default=0.01)
            Alpha in eq-2, hyper-parameter multiplied to soft nearest nieghbor
            loss before adding to cross entropy

        snnl_temperature (float, default=100)
            Temperature in eq-1. When None, it is a trainable parameter with a
            deafult temperature of 10.

        scale (float, default=10)
            scale, s, for large angular margin/large margin cosine loss

        margin (float, default=0.3)
            margin, m, for large angular margin/large margin cosine loss

    Return:
        loss, (top1, top5)
    """

    def __init__(self,
                 tensor_size: tuple,
                 n_labels: int,
                 loss_type: str = "entr",
                 measure: str = "dot",
                 add_center: bool = False,
                 center_alpha: float = 0.01,
                 center_scale: float = 0.5,
                 add_focal: bool = False,
                 focal_alpha: Type[Union[float, Tensor]] = 0.5,
                 focal_gamma: float = 2.,
                 add_hard_negative: bool = False,
                 hard_negative_p: float = 0.2,
                 lmgm_alpha: float = 0.01,
                 lmgm_coefficient: float = 0.1,
                 snnl_measure: str = "cosine",
                 snnl_alpha: float = 0.01,
                 snnl_temperature: float = 100.,
                 scale: float = 10.,
                 margin: float = 0.3,
                 **kwargs):
        super(Categorical, self).__init__()

        METHODS = ("entr", "smax",
                   "tsmax", "taylor_smax",
                   "aaml", "angular_margin",
                   "lmcl", "large_margin",
                   "lmgm", "gaussian_mixture",
                   "snnl", "soft_nn")
        MEASURES = ("cosine", "dot", "euclidean")

        # Checks
        n_embedding = compute_n_embedding(tensor_size)
        if not isinstance(n_labels, int):
            raise TypeError("Categorical: n_labels must be int: "
                            "{}".format(type(n_labels).__name__))
        self.n_labels = n_labels
        if "type" in kwargs.keys():
            loss_type = kwargs["type"]
            warnings.warn("Categorical: 'type' is deprecated, use 'loss_type' "
                          "instead", DeprecationWarning)
        if not isinstance(loss_type, str):
            raise TypeError("Categorical: loss_type must be str: "
                            "{}".format(type(loss_type).__name__))
        self.loss_type = loss_type.lower()
        if self.loss_type not in METHODS:
            raise ValueError("Categorical :: loss_type != " +
                             "/".join(METHODS) +
                             " : {}".format(self.loss_type))
        if not isinstance(measure, str):
            raise TypeError("Categorical: measure must be str: "
                            "{}".format(type(measure).__name__))
        self.measure = measure.lower()
        if self.measure not in MEASURES:
            raise ValueError("Categorical: measure != " +
                             "/".join(MEASURES) +
                             "{}".format(self.measure))

        # loss function
        if self.loss_type in ("entr", "smax", "tsmax", "taylor_smax"):
            self.loss_function = self._cross_entropy
        elif self.loss_type in ("aaml", "angular_margin"):
            if not isinstance(scale, (int, float)):
                raise TypeError("Categorical: scale for aaml/angular_margin "
                                "must be int/float")
            if not isinstance(margin, float):
                raise TypeError("Categorical: margin for aaml/angular_margin "
                                "must be float")
            self.scale = scale
            self.margin = margin
            self.loss_function = self._angular_margin
        elif self.loss_type in ("lmgm", "gaussian_mixture"):
            if not (isinstance(lmgm_alpha, float) and
                    isinstance(lmgm_coefficient, float)):
                raise TypeError("Categorical: lmgm_alpha/lmgm_coefficient"
                                "/both is not float")
            if self.loss_type in ("lmgm", "gaussian_mixture"):
                if self.measure == "dot":
                    raise ValueError("Categorical: measure must be "
                                     "cosine/euclidean for loss_type=lmgm")
            self.lmgm_alpha = lmgm_alpha
            self.lmgm_coefficient = lmgm_coefficient
            self.loss_function = self._gaussian_mixture
        elif self.loss_type in ("lmcl", "large_margin"):
            if not isinstance(scale, (int, float)):
                raise TypeError("Categorical: scale for lmcl/large_margin "
                                "must be int/float")
            if not isinstance(margin, float):
                raise TypeError("Categorical: margin for lmcl/large_margin "
                                "must be float")
            self.scale = scale
            self.margin = margin
            self.loss_function = self._large_margin
        elif self.loss_type in ("snnl", "soft_nn"):
            self.snnl_measure = snnl_measure.lower()
            if not isinstance(self.snnl_measure, str):
                raise TypeError("Categorical: snnl_measure must be str")
            if self.snnl_measure not in ("cosine", "euclidean"):
                raise ValueError("Categorical: snnl_measure must be "
                                 "'cosine'/'euclidean'")
            if not isinstance(snnl_alpha, float):
                raise TypeError("Categorical: snnl_alpha must be float")
            if snnl_temperature is None:
                self.temperature = nn.Parameter(torch.zeros(1).add(10))
            else:
                if not isinstance(snnl_temperature, (int, float)):
                    raise TypeError("Categorical: snnl_temperature must be "
                                    "int/float")
                self.temperature = snnl_temperature
            self.snnl_measure = snnl_measure
            self.snnl_alpha = snnl_alpha
            self.loss_function = self._soft_nn
        self.weight = nn.Parameter(
            F.normalize(torch.randn(n_labels, n_embedding), 2, 1))

        # Center loss
        if "center" in kwargs.keys():
            add_center = kwargs["center"]
            warnings.warn("Categorical: 'center' is deprecated, use "
                          "'add_center' instead", DeprecationWarning)
        if not isinstance(add_center, bool):
            raise TypeError("Categorical: add_center must be bool: "
                            "{}".format(type(add_center).__name__))
        self.add_center = add_center
        if self.add_center:
            if not (isinstance(center_alpha, float) and
                    isinstance(center_scale, float)):
                raise TypeError("Categorical: center_alpha/center_scale/both "
                                "is not float")
            self.register_buffer("center_alpha",
                                 torch.Tensor([center_alpha]).sum())
            self.register_buffer("center_scale",
                                 torch.Tensor([center_scale]).sum())
            self.centers = nn.Parameter(
                F.normalize(torch.randn(n_labels, n_embedding), p=2, dim=1))
            self.center_function = CenterFunction.apply

        # Focal loss
        if not isinstance(add_focal, bool):
            raise TypeError("Categorical: add_focal must be bool: "
                            "{}".format(type(add_focal).__name__))
        self.add_focal = add_focal
        if self.add_focal:
            if not isinstance(focal_alpha, (float, Tensor)):
                raise TypeError("Categorical: focal_alpha must be float/"
                                "torch.Tensor")
                if isinstance(focal_alpha, Tensor):
                    if focal_alpha.numel() != n_labels:
                        raise ValueError("Categorical: focal_alpha.numel() "
                                         "!= n_labels")
            if not isinstance(focal_gamma, float):
                raise TypeError("Categorical: focal_gamma must be float")
            if isinstance(focal_alpha, Tensor):
                self.register_buffer("focal_alpha", focal_alpha)
            else:
                self.focal_alpha = focal_alpha
            self.focal_gamma = focal_gamma

        # Hard negative mining
        if not isinstance(add_hard_negative, bool):
            raise TypeError("Categorical: add_hard_negative must be bool: "
                            "{}".format(type(add_hard_negative).__name__))
        self.add_hard_negative = add_hard_negative
        if self.add_focal and self.add_hard_negative:
            warnings.warn("Categorical: Both focal and hard negative mining "
                          "can not be True, add_hard_negative is set to "
                          "False")
            self.add_hard_negative = False
        if self.add_hard_negative:
            if not isinstance(hard_negative_p, float):
                raise TypeError("Categorical: hard_negative_p is not float")
            if not (0 < hard_negative_p < 1):
                raise ValueError("Categorical: hard_negative_p must be "
                                 "> 0 & < 1: {}".format(hard_negative_p))
            self.hard_negative_p = hard_negative_p
        self.tensor_size = (1, )

    def forward(self, tensor: Tensor, targets: Tensor):
        loss, (top1, top5) = self.loss_function(tensor, targets)
        if self.add_center:
            center_loss = self.center_function(
                tensor, targets.long(), self.centers, self.center_alpha,
                self.center_scale)
            loss = loss + center_loss
        return loss, (top1, top5)

    def _predictions(self, tensor: Tensor, measure: str = "dot") -> Tensor:
        if measure == "euclidean":
            # TODO: euclidean computation is not scalable to larger n_labels
            # euclidean is squared euclidean for stability
            return Measures.sqr_euclidean_pairwise(tensor, self.weight)
        elif measure == "cosine":
            return Measures.cosine_pairwise(tensor, self.weight)
        # default is "dot" product
        return tensor.mm(self.weight.t())

    def _cross_entropy(self, tensor: Tensor, targets: Tensor,
                       is_reponses: bool = False):
        r""" Taylor softmax, and softmax/cross entropy """
        if is_reponses:
            # used by other loss functions (angular_margin/gaussian_mixture/
            # large_margin)
            responses = tensor
        else:
            responses = self._predictions(tensor, self.measure)
            if self.measure == "euclidean":
                responses.neg_()
            (top1, top5) = compute_top15(responses.data, targets.data)
            if self.loss_type == "tsmax":  # Taylor series
                responses = 1 + responses + 0.5*(responses**2)

        if self.add_hard_negative:
            responses, targets = self._hard_negative_mining(responses, targets)

        if self.add_focal:
            """ The loss function is a dynamically scaled cross entropy loss,
            where the scaling factor decays to zero as confidence in the
            correct class increases. """
            loss = self._focal_loss(responses, targets)
        else:
            loss = F.nll_loss(responses.log_softmax(1), targets)
        if is_reponses:
            return loss
        return loss, (top1, top5)

    def _angular_margin(self, tensor: Tensor, targets: Tensor):
        r""" Additive angular margin loss or ArcFace """
        cos_theta = self._predictions(tensor, "cosine")
        (top1, top5) = compute_top15(cos_theta.data, targets.data)

        m, s = min(0.5, self.margin), max(self.scale, 2.)
        true_idx = one_hot_idx(targets, self.n_labels)
        cos_theta = cos_theta.view(-1)
        if torch.__version__.startswith("1.2."):  # pytorch 1.2 inplace issue
            with torch.no_grad():
                true_tensor = torch.zeros_like(cos_theta)
                true_tensor[true_idx] = 1
            cos_theta = cos_theta * (1 - true_tensor) + \
                ((cos_theta).mul(math.cos(m)) -
                 (cos_theta).pow(2).neg().add(1).pow(0.5).mul(math.sin(m))) * \
                true_tensor
        else:
            cos_theta[true_idx] = cos_theta[true_idx].mul(math.cos(m)) - \
                cos_theta[true_idx].pow(2).neg().add(1).pow(0.5).mul(
                                        math.sin(m))
        cos_theta = (cos_theta * s).view(tensor.size(0), -1)
        return self._cross_entropy(cos_theta, targets, True), (top1, top5)

    def _gaussian_mixture(self, tensor: Tensor, targets: Tensor):
        """ Large margin gaussian mixture or lmgm """
        # TODO euclidean computation is not scalable to larger n_labels
        # mahalanobis with identity covariance per paper = squared
        # euclidean -- does euclidean for stability
        # Switch to measure="cosine" if you have out of memory issues
        responses = self._predictions(tensor, self.measure)
        if self.measure != "euclidean":  # invert when not euclidean
            responses.neg_().add_(1)
        (top1, top5) = compute_top15(responses.data.neg(), targets.data)

        true_idx = one_hot_idx(targets, self.n_labels)
        responses = responses.view(-1)
        loss = self.lmgm_coefficient * (responses[true_idx]).mean()
        if torch.__version__.startswith("1.2."):  # pytorch 1.2 inplace issue
            with torch.no_grad():
                true_tensor = torch.zeros_like(responses)
                true_tensor[true_idx] = 1
            responses = responses * (1 - true_tensor) + \
                responses * true_tensor * (1 + self.lmgm_alpha)
        else:
            responses[true_idx] = responses[true_idx] * (1 + self.lmgm_alpha)
        loss = loss + self._cross_entropy(-responses.view(tensor.size(0), -1),
                                          targets, True)
        return loss, (top1, top5)

    def _large_margin(self, tensor: Tensor, targets: Tensor):
        r""" Large margin cosine loss or CosFace """
        cos_theta = self._predictions(tensor, "cosine")
        (top1, top5) = compute_top15(cos_theta.data, targets.data)

        m, s = min(0.5, self.margin), max(self.scale, 2.)
        true_idx = one_hot_idx(targets, self.n_labels)
        cos_theta = cos_theta.view(-1)
        if torch.__version__.startswith("1.2."):  # pytorch 1.2 inplace issue
            with torch.no_grad():
                ms = torch.zeros_like(cos_theta)
                ms[true_idx] = m
            cos_theta = cos_theta - ms
        else:
            cos_theta[true_idx] = cos_theta[true_idx] - m
        cos_theta = (cos_theta * s).view(tensor.size(0), -1)
        return self._cross_entropy(cos_theta, targets, True), (top1, top5)

    def _soft_nn(self, tensor: Tensor, targets: Tensor):
        r""" Soft nearest neighbor loss """
        loss, (top1, top5) = self._cross_entropy(tensor, targets)

        # soft nearest -- requires multiple samples per label in a batch
        same_label = targets.data.view(-1, 1).eq(targets.data.view(1, -1))
        valid = torch.eye(targets.numel()).to(targets.device).eq(0)
        if any((same_label * valid).sum(1)):
            # soft nearest neighbor loss is valid
            if self.snnl_measure == "cosine":
                distance = 1 - Measures.cosine_pairwise(tensor, tensor)
            else:
                distance = Measures.sqr_euclidean_pairwise(tensor, tensor)
            num = distance * (same_label * valid).to(distance.dtype).detach()
            num = (num).div(self.temperature).neg().exp().sum(1)
            den = distance * valid.to(distance.dtype).detach()
            den = (den).div(self.temperature).neg().exp().sum(1)
            snnl = (num / den.add(1e-6)).log().mean()  # eq - 1
            loss = loss + self.snnl_alpha * snnl  # eq - 2
        return loss, (top1, top5)

    def _hard_negative_mining(self, responses: Tensor, targets: Tensor):
        # get class probabilities and find n hard negatives
        p = responses.softmax(1)
        # zero out the genuine to find hard negatives
        genuine_idx = one_hot_idx(targets, self.n_labels)
        p = p.view(-1).contiguous()
        p[genuine_idx] = 0
        p = p.view(-1, self.n_labels)
        n = max(1, int(self.n_labels * self.hard_negative_p))
        hard_negatives = torch.argsort(p.detach(), dim=1)[:, -n:]
        # actual class prediction and n hard_negatives are computed
        new_responses = torch.cat(
            (responses.gather(1, targets.view(-1, 1)),
             responses.gather(1, hard_negatives.view(-1, n))), 1)
        # the new target is always zero given the above concatenate
        new_targets = targets.mul(0)
        return new_responses, new_targets

    def _focal_loss(self, responses: Tensor, targets: Tensor) -> Tensor:
        # https://arxiv.org/pdf/1708.02002.pdf  ::  eq-5
        n, n_labels = responses.shape
        p = responses.softmax(1)
        # hot_targets = one_hot(targets, responses.shape[1])
        # pt = p * hot_targets + (1 - p) * (1 - hot_targets)
        # zero out the genuine to find hard negatives
        genuine_idx = one_hot_idx(targets, n_labels)
        p = p.view(-1)
        pt_1st_term = p.mul(0)
        pt_1st_term[genuine_idx] = p[genuine_idx]
        pt_2nd_term = 1 - p
        pt_2nd_term[genuine_idx] = 0
        pt = pt_1st_term.view(n, -1) + pt_2nd_term.view(n, -1)
        if isinstance(self.focal_alpha, Tensor):
            # alpha is Tensor with per label balance
            return (- self.focal_alpha.view(1, -1) *
                    (1-pt).pow(self.focal_gamma) * pt.log()).sum(1).mean()
        return (- self.focal_alpha * (1-pt).pow(self.focal_gamma) *
                pt.log()).sum(1).mean()

    def __repr__(self):
        return "Categorical ({}): alpha={}, gamma={}, reduction = {}".format(
            self.alpha, self.gamma, self.reduction)


# from tensormonk.loss.utils import (compute_n_embedding, compute_top15,
#                                    one_hot, one_hot_idx)
# from tensormonk.utils import Measures
# from tensormonk.loss.center_function import CenterFunction
# tensor = torch.rand(3, 256)
# targets = torch.tensor([1, 3, 6])
# test = Categorical(256, 10, "aaml", measure="cosine")
# test(tensor, targets)[0].backward()
