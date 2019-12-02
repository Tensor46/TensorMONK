""" TensorMONK's :: loss :: BoxesLoss """

__all__ = ["BoxesLoss"]

import torch
import torch.nn.functional as F
from torch import Tensor
from .iou_loss import IOULoss
from .balanced_l1loss import BalancedL1Loss


class BoxesLoss(torch.nn.Module):
    r""" Boxes loss function for detection tasks.

    Args:
        method (str): Loss functions for bounding box predictions.
            options: "smooth_l1" | "balanced_l1" | "mse" | "iou" | "log_iou" |
                     "giou" | "log_giou"
            default: "log_iou"

        l1_alpha (float): Multiplier for |x| < 1 when method is "balanced_l1"
            default = 0.5

        l1_gamma (float): Multiplier for |x| >= 1 when method is "balanced_l1"
            default = 1.5

        reduction (str): The reduction applied to the output.
            options = None | "mean" | "sum" | "mean_of_sum"
                "mean_of_sum" is only valid for ("smooth_l1" | "balanced_l1")
            default = "mean"
    """

    KWARGS = ("method", "l1_alpha", "l1_gamma", "reduction")
    METHODS = ("smooth_l1", "balanced_l1", "mse",
               "iou", "log_iou", "giou", "log_giou")

    def __init__(self,
                 method: str = "log_iou",
                 l1_alpha: float = 0.5,
                 l1_gamma: float = 1.5,
                 reduction: str = "mean",
                 **kwargs):

        super(BoxesLoss, self).__init__()

        # checks
        if not isinstance(method, str):
            raise TypeError("BoxesLoss: method must be str: "
                            "{}".format(type(method).__name__))
        self._method = method.lower()
        if self._method not in BoxesLoss.METHODS:
            raise ValueError("BoxesLoss :: method != " +
                             "/".join(BoxesLoss.METHODS) +
                             " : {}".format(self._method))

        if not isinstance(l1_alpha, float):
            raise TypeError("BoxesLoss: l1_alpha must be float: "
                            "{}".format(type(l1_alpha).__name__))
        if not isinstance(l1_gamma, float):
            raise TypeError("BoxesLoss: l1_gamma must be float: "
                            "{}".format(type(l1_gamma).__name__))

        if reduction is not None:
            if not isinstance(reduction, str):
                raise TypeError("BoxesLoss: reduction must be str/None: "
                                "{}".format(type(reduction).__name__))
            reduction = reduction.lower()
            if reduction not in ["sum", "mean", "mean_of_sum"]:
                raise ValueError("BoxesLoss: reduction must be "
                                 "sum/mean/mean_of_sum/None: "
                                 "{}".format(reduction))
            if reduction == "mean_of_sum" and \
               self._method not in ("smooth_l1", "balanced_l1", "mse"):
                import warnings
                reduction = "mean"
                warnings.warn(
                    "BoxesLoss: Changed reduction to 'mean'. "
                    "'mean_of_sum' is invalid for {}".format(self._method))
        self._reduction = reduction

        if "iou" in self._method:
            self.function = IOULoss(method=self._method,
                                    reduction=self._reduction,
                                    box_form="d_ltrb")
        elif self._method == "balanced_l1":
            self.function = BalancedL1Loss(alpha=l1_alpha,
                                           gamma=l1_gamma,
                                           reduction=self._reduction)
        elif self._method == "smooth_l1":
            self.function = self._smooth_l1
            self._l1_alpha, self._l1_gamma = l1_alpha, l1_gamma
        else:
            self.function = self._mse

        self.tensor_size = 1,

    def forward(self,
                p_boxes: Tensor,  # predicted boxes from network
                t_boxes: Tensor,  # target boxes mapped to all possible pixels
                t_label: Tensor,  # target labels for each target box
                weights: Tensor = None):

        assert p_boxes.shape == t_boxes.shape, \
            "BoxesLoss: p_boxes.shape != t_boxes.shape"
        assert p_boxes.shape[-1] == 4 and t_boxes.shape[-1] == 4, \
            "BoxesLoss: p_boxes.shape[-1]/t_boxes.shape[-1] != 4"
        assert p_boxes.shape[:-1] == t_label.shape, \
            "BoxesLoss: p_boxes.shape[:-1] != t_label.shape"

        # filter p_boxes, t_boxes and weights given t_label
        p_boxes, t_boxes = p_boxes.view(-1, 4), t_boxes.view(-1, 4)
        valid = t_label.view(-1).gt(0)
        p_boxes, t_boxes = p_boxes[valid], t_boxes[valid]
        if weights is not None:
            assert weights.shape == t_label.shape, \
                "BoxesLoss: weights.shape != t_label.shape"
            weights = weights.view(-1)[valid]

        if "iou" in self._method:
            loss = self.function(p_boxes, t_boxes, weights)
        else:
            loss = self.function(p_boxes, t_boxes)

        return loss

    def _smooth_l1(self, p_boxes: Tensor, t_boxes: Tensor):
        loss = F.smooth_l1_loss(p_boxes, t_boxes, reduction='none')

        # reduction
        if self._reduction is None:
            return loss
        elif self._reduction == "sum":
            return loss.sum()
        else:
            if self._reduction == "mean_of_sum" and loss.ndim > 1:
                loss = loss.sum(-1)
            return loss.mean()

    def _mse(self, p_boxes: Tensor, t_boxes: Tensor):
        loss = F.mse_loss(p_boxes, t_boxes, reduction='none')
        # reduction
        if self._reduction is None:
            return loss
        elif self._reduction == "sum":
            return loss.sum()
        else:
            if self._reduction == "mean_of_sum" and loss.ndim > 1:
                loss = loss.sum(-1)
            return loss.mean()

    def __repr__(self):
        return "BoxesLoss: method={}".format(self._method)


# from tensormonk.loss import BalancedL1Loss, IOULoss
# p_boxes = torch.Tensor([[0.1, 0.2, 0.1, 0.3], [0.1, 0.2, 0.1, 0.3]])
# t_boxes = torch.Tensor([[0.2, 0.1, 0.2, 0.4], [0.2, 0.1, 0.2, 0.4]])
# t_label = torch.Tensor([1, 0]).long()
# BoxesLoss("smooth_l1")(p_boxes, t_boxes, t_label)
# BoxesLoss("smooth_l1", reduction="mean_of_sum")(p_boxes, t_boxes, t_label)
# BoxesLoss("balanced_l1")(p_boxes, t_boxes, t_label)
# BoxesLoss("balanced_l1", reduction="mean_of_sum")(p_boxes, t_boxes, t_label)
# BoxesLoss("iou")(p_boxes, t_boxes, t_label)
# BoxesLoss("log_iou")(p_boxes, t_boxes, t_label)
# BoxesLoss("giou")(p_boxes, t_boxes, t_label)
# BoxesLoss("log_giou")(p_boxes, t_boxes, t_label)
