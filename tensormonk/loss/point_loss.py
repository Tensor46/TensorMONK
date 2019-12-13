""" TensorMONK's :: loss :: PointLoss """

__all__ = ["PointLoss"]

import torch
import torch.nn.functional as F
from torch import Tensor


class PointLoss(torch.nn.Module):
    r""" Point loss function for detection tasks.

    Args:
        method (str): Loss functions for point predictions.
            options: "smooth_l1" | "mse"
            default: "mse"

        reduction (str): The reduction applied to the output.
            options = "mean" | "sum" | "mean_of_sum"
            default = "mean"
    """

    KWARGS = ("method", "reduction")
    METHODS = ("smooth_l1", "mse")

    def __init__(self,
                 method: str = "mse",
                 reduction: str = "mean",
                 **kwargs):

        super(PointLoss, self).__init__()

        # checks
        if not isinstance(method, str):
            raise TypeError("PointLoss: method must be str: "
                            "{}".format(type(method).__name__))
        self._method = method.lower()
        if self._method not in PointLoss.METHODS:
            raise ValueError("PointLoss :: method != " +
                             "/".join(PointLoss.METHODS) +
                             " : {}".format(self._method))

        if not isinstance(reduction, str):
            raise TypeError("PointLoss: reduction must be str: "
                            "{}".format(type(reduction).__name__))
        reduction = reduction.lower()
        if reduction not in ["sum", "mean", "mean_of_sum"]:
            raise ValueError("PointLoss: reduction must be sum/mean/"
                             "mean_of_sum: {}".format(reduction))
        self._reduction = reduction
        self.tensor_size = 1,

    def forward(self,
                p_point: Tensor,  # predicted point from network
                t_point: Tensor,  # target point mapped to all possible pixels
                t_label: Tensor,  # target labels for each target box
                **kwargs):
        assert p_point.size(0) == t_point.size(0) == t_label.size(0), \
            "PointLoss: p_point.size(0) != t_point.size(0) != t_label.size(0)"
        t_label = t_label.view(-1)
        p_point = p_point.view(t_label.size(0), -1, 2)
        t_point = t_point.view(t_label.size(0), -1, 2)
        assert p_point.shape == t_point.shape, \
            "PointLoss: p_point.shape != t_point.shape"

        # filter p_point and t_point given t_label
        valid = t_label.gt(0)
        p_point, t_point = p_point[valid], t_point[valid]
        loss = 0.
        for i in range(t_point.size(1)):
            _p, _t = p_point[:, i], t_point[:, i]
            with torch.no_grad():
                nans = torch.isnan(_t).sum(1).gt(0)
            if not (~ nans).sum():
                continue
            if "mse" == self._method:
                if self._reduction == "sum":
                    loss = loss + F.mse_loss(
                        _p[~ nans], _t[~ nans], reduction="sum")
                elif self._reduction == "mean":
                    loss = loss + F.mse_loss(
                        _p[~ nans], _t[~ nans], reduction="mean")
                else:
                    loss = loss + F.mse_loss(
                        _p[~ nans], _t[~ nans], reduction="none").sum(1).mean()
            else:
                if self._reduction == "sum":
                    loss = loss + F.smooth_l1_loss(
                        _p[~ nans], _t[~ nans], reduction="sum")
                elif self._reduction == "mean":
                    loss = loss + F.smooth_l1_loss(
                        _p[~ nans], _t[~ nans], reduction="mean")
                else:
                    loss = loss + F.smooth_l1_loss(
                        _p[~ nans], _t[~ nans], reduction="none").sum(1).mean()
        return loss / p_point.size(1)

    def __repr__(self):
        return "PointLoss: method={}".format(self._method)


# p_point = torch.Tensor([[0.1, 0.2, -0.1, 0.3, -0.1, 0.3],
#                         [0.1, 0.2,  0.4, 0.9, -0.8, 0.7]])
# t_point = torch.Tensor([[0.2, 0.1, -0.2, 0.4, -0.1, 0.3],
#                         [0.2, 0.1,  0.2, 0.4,  0.6, 0.1]])
# t_label = torch.Tensor([1, 1]).long()
# PointLoss("smooth_l1", "mean")(p_point, t_point, t_label)
# PointLoss("mse", "mean")(p_point, t_point, t_label)
# PointLoss("smooth_l1", "sum")(p_point, t_point, t_label)
# PointLoss("mse", "sum")(p_point, t_point, t_label)
# PointLoss("smooth_l1", "mean_of_sum")(p_point, t_point, t_label)
# PointLoss("mse", "mean_of_sum")(p_point, t_point, t_label)
