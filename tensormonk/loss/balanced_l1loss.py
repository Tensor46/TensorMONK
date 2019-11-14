""" TensorMONK's :: loss :: BalancedL1Loss """

import torch
import torch.nn as nn


euler_constant = 2.71828182845904523536  # Euler Napier's Constant


class BalancedL1Loss(nn.Module):
    r""" Balanced L1 Loss
    Libra R-CNN: Towards Balanced Learning for Object Detection
    Paper -- https://arxiv.org/pdf/1904.02701.pdf

    Args:
        alpha (float): Multiplier for |x| < 1
            default = 0.5

        gamma (float): Multiplier for |x| >= 1
            default = 1.5

        reduction (str): The reduction applied to the output.
            options = None | "mean" | "sum" | "mean_of_sum"
            default = None

            "mean_of_sum": sum of elements in last dimension followed by mean.
                           For bounding box or points.
    """
    def __init__(self,
                 alpha: float = 0.5,
                 gamma: float = 1.5,
                 reduction: str = None,
                 **kwargs):
        super(BalancedL1Loss, self).__init__()

        if not isinstance(alpha, float):
            raise TypeError("BalancedL1Loss: alpha must be float: "
                            "{}".format(type(alpha).__name__))
        if not isinstance(gamma, float):
            raise TypeError("BalancedL1Loss: gamma must be float: "
                            "{}".format(type(gamma).__name__))
        if reduction is not None:
            if not isinstance(reduction, str):
                raise TypeError("IOULoss: reduction must be str/None: "
                                "{}".format(type(reduction).__name__))
            reduction = reduction.lower()
            if reduction not in ["sum", "mean", "mean_of_sum"]:
                raise ValueError("IOULoss: reduction must be "
                                 "sum/mean/mean_of_sum/None: "
                                 "{}".format(reduction))

        self.alpha, self.gamma = alpha, gamma
        self.reduction = reduction
        self.tensor_size = 1,

    def forward(self, predicted: torch.Tensor, targets: torch.Tensor):
        assert predicted.shape == targets.shape, \
            "BalancedL1Loss: predicted.shape != targets.shape"

        delta = (predicted - targets).abs()
        # eq (9): alpha * log(b + 1) = gamma
        b = (euler_constant ** (self.gamma / self.alpha)) - 1
        # eq (8)
        C = (self.gamma / b) - self.alpha
        loss = torch.where(
            delta < 1,
            (self.alpha / b) * (b * delta + 1) * (b * delta + 1).log() -
            self.alpha * delta,
            (self.gamma * delta) + C)

        # reduction
        if self.reduction is None:
            return loss
        elif self.reduction == "sum":
            return loss.sum()
        else:
            if self.reduction == "mean_of_sum" and loss.ndim > 1:
                loss = loss.sum(-1)
            return loss.mean()


# predicted = torch.Tensor([[0.1, 0.2, 2.1, 1.2]])
# predicted.requires_grad_(True)
# targets = torch.Tensor([[0.2, 0.1, 0.2, 0.4]])
# BalancedL1Loss(0.5, 1.5)(predicted, targets)
# BalancedL1Loss(0.5, 1.5, "mean")(predicted, targets)
# BalancedL1Loss(0.5, 1.5, "sum")(predicted, targets)
# BalancedL1Loss(0.5, 1.5, "mean_of_sum")(predicted, targets)
