"""TensorMONK :: NormAbsMax."""

__all__ = ["NormAbsMax", "NormAbsMax2d"]

import torch
import torch.nn as nn
from typing import Union


class NormAbsMax(nn.Module):
    """Normalize the tensor such that output.abs().amax(dim) == value.

    Args:
        value (float, required): absolute max value of the tensor across dim.
        dim (int/tuple, optional): the dimension or dimensions to normalize
            (default = :obj:`-1`).
        eps (float, optional): a value added to the denominator for numerical
            stability (default = :obj:`1e-2`).

    :rtype: :class:`torch.Tensor`
    """

    def __init__(self, value: float, dim: Union[int, tuple] = -1,
                 eps: float = 1e-2):
        super(NormAbsMax, self).__init__()
        if not isinstance(value, (int, float)):
            raise TypeError(
                f"NormAbsMax: value must be float: {type(value).__name__}")
        if not isinstance(dim, (int, list, tuple)):
            raise TypeError(f"NormAbsMax: dim must be int/list/tuple: "
                            f"{type(dim).__name__}")
        if not isinstance(eps, float):
            raise TypeError(
                f"NormAbsMax: eps must be float: {type(eps).__name__}")
        self.value = value
        self.dim = dim
        self.eps = eps

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor_max = tensor.abs().amax(self.dim, keepdim=True)
        tensor_max = tensor_max.div(self.value).clamp(self.eps)
        return (tensor / tensor_max)

    def __repr__(self):
        value, dim, eps = self.value, self.dim, self.eps
        return f"NormAbsMax: value={value}, dim={dim}, eps={eps}"


class NormAbsMax2d(nn.Module):
    """Normalize the tensor such that output.abs().amax(dim) == value.

    Args:
        features (int, required): C from an expected input of size NCHW.
        value (float, required): absolute max value of the tensor across dim.
        eps (float, optional): the minimum value of denominator for numerical
            stability (default = :obj:`1e-2`).
        momentum (float, optional): the value used for the running_max
            computation (default = :obj:`1e-2`).

    :rtype: :class:`torch.Tensor`
    """

    def __init__(self, features: int, value: float, eps: float = 1e-2,
                 momentum: float = 0.01):
        super(NormAbsMax2d, self).__init__()
        if not isinstance(features, int):
            raise TypeError(f"NormAbsMax2d: features must be int: "
                            f"{type(features).__name__}")
        if not isinstance(value, (int, float)):
            raise TypeError(
                f"NormAbsMax2d: value must be float: {type(value).__name__}")
        if not isinstance(eps, float):
            raise TypeError(f"NormAbsMax2d: eps must be float: "
                            f"{type(eps).__name__}")
        if not isinstance(momentum, float):
            raise TypeError(f"NormAbsMax2d: momentum must be float: "
                            f"{type(momentum).__name__}")
        self.value = value
        self.eps = eps
        self.momentum = momentum
        self.register_buffer("running_max", torch.ones(features))

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.training:
            self.update(tensor)
        return (tensor / self.running_max[None, :, None, None])

    @torch.no_grad()
    def update(self, tensor: torch.Tensor):
        tensor_max = tensor.abs().amax((0, 2, 3))
        tensor_max.div_(self.value).clamp_(self.eps)
        tensor_max.mul_(self.momentum)
        self.running_max.mul_(1 - self.momentum)
        self.running_max.add_(tensor_max)

    def __repr__(self):
        value, momentum, eps = self.value, self.momentum, self.eps
        return f"NormAbsMax2d: value={value}, eps={eps}, momentum={momentum}"
