"""TensorMONK :: Normalize."""

__all__ = ["NormAbsMaxDynamic", "NormAbsMax2d"]

import math
import torch
import torch.nn as nn
from typing import Union


class NormAbsMaxDynamic(nn.Module):
    """Normalize the tensor such that output.abs().amax(dim) == value.

    Args:
        value (float, required): absolute max value of the tensor across dim.
        dim (int/tuple, optional): the dimension or dimensions to normalize
            (default = :obj:`-1`). When value is None, dim must be -1 and the
            dynamic value is computed per sample.
        eps (float, optional): a value added to the denominator for numerical
            stability (default = :obj:`1e-2`).

    :rtype: :class:`torch.Tensor`
    """

    def __init__(self, value: float, dim: Union[int, tuple] = -1,
                 eps: float = 1e-2):
        super(NormAbsMaxDynamic, self).__init__()
        if not (isinstance(value, (int, float)) or value is None):
            raise TypeError(f"NormAbsMaxDynamic: value must be float: "
                            f"{type(value).__name__}")
        if value is None or value == -1:
            if dim != -1:
                raise ValueError(f"NormAbsMaxDynamic: dim must be -1 when "
                                 f"value is None: {dim}")
        if isinstance(value, int) and not value == -1:
            raise ValueError(f"NormAbsMaxDynamic: value must be > 0: {value}")
        if not isinstance(dim, (int, list, tuple)):
            raise TypeError(f"NormAbsMaxDynamic: dim must be int/list/tuple: "
                            f"{type(dim).__name__}")
        if not isinstance(eps, float):
            raise TypeError(
                f"NormAbsMaxDynamic: eps must be float: {type(eps).__name__}")
        if value is None:  # computes the value based on incoming tensor
            value = -1
        self.register_buffer("value", torch.tensor(float(value)))
        self.dim = dim
        self.eps = eps

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.value.eq(-1):  # computes the value & dim using tensor
            if tensor.ndim == 2 or tensor.ndim == 3:
                # value increases with decrease number of features.
                dim = 1 if tensor.ndim == 2 else (1, 2)
                value = max(2, 8 / math.log10(tensor.shape[-1] ** 0.5))
            elif tensor.ndim == 4:
                # value increases with decrease in height and width.
                h, w = tensor.shape[2:]
                dim, value = (1, 2, 3), max(2, 8 / math.log10((h * w) ** 0.5))
            else:
                raise NotImplementedError
            value = torch.tensor(value)
            self.dim, self.value.data = dim, value.to(self.value.device)
        tensor_max = tensor.abs().amax(self.dim, keepdim=True)
        tensor_max = tensor_max.div(self.value).clamp(self.eps)
        return (tensor / tensor_max)

    def __repr__(self):
        value, dim, eps = self.value, self.dim, self.eps
        return f"NormAbsMaxDynamic: value={value}, dim={dim}, eps={eps}"


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
