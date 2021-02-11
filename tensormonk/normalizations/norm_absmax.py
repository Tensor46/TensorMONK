"""TensorMONK :: NormAbsMax."""

__all__ = ["NormAbsMax", ]

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
