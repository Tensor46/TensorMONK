""" TensorMONK :: layers :: Activations """

__all__ = ["Activations"]

import torch
import torch.nn as nn
import torch.nn.functional as F


def maxout(tensor: torch.Tensor) -> torch.Tensor:
    if not tensor.size(1) % 2 == 0:
        raise ValueError("MaxOut: tensor.size(1) must be divisible by n_splits"
                         ": {}".format(tensor.size(1)))
    return torch.max(*tensor.split(tensor.size(1)//2, 1))


class Activations(nn.Module):
    r"""Activation functions. Additional activation functions (other than those
    available in pytorch) are
    :obj:`"hsigm"` & :obj:`"hswish"` (`"Searching for MobileNetV3"
    <https://arxiv.org/pdf/1905.02244>`_),
    :obj:`"maxo"` (`"Maxout Networks" <https://arxiv.org/pdf/1302.4389>`_),
    :obj:`"mish"` (`"Mish: A Self Regularized Non-Monotonic Neural Activation
    Function" <https://arxiv.org/pdf/1908.08681v1>`_),
    :obj:`"squash"` (`"Dynamic Routing Between Capsules"
    <https://arxiv.org/abs/1710.09829>`_) and
    :obj:`"swish"` (`"SWISH: A Self-Gated Activation Function"
    <https://arxiv.org/pdf/1710.05941v1>`_).

    Args:
        tensor_size (tuple, required): Input tensor shape in BCHW
            (None/any integer >0, channels, height, width).
        activation (str, optional): The list of activation options are
            :obj:`"elu"`, :obj:`"gelu"`, :obj:`"hsigm"`, :obj:`"hswish"`,
            :obj:`"lklu"`, :obj:`"maxo"`, :obj:`"mish"`, :obj:`"prelu"`,
            :obj:`"relu"`, :obj:`"relu6"`, :obj:`"rmxo"`, :obj:`"selu"`,
            :obj:`"sigm"`, :obj:`"squash"`, :obj:`"swish"`, :obj:`"tanh"`.
            (default: :obj:`"relu"`)
        elu_alpha (float, optional): (default: :obj:`1.0`)
        lklu_negslope (float, optional): (default: :obj:`0.01`)

    .. code-block:: python

        import torch
        import tensormonk
        print(tensormonk.activations.Activations.METHODS)

        tensor_size = (None, 16, 4, 4)
        activation = "maxo"
        maxout = tensormonk.activations.Activations(tensor_size, activation)
        maxout(torch.randn(1, *tensor_size[1:]))

        tensor_size = (None, 16, 4)
        activation = "squash"
        squash = tensormonk.activations.Activations(tensor_size, activation)
        squash(torch.randn(1, *tensor_size[1:]))

        tensor_size = (None, 16)
        activation = "swish"
        swish = tensormonk.activations.Activations(tensor_size, activation)
        swish(torch.randn(1, *tensor_size[1:]))

    """

    METHODS = ["elu", "gelu", "hsigm", "hswish", "lklu", "maxo", "mish",
               "prelu", "relu", "relu6", "rmxo",
               "selu", "sigm", "squash", "swish", "tanh"]

    def __init__(self, tensor_size: tuple, activation: str = "relu", **kwargs):
        super(Activations, self).__init__()

        if activation is not None:
            activation = activation.lower()
        self.t_size = tensor_size
        self.activation = activation
        self.function = None
        if activation not in self.METHODS:
            raise ValueError("activation: Invalid activation " +
                             "/".join(self.METHODS) +
                             ": {}".format(activation))

        self.function = getattr(self, "_" + activation)
        if activation == "prelu":
            self.weight = nn.Parameter(torch.ones(1) * 0.1)
        if activation == "lklu":
            self.negslope = kwargs["lklu_negslope"] if "lklu_negslope" in \
                kwargs.keys() else 0.01
        if activation == "elu":
            self.alpha = kwargs["elu_alpha"] if "elu_alpha" in \
                kwargs.keys() else 1.0

        self.tensor_size = tensor_size
        if activation in ("maxo", "rmxo"):
            t_size = list(tensor_size)
            t_size[1] = t_size[1] // 2
            self.tensor_size = tuple(t_size)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.function is None:
            return tensor
        return self.function(tensor)

    def _relu(self, tensor: torch.Tensor):
        return F.relu(tensor)

    def _relu6(self, tensor: torch.Tensor):
        return F.relu6(tensor)

    def _lklu(self, tensor: torch.Tensor):
        return F.leaky_relu(tensor, self.negslope)

    def _elu(self, tensor: torch.Tensor):
        return F.elu(tensor, self.alpha)

    def _gelu(self, tensor: torch.Tensor):
        return F.gelu(tensor)

    def _prelu(self, tensor: torch.Tensor):
        return F.prelu(tensor, self.weight)

    def _selu(self, tensor: torch.Tensor):
        return F.selu(tensor)

    def _tanh(self, tensor: torch.Tensor):
        return torch.tanh(tensor)

    def _sigm(self, tensor: torch.Tensor):
        return torch.sigmoid(tensor)

    def _maxo(self, tensor: torch.Tensor):
        if not tensor.size(1) % 2 == 0:
            raise ValueError("MaxOut: tensor.size(1) must be divisible by 2"
                             ": {}".format(tensor.size(1)))
        return torch.max(*tensor.split(tensor.size(1)//2, 1))

    def _rmxo(self, tensor: torch.Tensor):
        return self._maxo(F.relu(tensor))

    def _swish(self, tensor: torch.Tensor):
        return tensor * torch.sigmoid(tensor)

    def _mish(self, tensor: torch.Tensor):
        return tensor * F.softplus(tensor).tanh()

    def _squash(self, tensor: torch.Tensor):
        if not tensor.dim() == 3:
            raise ValueError("Squash requires 3D tensors: {}".format(
                tensor.dim()))
        sum_squares = (tensor ** 2).sum(2, True)
        return (sum_squares/(1+sum_squares)) * tensor / sum_squares.pow(0.5)

    def _hsigm(self, tensor: torch.Tensor):
        return F.relu6(tensor + 3) / 6

    def _hswish(self, tensor: torch.Tensor):
        return self._hsigm(tensor) * tensor

    def __repr__(self):
        return self.activation

    @staticmethod
    def available() -> list:
        return Activations.METHODS

    def flops(self) -> int:
        import numpy as np
        flops = 0
        numel = np.prod(self.t_size[1:])
        if self.activation == "elu":
            # max(0, x) + min(0, alpha*(exp(x)-1))
            flops = numel * 5
        elif self.activation in ("lklu", "prelu", "sigm"):
            flops = numel * 3
        elif self.activation == "maxo":
            # torch.max(*x.split(x.size(1)//2, 1))
            flops = numel / 2
        elif self.activation == "mish":
            # x * tanh(ln(1 + e^x))
            flops = numel * 5
        elif self.activation == "relu":
            # max(0, x)
            flops = numel
        elif self.activation == "relu6":
            # min(6, max(0, x))
            flops = numel * 2
        elif self.activation == "rmxo":
            # maxo(relu(x))
            flops = int(numel * 1.5)
        elif self.activation == "squash":
            # sum_squares = (tensor**2).sum(2, True)
            # (sum_squares/(1+sum_squares)) * tensor / sum_squares.pow(0.5)
            flops = numel * 4 + self.t_size[1] * 2
        elif self.activation == "swish":
            # x * sigm(x)
            flops = numel * 4
        elif self.activation == "tanh":
            # (exp(x) - exp(-x)) / (exp(x) + exp(-x))
            flops = numel * 9
        elif self.activation == "hsigm":
            # min(6, max(0, x + 3)) / 6
            flops = numel * 4
        elif self.activation == "hswish":
            # x * min(6, max(0, x + 3)) / 6
            flops = numel * 8
        return flops
