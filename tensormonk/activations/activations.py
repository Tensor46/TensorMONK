""" TensorMONK :: layers :: Activations """

__all__ = ["Activations", "maxout", "mish", "squash", "swish"]

import torch
import torch.nn as nn
import torch.nn.functional as F


def squash(tensor: torch.Tensor) -> torch.Tensor:
    if not tensor.dim() == 3:
        raise ValueError("Squash requires 3D tensors: {}".format(
            tensor.dim()))
    sum_squares = (tensor**2).sum(2, True)
    return (sum_squares/(1+sum_squares)) * tensor / sum_squares.pow(0.5)


def swish(tensor: torch.Tensor) -> torch.Tensor:
    return tensor * torch.sigmoid(tensor)


def maxout(tensor: torch.Tensor) -> torch.Tensor:
    if not tensor.size(1) % 2 == 0:
        raise ValueError("MaxOut: tensor.size(1) must be divisible by n_splits"
                         ": {}".format(tensor.size(1)))
    return torch.max(*tensor.split(tensor.size(1)//2, 1))


def mish(tensor: torch.Tensor) -> torch.Tensor:
    return tensor * tensor.exp().add(1).log().tanh()


class Activations(nn.Module):
    r""" All the usual activations along with maxout, relu + maxout and swish.
    MaxOut (maxo) - https://arxiv.org/pdf/1302.4389.pdf
    Swish - https://arxiv.org/pdf/1710.05941v1.pdf
    Mish - https://arxiv.org/pdf/1908.08681v1.pdf

    Args:
        tensor_size: shape of tensor in BCHW
            (None/any integer >0, channels, height, width)
        activation: relu/relu6/lklu/elu/gelu/prelu/tanh/selu/sigm/maxo/rmxo
            /swish/mish, default=relu
    """
    def __init__(self, tensor_size: tuple, activation: str = "relu", **kwargs):
        super(Activations, self).__init__()

        if activation is not None:
            activation = activation.lower()
        self.t_size = tensor_size
        self.activation = activation
        self.function = None
        if activation not in self.available():
            raise ValueError("activation: Invalid activation " +
                             "/".join(self.available()) +
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

        t_size = list(tensor_size)
        t_size[1] = t_size[1] // 2
        self.tensor_size = tensor_size if activation not in ("maxo", "rmxo") \
            else tuple(t_size)

    def forward(self, tensor: torch.Tensor):
        if self.function is None:
            return tensor
        return self.function(tensor)

    def _relu(self, tensor):
        return F.relu(tensor)

    def _relu6(self, tensor):
        return F.relu6(tensor)

    def _lklu(self, tensor):
        return F.leaky_relu(tensor, self.negslope)

    def _elu(self, tensor):
        return F.elu(tensor, self.alpha)

    def _gelu(self, tensor):
        return F.gelu(tensor)

    def _prelu(self, tensor):
        return F.prelu(tensor, self.weight)

    def _selu(self, tensor):
        return F.selu(tensor)

    def _tanh(self, tensor):
        return torch.tanh(tensor)

    def _sigm(self, tensor):
        return torch.sigmoid(tensor)

    def _maxo(self, tensor):
        return maxout(tensor)

    def _rmxo(self, tensor):
        return maxout(F.relu(tensor))

    def _swish(self, tensor):
        return swish(tensor)

    def _mish(self, tensor):
        return mish(tensor)

    def _squash(self, tensor):
        return squash(tensor)

    def __repr__(self):
        return self.activation

    @staticmethod
    def available():
        return ["elu", "gelu", "lklu", "maxo", "mish", "prelu", "relu",
                "relu6", "rmxo", "selu", "sigm", "squash", "swish", "tanh"]

    def flops(self):
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
            flops = numel*4 + self.t_size[1]*2
        elif self.activation == "swish":
            # x * sigm(x)
            flops = numel * 4
        elif self.activation == "tanh":
            # (exp(x) - exp(-x)) / (exp(x) + exp(-x))
            flops = numel * 7
        return flops


# Activations.available()
# x = torch.rand(3, 4, 10, 10).mul(2).add(-1)
# test = Activations("prelu")
# test(x).min()
