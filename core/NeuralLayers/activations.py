""" TensorMONK's :: NeuralLayers :: Activations                             """

__all__ = ["Activations", ]

import torch
import torch.nn as nn
import torch.nn.functional as F


class Activations(nn.Module):
    r""" All the usual activations along with maxout, relu + maxout and swish.
    MaxOut (maxo) - https://arxiv.org/pdf/1302.4389.pdf
    Swish - https://arxiv.org/pdf/1710.05941v1.pdf

    Args:
        activation: relu/relu6/lklu/elu/prelu/tanh/sigm/maxo/rmxo/swish
        channels: parameter for prelu, default is 1
    """
    def __init__(self, activation="relu", channels=1):
        super(Activations, self).__init__()

        if activation is not None:
            activation = activation.lower()
        self.activation = activation
        self.function = None
        if activation in self.available():
            self.function = getattr(self, "_" + activation)
            if activation == "prelu":
                self.weight = nn.Parameter(torch.rand(channels))

    def forward(self, tensor):
        if self.function is None:
            return tensor
        return self.function(tensor)

    def _relu(self, tensor):
        return F.relu(tensor)

    def _relu6(self, tensor):
        return F.relu6(tensor)

    def _lklu(self, tensor):
        return F.leaky_relu(tensor)

    def _elu(self, tensor):
        return F.elu(tensor)

    def _prelu(self, tensor):
        return F.prelu(tensor, self.weight)

    def _tanh(self, tensor):
        return F.tanh(tensor)

    def _sigm(self, tensor):
        return torch.sigmoid(tensor)

    def _maxo(self, tensor):
        assert tensor.size(1) % 2 == 0, "MaxOut: tensor.size(1) must be even"
        return torch.max(*tensor.split(tensor.size(1)//2, 1))

    def _rmxo(self, tensor):
        return self._maxo(F.relu(tensor))

    def _swish(self, tensor):
        return tensor * torch.sigmoid(tensor)

    @staticmethod
    def available():
        return ["relu", "relu6", "lklu", "elu", "prelu", "tanh", "sigm",
                "maxo", "rmxo", "swish"]


# Activations.available()
# x = torch.rand(3, 4, 10, 10).mul(2).add(-1)
# test = Activations("prelu")
# test(x).min()
