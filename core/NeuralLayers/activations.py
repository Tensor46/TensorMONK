""" TensorMONK's :: NeuralLayers :: Activations                              """

__all__ = ["Activations", ]

import torch
import torch.nn as nn
import torch.nn.functional as F
# ============================================================================ #


class MaxOut(nn.Module):
    """ Implemented https://arxiv.org/pdf/1302.4389.pdf """
    def __init__(self):
        super(MaxOut, self).__init__()

    def forward(self, tensor):
        return torch.max(*tensor.split(tensor.size(1)//2, 1))
# ============================================================================ #


class ReluMaxOut(nn.Module):
    """ maxout(relu(x)) """
    def __init__(self):
        super(ReluMaxOut, self).__init__()

    def forward(self, tensor):
        tensor = F.relu(tensor)
        return torch.max(*tensor.split(tensor.size(1)//2, 1))
# ============================================================================ #


class Swish(nn.Module):
    """ Implemented https://arxiv.org/pdf/1710.05941v1.pdf """
    def __init__(self):
        super(Swish, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, tensor):
        return tensor * self.sigmoid(tensor)
# ============================================================================ #


def Activations(activation):
    if activation == "relu":
        return nn.ReLU()
    if activation == "relu6":
        return nn.ReLU6()
    if activation == "lklu":
        return nn.LeakyReLU()
    if activation == "tanh":
        return nn.Tanh()
    if activation == "sigm":
        return nn.Sigmoid()
    if activation == "maxo":
        return MaxOut()
    if activation == "rmxo":
        return ReluMaxOut()
    if activation == "swish":
        return Swish()
    return None


# x = torch.rand(3,3,10,10)
# test = Activations("sigm")
# test(x).max()
