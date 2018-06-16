""" TensorMONK's :: NeuralLayers :: Convolution                              """

__all__ = ["Convolution", ]

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


def ActivationFNs(activation):
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
# ============================================================================ #


class Convolution(nn.Module):
    def __init__(self, tensor_size, filter_size, out_channels, strides=(1, 1),
                 pad=True, activation="relu", dropout=0., batch_nm=False,
                 pre_nm=False, groups=1, weight_norm=False, *args, **kwargs):
        super(Convolution, self).__init__()
        # Checks
        assert len(tensor_size) == 4 and type(tensor_size) in [list, tuple], \
            "Convolution -- tensor_size must be of length 4 (tuple or list)"
        assert type(filter_size) in [int, list, tuple], "Convolution -- filter_size must be int/tuple/list"
        if isinstance(filter_size, int):
            filter_size = (filter_size, filter_size)
        if isinstance(filter_size, list):
            filter_size = tuple(filter_size)
        assert len(filter_size) == 2, "Convolution -- filter_size length must be 2"
        assert type(strides) in [int, list, tuple], "Convolution -- strides must be int/tuple/list"
        if isinstance(strides, int):
            strides = (strides, strides)
        if isinstance(strides, list):
            strides = tuple(strides)
        assert len(strides) == 2, "Convolution -- strides length must be 2"
        assert isinstance(pad, bool), "Convolution -- pad must be boolean"
        assert isinstance(dropout, float), "Convolution -- dropout must be float"
        activation = activation.lower()
        # Modules
        padding = (filter_size[0]//2, filter_size[1]//2) if pad else 0
        if dropout > 0.:
            self.dropout = nn.Dropout2d(dropout)
        pre_expansion, pst_expansion = 1, 1
        if pre_nm and activation in ("maxo", "rmxo"):
            pre_expansion = 2
        if not pre_nm and activation in ("maxo", "rmxo"):
            pst_expansion = 2
        if pre_nm:
            if batch_nm:
                self.Normalization = nn.BatchNorm2d(tensor_size[1])
            act = ActivationFNs(activation)
            if act is not None:
                self.Activation = act
        if weight_norm:
            """ https://arxiv.org/pdf/1602.07868.pdf """
            self.Convolution = nn.utils.weight_norm(nn.Conv2d(tensor_size[1]//pre_expansion, out_channels*pst_expansion, filter_size,
                                                              strides, padding, bias=False, groups=groups), name='weight')
        else:
            self.Convolution = nn.Conv2d(tensor_size[1]//pre_expansion, out_channels*pst_expansion, filter_size, strides, padding, bias=False, groups=groups)
            torch.nn.init.orthogonal_(self.Convolution.weight)
        if not pre_nm:
            if batch_nm:
                self.Normalization = nn.BatchNorm2d(out_channels*pst_expansion)
            act = ActivationFNs(activation)
            if act is not None:
                self.Activation = act
        self.pre_nm = pre_nm
        # out tensor size
        self.tensor_size = (tensor_size[0], out_channels,
                            int(1+(tensor_size[2] + (filter_size[0]//2 * 2 if pad else 0) - filter_size[0])/strides[0]),
                            int(1+(tensor_size[3] + (filter_size[1]//2 * 2 if pad else 0) - filter_size[1])/strides[1]))

    def forward(self, tensor):
        if hasattr(self, "dropout"):
            tensor = self.dropout(tensor)
        if self.pre_nm:
            if hasattr(self, "Normalization"):
                tensor = self.Normalization(tensor)
            if hasattr(self, "Activation"):
                tensor = self.Activation(tensor)
            tensor = self.Convolution(tensor)
        else:
            tensor = self.Convolution(tensor)
            if hasattr(self, "Normalization"):
                tensor = self.Normalization(tensor)
            if hasattr(self, "Activation"):
                tensor = self.Activation(tensor)
        return tensor


# x = torch.rand(3,3,10,10)
# test = Convolution((1,3,10,10), (3,3), 16, (2,2), False, "rmxo", 0., True, False)
# test(x).size()
