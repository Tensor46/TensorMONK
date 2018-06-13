""" TensorMONK's :: NeuralLayers :: Convolution                              """

__all__ = ["Convolution", ]

import torch
import torch.nn as nn
# ============================================================================ #


class MaxOut(nn.Module):
    """ Implemented https://arxiv.org/pdf/1302.4389.pdf """
    def __init__(self):
        super(MaxOut, self).__init__()

    def forward(self, tensor):
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


def ActivationFNs(activation, pre_nm):
    if activation == "relu":
        return 1, 1, nn.ReLU()
    if activation == "relu6":
        return 1, 1, nn.ReLU6()
    if activation == "lklu":
        return 1, 1, nn.LeakyReLU()
    if activation == "tanh":
        return 1, 1, nn.Tanh()
    if activation == "sigm":
        return 1, 1, nn.Sigmoid()
    if activation == "maxo":
        return (2, 1, MaxOut()) if pre_nm else (1, 2, MaxOut())
    if activation == "swish":
        return 1, 1, Swish()
    return 1, 1, None
# ============================================================================ #


class Convolution(nn.Module):
    def __init__(self, tensor_size, filter_size, out_channels, strides=(1, 1),
                 pad=True, activation="relu", dropout=0., batch_nm=False,
                 pre_nm=False, groups=1, weight_nm=False, *args, **kwargs):
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
        assert isinstance(activation, str), "Convolution -- pad must be str"
        assert isinstance(dropout, float), "Convolution -- dropout must be float"
        assert isinstance(batch_nm, bool), "Convolution -- batch_nm must be boolean"
        assert isinstance(pre_nm, bool), "Convolution -- dropout must be boolean"
        assert isinstance(groups, int), "Convolution -- dropout must be int"
        assert isinstance(weight_nm, bool), "Convolution -- dropout must be boolean"
        activation = activation.lower()
        # Modules
        padding = (filter_size[0]//2, filter_size[1]//2) if pad else 0
        if dropout > 0.:
            self.dropout = nn.Dropout2d(dropout)
        pre, pst = 1, 1
        if pre_nm:
            if batch_nm:
                self.Normalization = nn.BatchNorm2d(tensor_size[1])
            pre, pst, act = ActivationFNs(activation, pre_nm)
            if act is not None:
                self.Activation = act
        if weight_nm:
            """ https://arxiv.org/pdf/1602.07868.pdf """
            self.Convolution = nn.utils.weight_norm(nn.Conv2d(tensor_size[1]//pre, out_channels*pst, filter_size,
                                                              strides, padding, bias=False, groups=groups), name='weight')
        else:
            self.Convolution = nn.Conv2d(tensor_size[1]//pre, out_channels*pst, filter_size, strides, padding, bias=False, groups=groups)
            torch.nn.init.orthogonal_(self.Convolution.weight)
        if not pre_nm:
            if batch_nm:
                self.Normalization = nn.BatchNorm2d(out_channels*pst)
            pre, pst, act = ActivationFNs(activation, pre_nm)
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
# test = Convolution((1,3,10,10), (3,3), 16, (2,2), False, "maxo", 0.5, True, False)
# test(x).size()
