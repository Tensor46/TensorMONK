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


class ConvolutionTranspose(nn.Module):
    def __init__(self, tensor_size, filter_size, out_channels, strides=(1, 1),
                 pad=True, activation="relu", dropout=0., batch_nm=False,
                 pre_nm=False, groups=1, weight_norm=False, *args, **kwargs):
        super(ConvolutionTranspose, self).__init__()
        # Checks
        assert len(tensor_size) == 4 and type(tensor_size) in [list, tuple], \
            "ConvolutionTranspose -- tensor_size must be of length 4 (tuple or list)"
        assert type(filter_size) in [int, list, tuple], "Convolution -- filter_size must be int/tuple/list"
        if isinstance(filter_size, int):
            filter_size = (filter_size, filter_size)
        if isinstance(filter_size, list):
            filter_size = tuple(filter_size)
        assert len(filter_size) == 2, "ConvolutionTranspose -- filter_size length must be 2"
        assert type(strides) in [int, list, tuple], "ConvolutionTranspose -- strides must be int/tuple/list"
        if isinstance(strides, int):
            strides = (strides, strides)
        if isinstance(strides, list):
            strides = tuple(strides)
        assert len(strides) == 2, "ConvolutionTranspose -- strides length must be 2"
        assert isinstance(pad, bool), "ConvolutionTranspose -- pad must be boolean"
        assert isinstance(dropout, float), "ConvolutionTranspose -- dropout must be float"

        pre_nm = True # pre_nm is always turned on
        activation = activation.lower()
        # Modules
        padding = (filter_size[0]//2, filter_size[1]//2) if pad else (0, 0)
        if dropout > 0.:
            self.dropout = nn.Dropout2d(dropout)

        if batch_nm:
            self.Normalization = nn.BatchNorm2d(tensor_size[1])
        pre, pst, act = ActivationFNs(activation, pre_nm)
        if act is not None:
            self.Activation = act

        if weight_norm:
            """ https://arxiv.org/pdf/1602.07868.pdf """
            self.ConvolutionTranspose = nn.utils.weight_norm(nn.ConvTranspose2d(tensor_size[1]//pre, out_channels*pst, filter_size,
                                                             strides, padding, bias=False, groups=groups), name='weight')
        else:
            self.ConvolutionTranspose = nn.ConvTranspose2d(tensor_size[1]//pre, out_channels*pst, filter_size, strides, padding, bias=False, groups=groups)
            torch.nn.init.orthogonal_(self.ConvolutionTranspose.weight)

        # out tensor size
        self.tensor_size = (tensor_size[0], out_channels,
                            (tensor_size[2] - 1)*strides[0] - 2*padding[0] + filter_size[0],
                            (tensor_size[3] - 1)*strides[1] - 2*padding[1] + filter_size[1],)


    def forward(self, tensor, output_size=None):
        if hasattr(self, "dropout"):
            tensor = self.dropout(tensor)
        if hasattr(self, "Normalization"):
            tensor = self.Normalization(tensor)
        if hasattr(self, "Activation"):
            tensor = self.Activation(tensor)

        if output_size is None:
            output_size = self.tensor_size
        output_size = (tensor.size(0), output_size[1], output_size[2], output_size[3])
        tensor = self.ConvolutionTranspose(tensor, output_size=output_size)
        return tensor


from core.NeuralLayers import *
tensor_size = (1, 16, 5, 5)
tensor = torch.rand(*tensor_size)
test = ConvolutionTranspose(tensor_size, (3,3), 3, (2,2), True, "relu", 0.5, True, False)
test(tensor).size()


# x = torch.rand(3,3,10,10)
# test = ConvolutionTranspose((1,3,10,10), (3,3), 16, (2,2), True, "relu", 0.5, True, False)
# test(x,).size()
# test(x, (1, 16, 20, 20)).size()
