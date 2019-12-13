""" TensorMONK :: layers :: CarryResidue """

__all__ = ["ResidualOriginal", "ResidualComplex", "ResidualInverted",
           "ResidualShuffle", "ResidualNeXt",
           "SEResidualComplex", "SEResidualNeXt",
           "SimpleFire", "CarryModular", "DenseBlock",
           "ContextNet_Bottleneck", "SeparableConvolution", "MBBlock"]

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .convolution import Convolution
from ..activations import Activations
from ..regularizations import DropOut
from .utils import check_strides, check_residue, update_kwargs, compute_flops
from copy import deepcopy
import random


def drop_connect(tensor: torch.Tensor, p: float):
    n = tensor.size(0)
    retain = (torch.rand(n, dtype=tensor.dtype) + 1 - p).floor()
    if retain.sum() == 0:
        retain[random.randint(0, n-1)] = 1
    retain = retain.view(-1, *([1] * (tensor.dim()-1))).to(tensor.device)
    return tensor / (1 - p) * retain


class SEBlock(nn.Module):
    r""" Squeeze-and-Excitation """
    def __init__(self, tensor_size, r=16, **kwargs):
        super(SEBlock, self).__init__()

        show_msg = "x".join(["_"]+[str(x)for x in tensor_size[1:]]) + " += "
        self.squeeze = nn.Parameter(torch.randn(
            tensor_size[1]//r, tensor_size[1], 1, 1))
        self.excitation = nn.Parameter(torch.randn(
            tensor_size[1], tensor_size[1]//r, 1, 1))
        nn.init.kaiming_uniform_(self.squeeze)
        nn.init.kaiming_uniform_(self.excitation)
        show_msg += "(pool -> conv({}) -> ".format(
            "x".join(map(str, self.squeeze.shape))) + "relu -> "
        show_msg += "conv({}) -> ".format(
            "x".join(map(str, self.excitation.shape))) + "sigm)"
        self.show_msg = show_msg
        self.r = r
        self.tensor_size = tensor_size

    def forward(self, tensor):
        se = F.avg_pool2d(tensor, tensor.shape[2:])
        se = F.relu(F.conv2d(se, self.squeeze, None))
        se = torch.sigmoid(F.conv2d(se, self.excitation, None))
        return tensor * se

    def flops(self):
        return self.tensor_size[1]*self.r*2 + np.prod(self.tensor_size[1:])*2

    def __repr__(self):
        return self.show_msg
# =========================================================================== #


class ResidualOriginal(nn.Module):
    r""" Residual block with two 3x3 convolutions - used in ResNet18 and
    ResNet34. All args are similar to Convolution.
    """
    def __init__(self, tensor_size, filter_size, out_channels, strides=1,
                 pad=True, activation="relu", dropout=0., normalization=None,
                 pre_nm=False, groups=1, weight_nm=False, equalized=False,
                 shift=False, bias=False, dropblock=False,
                 dropconnect=False, seblock=False, r=16, **kwargs):

        super(ResidualOriginal, self).__init__()
        self.is_dropconnect = dropconnect
        self.p = dropout
        dropout = 0. if dropconnect else dropout
        self.dropout = DropOut(tensor_size, dropout, dropblock, **kwargs)

        kwgs = deepcopy(kwargs)
        kwargs = update_kwargs(kwargs, None, None, None, None, True,
                               activation, 0., normalization, pre_nm, None,
                               weight_nm, equalized, shift, bias)

        self.Block1 = Convolution(tensor_size, filter_size, out_channels,
                                  strides, **kwargs)
        self.Block2 = Convolution(self.Block1.tensor_size, filter_size,
                                  out_channels, 1, **kwargs)
        if seblock:
            self.seblock = SEBlock(self.Block2.tensor_size, r)
        if check_residue(strides, tensor_size, out_channels):
            if not pre_nm:
                kwargs["activation"] = ""
            self.edit_residue = Convolution(tensor_size, 1, out_channels,
                                            strides, **kwargs)
        if not pre_nm and activation is not None:
            if activation.lower() in Activations.available():
                self.activation = Activations(self.Block2.tensor_size,
                                              activation.lower(), **kwgs)
        self.tensor_size = self.Block2.tensor_size

    def forward(self, tensor):
        if self.dropout is not None:  # for dropout
            tensor = self.dropout(tensor)
        residue = self.edit_residue(tensor) if hasattr(self, "edit_residue") \
            else tensor
        tensor = self.Block2(self.Block1(tensor))
        if hasattr(self, "seblock"):
            tensor = self.seblock(tensor)
        if self.is_dropconnect and self.p > 0.:
            tensor = drop_connect(tensor, self.p)
        tensor = tensor + residue
        if hasattr(self, "activation"):
            tensor = self.activation(tensor)
        return tensor

    def flops(self):
        return compute_flops(self) + np.prod(self.tensor_size[1:])
# =========================================================================== #


class ResidualComplex(nn.Module):
    r"""Bottleneck Residual block with 1x1 (out_channels//4), 3x3
    (out_channels//4), and 1x1 (out_channels) convolution - used in ResNet50,
    ResNet101, and ResNet152. All args are similar to Convolution.
    """
    def __init__(self, tensor_size, filter_size, out_channels, strides=1,
                 pad=True, activation="relu", dropout=0., normalization=None,
                 pre_nm=False, groups=1, weight_nm=False, equalized=False,
                 shift=False, bias=False, dropblock=False,
                 dropconnect=False, seblock=False, r=16, **kwargs):
        super(ResidualComplex, self).__init__()
        self.is_dropconnect = dropconnect
        self.p = dropout
        dropout = 0. if dropconnect else dropout
        self.dropout = DropOut(tensor_size, dropout, dropblock, **kwargs)
        kwgs = deepcopy(kwargs)
        kwargs = update_kwargs(kwargs, None, None, None, None, True,
                               activation, 0., normalization, pre_nm, None,
                               weight_nm, equalized, shift, bias)

        self.Block1 = Convolution(tensor_size, 1, out_channels//4, strides,
                                  **kwargs)
        self.Block2 = \
            Convolution(self.Block1.tensor_size, filter_size, out_channels//4,
                        1, groups=groups, **kwargs)
        if not pre_nm:
            kwargs["activation"] = ""
        self.Block3 = \
            Convolution(self.Block2.tensor_size, 1, out_channels, 1, **kwargs)
        if seblock:
            self.seblock = SEBlock(self.Block3.tensor_size, r)
        if check_residue(strides, tensor_size, out_channels):
            self.edit_residue = Convolution(tensor_size, 1, out_channels,
                                            strides, **kwargs)
        if not pre_nm and activation is not None:
            if activation.lower() in Activations.available():
                self.activation = Activations(self.Block3.tensor_size,
                                              activation.lower(), **kwgs)
        self.tensor_size = self.Block3.tensor_size

    def forward(self, tensor):
        if self.dropout is not None:  # for dropout
            tensor = self.dropout(tensor)
        residue = self.edit_residue(tensor) if hasattr(self, "edit_residue") \
            else tensor
        tensor = self.Block3(self.Block2(self.Block1(tensor)))
        if hasattr(self, "seblock"):
            tensor = self.seblock(tensor)
        if self.is_dropconnect and self.p > 0.:
            tensor = drop_connect(tensor, self.p)
        tensor = tensor + residue
        if hasattr(self, "activation"):
            tensor = self.activation(tensor)
        return tensor

    def flops(self):
        return compute_flops(self) + np.prod(self.tensor_size[1:])
# =========================================================================== #


class SEResidualComplex(nn.Module):
    r"""Bottleneck Residual block with squeeze and excitation added. All args
    are similar to Convolution.
    Implemented - https://arxiv.org/pdf/1709.01507.pdf

    Args:
        r: channel reduction factor, default = 16
    """
    def __init__(self, tensor_size, filter_size, out_channels, strides=1,
                 pad=True, activation="relu", dropout=0., normalization=None,
                 pre_nm=False, groups=1, weight_nm=False, equalized=False,
                 shift=False, bias=False, dropblock=False, r=16, **kwargs):
        super(SEResidualComplex, self).__init__()
        self.dropout = DropOut(tensor_size, dropout, dropblock, **kwargs)
        kwgs = deepcopy(kwargs)
        kwargs = update_kwargs(kwargs, None, None, None, None, True,
                               activation, 0., normalization, pre_nm, None,
                               weight_nm, equalized, shift, bias)

        self.Block1 = Convolution(tensor_size, 1, out_channels//4, strides,
                                  **kwargs)
        self.Block2 = \
            Convolution(self.Block1.tensor_size, filter_size, out_channels//4,
                        1, groups=groups, **kwargs)
        if not pre_nm:
            kwargs["activation"] = ""
        self.Block3 = \
            Convolution(self.Block2.tensor_size, 1, out_channels, 1, **kwargs)

        se = [nn.AvgPool2d(self.Block3.tensor_size[2:], stride=(1, 1)),
              Convolution((1, out_channels, 1, 1), 1, out_channels//r, 1,
                          False, "relu"),
              Convolution((1, out_channels//r, 1, 1), 1, out_channels, 1,
                          False, "sigm")]
        self.SE = nn.Sequential(*se)

        if check_residue(strides, tensor_size, out_channels):
            self.edit_residue = Convolution(tensor_size, 1, out_channels,
                                            strides, **kwargs)
        if not pre_nm and activation is not None:
            if activation.lower() in Activations.available():
                self.activation = Activations(self.Block3.tensor_size,
                                              activation.lower(), **kwgs)
        self.tensor_size = self.Block3.tensor_size

    def forward(self, tensor):
        if self.dropout is not None:  # for dropout
            tensor = self.dropout(tensor)
        residue = self.edit_residue(tensor) if hasattr(self, "edit_residue") \
            else tensor
        tensor = self.Block3(self.Block2(self.Block1(tensor)))
        tensor = tensor * self.SE(tensor)
        tensor = tensor + residue
        if hasattr(self, "activation"):
            tensor = self.activation(tensor)
        return tensor

    def flops(self):
        return compute_flops(self) + np.prod(self.tensor_size[1:]) * 2
# =========================================================================== #


class ResidualNeXt(nn.Module):
    r"""Bottleneck Residual block with 1x1 (out_channels//2), 3x3
    (out_channels//2, & groups = out_channels//2), and 1x1 (out_channels)
    convolution. All args are similar to Convolution.
    Implemented - https://arxiv.org/pdf/1611.05431.pdf
    """
    def __init__(self, tensor_size, filter_size, out_channels, strides=1,
                 pad=True, activation="relu", dropout=0., normalization=None,
                 pre_nm=False, groups=32, weight_nm=False, equalized=False,
                 shift=False, bias=False, dropblock=False,
                 dropconnect=False, seblock=False, r=16, **kwargs):
        super(ResidualNeXt, self).__init__()
        self.is_dropconnect = dropconnect
        self.p = dropout
        dropout = 0. if dropconnect else dropout
        self.dropout = DropOut(tensor_size, dropout, dropblock, **kwargs)

        kwargs = update_kwargs(kwargs, None, None, None, None, True,
                               activation, 0., normalization, pre_nm, None,
                               weight_nm, equalized, shift, bias)

        self.Block1 = Convolution(tensor_size, 1, out_channels//2, 1, **kwargs)
        self.Block2 = \
            Convolution(self.Block1.tensor_size, filter_size, out_channels//2,
                        strides, groups=groups, **kwargs)
        self.Block3 = \
            Convolution(self.Block2.tensor_size, 1, out_channels, 1, **kwargs)
        if seblock:
            self.seblock = SEBlock(self.Block3.tensor_size, r)
        if check_residue(strides, tensor_size, out_channels):
            kwargs["activation"] = ""
            self.edit_residue = Convolution(tensor_size, 1, out_channels,
                                            strides, **kwargs)
        self.tensor_size = self.Block3.tensor_size

    def forward(self, tensor):
        if self.dropout is not None:  # for dropout
            tensor = self.dropout(tensor)
        residue = self.edit_residue(tensor) if hasattr(self, "edit_residue") \
            else tensor
        tensor = self.Block3(self.Block2(self.Block1(tensor)))
        if hasattr(self, "seblock"):
            tensor = self.seblock(tensor)
        if self.is_dropconnect and self.p > 0.:
            tensor = drop_connect(tensor, self.p)
        tensor = tensor + residue
        return tensor

    def flops(self):
        return compute_flops(self) + np.prod(self.tensor_size[1:])
# =========================================================================== #


class SEResidualNeXt(nn.Module):
    r"""Custom module combining both Squeeze-and-Excitation and ResNeXt. All
    args are similar to SEResidualComplex.
    """
    def __init__(self, tensor_size, filter_size, out_channels, strides=1,
                 pad=True, activation="relu", dropout=0., normalization=None,
                 pre_nm=False, groups=32, weight_nm=False, equalized=False,
                 shift=False, bias=False, dropblock=False, r=16, **kwargs):
        super(SEResidualNeXt, self).__init__()
        self.dropout = DropOut(tensor_size, dropout, dropblock, **kwargs)

        kwargs = update_kwargs(kwargs, None, None, None, None, True,
                               activation, 0., normalization, pre_nm, None,
                               weight_nm, equalized, shift, bias)

        self.Block1 = Convolution(tensor_size, 1, out_channels//2, 1, **kwargs)
        self.Block2 = \
            Convolution(self.Block1.tensor_size, filter_size, out_channels//2,
                        strides, groups=groups, **kwargs)
        self.Block3 = \
            Convolution(self.Block2.tensor_size, 1, out_channels, 1, **kwargs)
        se = [nn.AvgPool2d(self.Block3.tensor_size[2:], stride=(1, 1)),
              Convolution((1, out_channels, 1, 1), 1, out_channels//r, 1,
                          False, "relu"),
              Convolution((1, out_channels//r, 1, 1), 1, out_channels, 1,
                          False, "sigm")]
        self.SE = nn.Sequential(*se)

        if check_residue(strides, tensor_size, out_channels):
            kwargs["activation"] = ""
            self.edit_residue = Convolution(tensor_size, 1, out_channels,
                                            strides, **kwargs)
        self.tensor_size = self.Block3.tensor_size

    def forward(self, tensor):
        if self.dropout is not None:  # for dropout
            tensor = self.dropout(tensor)
        residue = self.edit_residue(tensor) if hasattr(self, "edit_residue")\
            else tensor
        tensor = self.Block3(self.Block2(self.Block1(tensor)))
        return tensor * self.SE(tensor) + residue

    def flops(self):
        return compute_flops(self) + np.prod(self.tensor_size[1:])
# =========================================================================== #


class SeparableConvolution(nn.Module):
    r""" SeparableConvolution """
    def __init__(self, tensor_size, filter_size, out_channels, strides=1,
                 pad=True, activation="relu", dropout=0., normalization=None,
                 pre_nm=False, groups=1, weight_nm=False, equalized=False,
                 shift=False, bias=False, dropblock=False, **kwargs):
        super(SeparableConvolution, self).__init__()

        self.dropout = DropOut(tensor_size, dropout, dropblock, **kwargs)
        self.Block1 = Convolution(
            tensor_size, filter_size, tensor_size[1], strides, pad, activation,
            0., normalization, pre_nm, tensor_size[1], weight_nm, equalized,
            shift, bias, dropblock, **kwargs)
        self.Block2 = Convolution(self.Block1.tensor_size, 1, out_channels,
                                  1, True, None)
        self.tensor_size = self.Block2.tensor_size

    def forward(self, tensor):
        if self.dropout is not None:  # for dropout
            tensor = self.dropout(tensor)
        return self.Block2(self.Block1(tensor))

    def flops(self):
        return compute_flops(self)
# =========================================================================== #


class ResidualInverted(nn.Module):
    r""" Support for MobileNetV2 - https://arxiv.org/pdf/1801.04381.pdf
    All args are similar to Convolution."""
    def __init__(self, tensor_size, filter_size, out_channels, strides=1,
                 pad=True, activation="relu", dropout=0., normalization=None,
                 pre_nm=False, groups=1, weight_nm=False, equalized=False,
                 shift=False, bias=False, dropblock=False, t=1, t_in=False,
                 dropconnect=False, seblock=False, r=16, **kwargs):
        super(ResidualInverted, self).__init__()
        self.is_dropconnect = dropconnect
        self.p = dropout
        dropout = 0. if dropconnect else dropout
        self.dropout = DropOut(tensor_size, dropout, dropblock, **kwargs)

        kwargs = update_kwargs(kwargs, None, None, None, None, True,
                               activation, 0., normalization, pre_nm, None,
                               weight_nm, equalized, shift, bias)
        channels = int((tensor_size[1] if t_in else out_channels) * t)

        self.Block1 = Convolution(tensor_size, 1, channels, 1, **kwargs)
        self.Block2 = \
            Convolution(self.Block1.tensor_size, filter_size, channels,
                        strides, groups=channels, **kwargs)
        kwargs["activation"] = ""
        self.Block3 = Convolution(self.Block2.tensor_size, 1, out_channels,
                                  1, **kwargs)
        if seblock:
            self.seblock = SEBlock(self.Block3.tensor_size, r)
        self.skip_residue = True if check_strides(strides) else False
        if not self.skip_residue and tensor_size[1] != out_channels:
            self.edit_residue = Convolution(tensor_size, 1, out_channels,
                                            strides, **kwargs)
        self.tensor_size = self.Block3.tensor_size

    def forward(self, tensor):
        if self.dropout is not None:  # for dropout
            tensor = self.dropout(tensor)
        if not self.skip_residue:  # For strides > 1
            residue = self.edit_residue(tensor) if \
                hasattr(self, "edit_residue") else tensor
        tensor = self.Block3(self.Block2(self.Block1(tensor)))
        if hasattr(self, "seblock"):
            tensor = self.seblock(tensor)
        if self.is_dropconnect and self.p > 0.:
            tensor = drop_connect(tensor, self.p)
        return tensor if self.skip_residue else tensor + residue

    def flops(self):
        # residue addition
        flops = 0 if self.skip_residue else np.prod(self.tensor_size[1:])
        return compute_flops(self) + flops
# =========================================================================== #


class ChannelShuffle(nn.Module):
    r""" https://arxiv.org/pdf/1707.01083.pdf """
    def __init__(self, groups, *args, **kwargs):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, tensor):
        tensor_size = tensor.size()
        tensor = tensor.view(tensor_size[0], self.groups, -1,
                             tensor_size[2], tensor_size[3])
        tensor = tensor.transpose(2, 1).contiguous()
        return tensor.view(tensor_size[0], -1,
                           tensor_size[2], tensor_size[3]).contiguous()


class ResidualShuffle(nn.Module):
    r""" ShuffleNet supporting block - https://arxiv.org/pdf/1707.01083.pdf
    All args are similar to Convolution. """
    def __init__(self, tensor_size, filter_size, out_channels, strides=1,
                 pad=True, activation="relu", dropout=0., normalization=None,
                 pre_nm=False, groups=4, weight_nm=False, equalized=False,
                 shift=False, bias=False, dropblock=False, **kwargs):
        super(ResidualShuffle, self).__init__()
        self.dropout = DropOut(tensor_size, dropout, dropblock, **kwargs)
        kwgs = deepcopy(kwargs)
        kwargs = update_kwargs(kwargs, None, None, out_channels, None,
                               True, activation, 0., normalization, pre_nm,
                               groups, weight_nm, equalized, shift, bias)

        self.Block1 = Convolution(tensor_size, 1, **kwargs)
        self.Shuffle = ChannelShuffle(groups)
        kwargs["activation"] = ""
        self.Block2 = Convolution(self.Block1.tensor_size, filter_size,
                                  strides=strides, **kwargs)
        self.Block3 = Convolution(self.Block2.tensor_size, 1, **kwargs)

        self._flops = 0
        if check_strides(strides) and tensor_size[1] == out_channels:
            sz = strides + (1 if strides % 2 == 0 else 0)
            self.edit_residue = nn.AvgPool2d(sz, strides, sz//2)
            self._flops += tensor_size[1]*self.Block3.tensor_size[2] * \
                self.Block3.tensor_size[3]*(sz*sz+1)
        elif not check_strides(strides) and tensor_size[1] != out_channels:
            self.edit_residue = Convolution(tensor_size, 1, out_channels,
                                            **kwargs)
        elif check_strides(strides) and tensor_size[1] != out_channels:
            sz = strides + (1 if strides % 2 == 0 else 0)
            t_size = (1, tensor_size[1], self.Block3.tensor_size[2],
                      self.Block3.tensor_size[3])
            self.edit_residue = [nn.AvgPool2d(3, 2, 1),
                                 Convolution(t_size, 1, **kwargs)]
            self.edit_residue = nn.Sequential(*self.edit_residue)
            self._flops = tensor_size[1]*self.Block3.tensor_size[2] * \
                self.Block3.tensor_size[3]*(sz*sz+1)

        self.tensor_size = self.Block3.tensor_size

        if activation in ("maxo", "rmxo"):  # switch to retain out_channels
            activation = "relu"
        self.Activation = Activations(self.Block3.tensor_size,
                                      activation, **kwgs)

    def forward(self, tensor):
        if self.dropout is not None:
            tensor = self.dropout(tensor)
        residue = self.edit_residue(tensor) if hasattr(self, "edit_residue") \
            else tensor
        tensor = self.Block3(self.Block2(self.Shuffle(self.Block1(tensor))))
        return self.Activation(tensor + residue)

    def flops(self):
        return compute_flops(self)+np.prod(self.tensor_size[1:])+self._flops
# =========================================================================== #


class SimpleFire(nn.Module):
    r"""Fire block for SqueezeNet support. All args are similar to Convolution.
    Implemented - https://arxiv.org/pdf/1602.07360.pdf
    """
    def __init__(self, tensor_size, filter_size, out_channels, strides=1,
                 pad=True, activation="relu", dropout=0., normalization=None,
                 pre_nm=False, groups=1, weight_nm=False, equalized=False,
                 shift=False, bias=False, dropblock=False, **kwargs):
        super(SimpleFire, self).__init__()
        self.dropout = DropOut(tensor_size, dropout, dropblock, **kwargs)

        kwargs = update_kwargs(kwargs, None, None, None,
                               None, True, None, 0., normalization, pre_nm,
                               groups, weight_nm, equalized, shift, bias)

        self.Shrink = Convolution(tensor_size, 1, out_channels//4, 1,
                                  activation=None, **kwargs)
        self.Block3x3 = Convolution(self.Shrink.tensor_size, filter_size,
                                    out_channels//2, strides,
                                    activation=activation, **kwargs)
        self.Block1x1 = Convolution(self.Shrink.tensor_size, 1,
                                    out_channels - out_channels//2, strides,
                                    activation=activation, **kwargs)
        self.tensor_size = (1, out_channels) + self.Block3x3.tensor_size[2:]

    def forward(self, tensor):
        if self.dropout is not None:
            tensor = self.dropout(tensor)
        tensor = self.Shrink(tensor)
        return torch.cat((self.Block3x3(tensor), self.Block1x1(tensor)), 1)

    def flops(self):
        return compute_flops(self)
# =========================================================================== #


class CarryModular(nn.Module):
    r"""Similar to residual connection that concatenate the output to the input
    when in_channels is less than out_channels. When in_channels is equal to
    out_channels, removes the first growth_rate input channels
    (tensor[:, :growth_rate]) and concatenates the new growth_rate at the end
    (tensor[:, -growth_rate:]).
    All args are similar to Convolution and requires out_channels >=
    tensor_size[1].

    Args:
        growth_rate: out_channels of each sub block
        block: any convolutional block is accepted (nn.Sequential or nn.Module)
            default = SimpleFire
        carry_network: only active when strides is >1. When string input =
            "avg", does average pooling else max pool. Also, accepts
            nn.Sequential/nn.Module. default = average pool.
    """
    def __init__(self, tensor_size, filter_size, out_channels, strides=1,
                 pad=True, activation="relu", dropout=0., normalization=None,
                 pre_nm=False, groups=1, weight_nm=False, equalized=False,
                 shift=False, bias=False, dropblock=False, growth_rate=32,
                 block=SimpleFire, carry_network="avg", **kwargs):
        super(CarryModular, self).__init__()
        pad = True
        self.dropout = DropOut(tensor_size, dropout, dropblock, **kwargs)

        if tensor_size[1] < out_channels:  # adjusts growth_rate
            growth_rate = out_channels - tensor_size[1]
        else:
            self.dynamic = True

        self.network1 = block \
            if isinstance(block, torch.nn.modules.container.Sequential) else \
            block(tensor_size, filter_size, growth_rate, strides, pad,
                  activation, 0., normalization, pre_nm, groups, weight_nm,
                  equalized, shift, bias, **kwargs)

        self._flops = 0
        if check_strides(strides):
            if isinstance(carry_network, str):
                self.network2 = nn.AvgPool2d((3, 3), stride=(2, 2), padding=1)\
                    if carry_network.lower() == "avg" else \
                    nn.MaxPool2d((3, 3), stride=(2, 2), padding=1)
                self._flops = tensor_size[1]*(tensor_size[2]//2) * \
                    (tensor_size[3]//2) * (3*3+1)
            elif (isinstance(carry_network, list) or
                  isinstance(carry_network, tuple)):
                self.network2 = nn.Sequential(*carry_network)
            elif isinstance(carry_network,
                            torch.nn.modules.container.Sequential):
                self.network2 = carry_network
            else:
                raise NotImplementedError

        if isinstance(block, torch.nn.modules.container.Sequential):
            _tensor_size = self.network1[-1].tensor_size
        else:
            _tensor_size = self.network1.tensor_size
        self.tensor_size = (_tensor_size[0], out_channels,
                            _tensor_size[2], _tensor_size[3])

    def forward(self, tensor):
        if hasattr(self, "pre_network"):  # for dropout
            tensor = self.pre_network(tensor)
        return torch.cat((self.network1(tensor), self.network2(tensor)), 1)

    def flops(self):
        return compute_flops(self) + self._flops
# =========================================================================== #


class DenseBlock(nn.Module):
    r""" For DenseNet - https://arxiv.org/pdf/1608.06993.pdf
    All args are similar to Convolution and requires out_channels =
    tensor_size[1] + growth_rate * n_blocks.

    Args:
        growth_rate: out_channels of each sub block
        block: any convolutional block is accepted, default = Convolution
        n_blocks: number of sub blocks, default = 4
        multiplier: growth_rate multiplier for 1x1 convolution
    """
    def __init__(self, tensor_size, filter_size, out_channels, strides=1,
                 pad=True, activation="relu", dropout=0., normalization=None,
                 pre_nm=False, groups=1, weight_nm=False, equalized=False,
                 shift=False, bias=False, dropblock=False,
                 growth_rate=16, block=Convolution, n_blocks=4,
                 multiplier=4, **kwargs):
        super(DenseBlock, self).__init__()
        assert out_channels == tensor_size[1] + growth_rate * n_blocks, \
            "DenseBlock -- out_channels != tensor_size[1]+growth_rate*n_blocks"
        kwargs = update_kwargs(kwargs, None, None, None, None, True,
                               activation, 0., normalization, pre_nm,
                               groups, weight_nm, equalized, shift, bias)
        self.dropout = DropOut(tensor_size, dropout, dropblock, **kwargs)

        tensor_size = list(tensor_size)
        self._flops = 0
        if check_strides(strides):  # Update tensor_size
            tensor_size[0] = 1
            sz = strides + (1 if strides % 2 == 0 else 0)
            tensor_size = list(F.avg_pool2d(torch.rand(*tensor_size),
                                            sz, strides, sz//2).size())
            self.pool = nn.AvgPool2d(sz, strides, sz//2)
            self._flops += np.prod(tensor_size[1:]) * (sz*sz+1)

        for n in range(1, n_blocks+1):
            c = growth_rate*multiplier
            t_size = (1, c, tensor_size[2], tensor_size[3])
            dense = [block(tuple(tensor_size), 1, c, **kwargs),
                     block(t_size, filter_size, growth_rate, **kwargs)]
            setattr(self, "block" + str(n), nn.Sequential(*dense))
            tensor_size[1] += growth_rate
        self.n_blocks = n_blocks

        self.tensor_size = (tensor_size[0], out_channels, tensor_size[2],
                            tensor_size[3])

    def forward(self, tensor):
        if self.dropout is not None:
            tensor = self.dropout(tensor)
        if hasattr(self, "pool"):
            tensor = self.pool(tensor)

        for n in range(1, self.n_blocks+1):
            tensor = torch.cat((tensor,
                                getattr(self, "block"+str(n))(tensor)), 1)
        return tensor

    def flops(self):
        return compute_flops(self) + np.prod(self.tensor_size[1:])*3*3 + \
            self._flops
# =========================================================================== #


class ContextNet_Bottleneck(nn.Module):
    r""" bottleneck for contextnet - https://arxiv.org/pdf/1805.04554.pdf
    - Table 1 """
    def __init__(self, tensor_size, filter_size, out_channels, strides=1,
                 pad=True, activation="relu", dropout=0., normalization=None,
                 pre_nm=False, groups=1, weight_nm=False, equalized=False,
                 shift=False, expansion=1, *args, **kwargs):
        super(ContextNet_Bottleneck, self).__init__()

        if dropout > 0.:
            self.pre_network = nn.Dropout2d(dropout)

        kwargs = update_kwargs(kwargs, tensor_size, filter_size, out_channels,
                               strides, True, activation, 0., normalization,
                               pre_nm, groups, weight_nm, equalized, shift)

        self.network = nn.Sequential()
        kwargs["filter_size"] = 1
        kwargs["out_channels"] = tensor_size[1]*expansion
        kwargs["strides"] = 1
        self.network.add_module("Block1x1_t", Convolution(**kwargs))
        kwargs["tensor_size"] = self.network[-1].tensor_size
        kwargs["filter_size"] = filter_size
        kwargs["out_channels"] = tensor_size[1]*expansion
        kwargs["strides"] = strides
        kwargs["groups"] = tensor_size[1]
        self.network.add_module("Block3x3_DW11", Convolution(**kwargs))
        kwargs["tensor_size"] = self.network[-1].tensor_size
        kwargs["filter_size"] = 1  # cross check -- why name Block3x3_DW12?
        kwargs["out_channels"] = tensor_size[1]*expansion
        kwargs["strides"] = 1
        kwargs["groups"] = groups
        self.network.add_module("Block3x3_DW12", Convolution(**kwargs))
        kwargs["tensor_size"] = self.network[-1].tensor_size
        kwargs["filter_size"] = 1
        kwargs["out_channels"] = out_channels
        kwargs["activation"] = ""
        kwargs["groups"] = groups
        self.network.add_module("Block1x1", Convolution(**kwargs))

        if check_residue(strides, tensor_size, out_channels):
            kwargs["tensor_size"] = tensor_size
            kwargs["filter_size"] = 1
            kwargs["out_channels"] = out_channels
            kwargs["strides"] = strides
            kwargs["activation"] = ""
            self.edit_residue = Convolution(**kwargs)
        self.tensor_size = self.network[-1].tensor_size

    def forward(self, tensor):
        if hasattr(self, "pre_network"):  # for dropout
            tensor = self.pre_network(tensor)
        if hasattr(self, "edit_residue"):
            return self.network(tensor) + self.edit_residue(tensor)
        return self.network(tensor) + tensor

    def flops(self):
        return compute_flops(self) + np.prod(self.tensor_size[1:])
# =========================================================================== #


class MBBlock(nn.Module):
    r""" Support for EfficientNets - https://arxiv.org/pdf/1905.11946.pdf

    Args (not in Convolution):
        expansion (int): Expansion factor of tensor_size[1] for depthwise
            convolution. initial 1x1 convolution is ignored when 1.

            default = 1

        seblock (bool): Adds Squeese and Excitation block.

            default = False

        r (int): factor for squeeze and excitation

            default = 4

        ichannels (int): This overwrites expansion parameter

            default = None
    """
    def __init__(self, tensor_size, filter_size, out_channels, strides=1,
                 pad=True, activation="swish", dropout=0.,
                 normalization="batch", pre_nm=False,
                 expansion=1, seblock=False, r=4, ichannels=None, **kwargs):
        super(MBBlock, self).__init__()

        self.p = dropout
        if ichannels is None:
            ichannels = int(tensor_size[1] * expansion)

        if ichannels != tensor_size[1]:
            self.expand = Convolution(tensor_size, 1, ichannels, 1, True,
                                      activation, 0., normalization, pre_nm,
                                      **kwargs)
        t_size = self.expand.tensor_size if expansion > 1 else tensor_size

        self.depthwise = Convolution(t_size, filter_size, ichannels, strides,
                                     True, activation, 0., normalization,
                                     pre_nm, groups=ichannels, **kwargs)
        if seblock:
            self.squeeze = Convolution(self.depthwise.tensor_size, 1,
                                       tensor_size[1]//r, 1, True, activation,
                                       bias=True)
            self.excitation = Convolution(self.squeeze.tensor_size, 1,
                                          self.depthwise.tensor_size[1], 1,
                                          True, "sigm", bias=True)
        self.shrink = Convolution(self.depthwise.tensor_size, 1, out_channels,
                                  1, True, None, 0., normalization,
                                  pre_nm, **kwargs)
        self.tensor_size = self.shrink.tensor_size

    def forward(self, tensor):
        o = self.expand(tensor) if hasattr(self, "expand") else tensor
        o = self.depthwise(o)
        if hasattr(self, "squeeze"):
            o = o * self.excitation(self.squeeze(F.adaptive_avg_pool2d(o, 1)))
        o = self.shrink(o)
        if tensor.shape[1:] == o.shape[1:]:
            if self.p > 0. and self.training:
                o = drop_connect(o, self.p)
            return tensor + o
        return o


# from tensormonk.layers import Convolution
# from tensormonk.activations import Activations
# from tensormonk.regularizations import DropOut
# from tensormonk.layers.utils import check_strides, check_residue
# from tensormonk.layers.utils import update_kwargs, compute_flops
# tensor_size = (3, 64, 10, 10)
# x = torch.rand(*tensor_size)
# test = ResidualOriginal(tensor_size, 3, 64, 2, False, "relu", 0.,
#                         "batch", False)
# test(x).size()
# test.flops()
# %timeit test(x).size()
# test = ResidualComplex(tensor_size, 3, 64, 2, False, "relu", 0., "batch",
#                        False)
# test(x).size()
# test.flops()
# %timeit test(x).size()
# test = ResidualInverted(tensor_size, 3, 64, 1, False, "relu", 0., "batch",
#                         False)
# test(x).size()
# test.flops()
# %timeit test(x).size()
# test = ResidualShuffle(tensor_size, 3, 64, 2, False, "relu", 0., "batch",
#                        False)
# test(x).size()
# test.flops()
# %timeit test(x).size()
# test = SimpleFire(tensor_size, 3, 64, 2, False, "relu", 0.1, None, False)
# test(x).size()
# test.flops()
# %timeit test(x).size()
# test = CarryModular(tensor_size, 3, 128, 2, False, "relu", 0., None, False)
# test(x).size()
# test.flops()
# %timeit test(x).size()
# test = SEResidualComplex(tensor_size, 3, 64, 2, False, "relu", 0., "batch",
#                          False)
# test(x).size()
# test.flops()
# %timeit test(x).size()
# test = ResidualNeXt(tensor_size, 3, 64, 2, False, "relu", 0., "batch", False)
# test(x).size()
# test.flops()
# %timeit test(x).size()
# test = SEResidualNeXt(tensor_size, 3, 64, 2, False, "relu", 0., "batch",
#                       False)
# test(x).size()
# test.flops()
# %timeit test(x).size()
# test = DenseBlock(tensor_size, 3, 128, 2, True, "relu", 0., "batch", False)
# test(x).size()
# test.flops()
# %timeit test(x).size()
# test = ContextNet_Bottleneck(tensor_size, 3, 128, 1)
# test(x).size()
# test.flops()
# %timeit test(x).size()
# test = MBBlock(tensor_size, 3, 64, 1, True, "swish", 0.5, seblock=True)
# test(torch.rand(*tensor_size)).size()
# test
# %timeit test(x).size()
