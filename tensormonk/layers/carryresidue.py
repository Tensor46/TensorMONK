""" TensorMONK :: layers :: CarryResidue """

__all__ = ["ResidualOriginal", "ResidualComplex", "ResidualInverted",
           "ResidualShuffle", "ResidualNeXt",
           "SEResidualComplex", "SEResidualNeXt",
           "SimpleFire", "CarryModular", "DenseBlock",
           "ContextNet_Bottleneck"]

import torch
import torch.nn as nn
import torch.nn.functional as F
from .convolution import Convolution
from ..activations import Activations
from ..regularizations import DropOut
from .utils import check_strides, check_residue, update_kwargs
from copy import deepcopy
# =========================================================================== #


class ResidualOriginal(nn.Module):
    r""" Residual block with two 3x3 convolutions - used in ResNet18 and
    ResNet34. All args are similar to Convolution.
    """
    def __init__(self, tensor_size, filter_size, out_channels, strides=1,
                 pad=True, activation="relu", dropout=0., normalization=None,
                 pre_nm=False, groups=1, weight_nm=False, equalized=False,
                 shift=False, bias=False, dropblock=True, **kwargs):

        super(ResidualOriginal, self).__init__()
        self.dropout = DropOut(tensor_size, dropout, dropblock, **kwargs)
        kwgs = deepcopy(kwargs)
        kwargs = update_kwargs(kwargs, None, None, None, None, True,
                               activation, 0., normalization, pre_nm, None,
                               weight_nm, equalized, shift, bias)

        self.Block1 = Convolution(tensor_size, filter_size, out_channels,
                                  strides, **kwargs)
        self.Block2 = Convolution(self.Block1.tensor_size, filter_size,
                                  out_channels, 1, **kwargs)

        if check_residue(strides, tensor_size, out_channels):
            if not pre_nm:
                kwargs["activation"] = ""
            self.edit_residue = Convolution(tensor_size, 1, out_channels,
                                            strides, **kwargs)
        if not pre_nm and activation is not None:
            if activation.lower() in Activations.available():
                self.activation = Activations(activation.lower(), out_channels,
                                              **kwgs)
        self.tensor_size = self.Block2.tensor_size

    def forward(self, tensor):
        if self.dropout is not None:  # for dropout
            tensor = self.dropout(tensor)
        residue = self.edit_residue(tensor) if hasattr(self, "edit_residue") \
            else tensor
        tensor = self.Block2(self.Block1(tensor)) + residue
        if hasattr(self, "activation"):
            tensor = self.activation(tensor)
        return tensor
# =========================================================================== #


class ResidualComplex(nn.Module):
    r"""Bottleneck Residual block with 1x1 (out_channels//4), 3x3
    (out_channels//4), and 1x1 (out_channels) convolution - used in ResNet50,
    ResNet101, and ResNet152. All args are similar to Convolution.
    """
    def __init__(self, tensor_size, filter_size, out_channels, strides=1,
                 pad=True, activation="relu", dropout=0., normalization=None,
                 pre_nm=False, groups=1, weight_nm=False, equalized=False,
                 shift=False, bias=False, dropblock=True, **kwargs):
        super(ResidualComplex, self).__init__()
        self.dropout = DropOut(tensor_size, dropout, dropblock, **kwargs)
        kwgs = deepcopy(kwargs)
        kwargs = update_kwargs(kwargs, None, None, None, None, True,
                               activation, 0., normalization, pre_nm, None,
                               weight_nm, equalized, shift, bias)

        self.Block1 = Convolution(tensor_size, 1, out_channels//4, 1, **kwargs)
        self.Block2 = \
            Convolution(self.Block1.tensor_size, filter_size, out_channels//4,
                        strides, groups=groups, **kwargs)
        if not pre_nm:
            kwargs["activation"] = ""
        self.Block3 = \
            Convolution(self.Block2.tensor_size, 1, out_channels, 1, **kwargs)

        if check_residue(strides, tensor_size, out_channels):
            self.edit_residue = Convolution(tensor_size, 1, out_channels,
                                            strides, **kwargs)
        if not pre_nm and activation is not None:
            if activation.lower() in Activations.available():
                self.activation = Activations(activation.lower(), out_channels,
                                              **kwgs)
        self.tensor_size = self.Block3.tensor_size

    def forward(self, tensor):
        if self.dropout is not None:  # for dropout
            tensor = self.dropout(tensor)
        residue = self.edit_residue(tensor) if hasattr(self, "edit_residue") \
            else tensor
        tensor = self.Block3(self.Block2(self.Block1(tensor))) + residue
        if hasattr(self, "activation"):
            tensor = self.activation(tensor)
        return tensor
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
                 shift=False, bias=False, dropblock=True, r=16, **kwargs):
        super(SEResidualComplex, self).__init__()
        self.dropout = DropOut(tensor_size, dropout, dropblock, **kwargs)
        kwgs = deepcopy(kwargs)
        kwargs = update_kwargs(kwargs, None, None, None, None, True,
                               activation, 0., normalization, pre_nm, None,
                               weight_nm, equalized, shift, bias)

        self.Block1 = Convolution(tensor_size, 1, out_channels//4, 1, **kwargs)
        self.Block2 = \
            Convolution(self.Block1.tensor_size, filter_size, out_channels//4,
                        strides, groups=groups, **kwargs)
        if not pre_nm:
            kwargs["activation"] = ""
        self.Block3 = \
            Convolution(self.Block2.tensor_size, 1, out_channels, 1, **kwargs)

        se = [Convolution((1, out_channels, 1, 1), 1, out_channels//r, 1,
                          False, "relu"),
              Convolution((1, out_channels//r, 1, 1), 1, out_channels, 1,
                          False, "sigm")]
        self.SE = nn.Sequential(*se)

        if check_residue(strides, tensor_size, out_channels):
            self.edit_residue = Convolution(tensor_size, 1, out_channels,
                                            strides, **kwargs)
        if not pre_nm and activation is not None:
            if activation.lower() in Activations.available():
                self.activation = Activations(activation.lower(), out_channels,
                                              **kwgs)
        self.tensor_size = self.Block3.tensor_size

    def forward(self, tensor):
        if self.dropout is not None:  # for dropout
            tensor = self.dropout(tensor)
        residue = self.edit_residue(tensor) if hasattr(self, "edit_residue") \
            else tensor
        tensor = self.Block3(self.Block2(self.Block1(tensor)))
        tensor = tensor * self.SE(F.avg_pool2d(tensor, tensor.shape[2:]))
        tensor = tensor + residue
        if hasattr(self, "activation"):
            tensor = self.activation(tensor)
        return tensor
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
                 shift=False, bias=False, dropblock=True, **kwargs):
        super(ResidualNeXt, self).__init__()
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
        return self.Block3(self.Block2(self.Block1(tensor))) + residue
# =========================================================================== #


class SEResidualNeXt(nn.Module):
    r"""Custom module combining both Squeeze-and-Excitation and ResNeXt. All
    args are similar to SEResidualComplex.
    """
    def __init__(self, tensor_size, filter_size, out_channels, strides=1,
                 pad=True, activation="relu", dropout=0., normalization=None,
                 pre_nm=False, groups=32, weight_nm=False, equalized=False,
                 shift=False, bias=False, dropblock=True, r=16, **kwargs):
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
        self.SqueezeExcitation = nn.Sequential(*se)

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
        return tensor * self.SqueezeExcitation(tensor) + residue
# =========================================================================== #


class ResidualInverted(nn.Module):
    r""" Support for MobileNetV2 - https://arxiv.org/pdf/1801.04381.pdf
    All args are similar to Convolution."""
    def __init__(self, tensor_size, filter_size, out_channels, strides=1,
                 pad=True, activation="relu", dropout=0., normalization=None,
                 pre_nm=False, groups=1, weight_nm=False, equalized=False,
                 shift=False, bias=False, dropblock=True, t=1, **kwargs):
        super(ResidualInverted, self).__init__()
        self.dropout = DropOut(tensor_size, dropout, dropblock, **kwargs)

        kwargs = update_kwargs(kwargs, None, None, None, None, True,
                               activation, 0., normalization, pre_nm, None,
                               weight_nm, equalized, shift, bias)

        self.Block1 = Convolution(tensor_size, 1, out_channels*t, 1, **kwargs)
        self.Block2 = \
            Convolution(self.Block1.tensor_size, filter_size, out_channels*t,
                        strides, groups=out_channels*t, **kwargs)
        kwargs["activation"] = ""
        self.Block3 = Convolution(self.Block2.tensor_size, 1, out_channels,
                                  1, **kwargs)

        self.skip_residue = True if check_strides(strides) else False
        if not self.skip_residue and tensor_size[1] != out_channels:
            self.edit_residue = Convolution(tensor_size, 1, out_channels,
                                            strides, **kwargs)
        self.tensor_size = self.Block3.tensor_size

    def forward(self, tensor):
        if self.dropout is not None:  # for dropout
            tensor = self.dropout(tensor)
        if self.skip_residue:  # For strides > 1
            return self.Block3(self.Block2(self.Block1(tensor)))
        residue = self.edit_residue(tensor) if hasattr(self, "edit_residue")\
            else tensor
        return self.Block3(self.Block2(self.Block1(tensor))) + residue
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
                 shift=False, bias=False, dropblock=True, **kwargs):
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

        if check_strides(strides) and tensor_size[1] == out_channels:
            self.edit_residue = nn.AvgPool2d(3, 2, 1)
        elif not check_strides(strides) and tensor_size[1] != out_channels:
            self.edit_residue = Convolution(tensor_size, 1, out_channels,
                                            **kwargs)
        elif check_strides(strides) and tensor_size[1] != out_channels:
            t_size = (1, tensor_size[1], self.Block3.tensor_size[2],
                      self.Block3.tensor_size[3])
            self.edit_residue = [nn.AvgPool2d(3, 2, 1),
                                 Convolution(t_size, 1, **kwargs)]
            self.edit_residue = nn.Sequential(*self.edit_residue)

        self.tensor_size = self.Block3.tensor_size

        if activation in ("maxo", "rmxo"):  # switch to retain out_channels
            activation = "relu"
        self.Activation = Activations(activation, out_channels, **kwgs)

    def forward(self, tensor):
        if self.dropout is not None:
            tensor = self.dropout(tensor)
        residue = self.edit_residue(tensor) if hasattr(self, "edit_residue") \
            else tensor
        tensor = self.Block3(self.Block2(self.Shuffle(self.Block1(tensor))))
        return self.Activation(tensor + residue)
# =========================================================================== #


class SimpleFire(nn.Module):
    r"""Fire block for SqueezeNet support. All args are similar to Convolution.
    Implemented - https://arxiv.org/pdf/1602.07360.pdf
    """
    def __init__(self, tensor_size, filter_size, out_channels, strides=1,
                 pad=True, activation="relu", dropout=0., normalization=None,
                 pre_nm=False, groups=1, weight_nm=False, equalized=False,
                 shift=False, bias=False, dropblock=True, **kwargs):
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
                 shift=False, bias=False, dropblock=True, growth_rate=32,
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

        if check_strides(strides):
            if isinstance(carry_network, str):
                self.network2 = nn.AvgPool2d((3, 3), stride=(2, 2), padding=1)\
                    if carry_network.lower() == "avg" else \
                    nn.MaxPool2d((3, 3), stride=(2, 2), padding=1)
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
                 shift=False, bias=False, dropblock=True,
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
        if check_strides(strides):  # Update tensor_size
            tensor_size[0] = 1
            tensor_size = list(F.avg_pool2d(torch.rand(*tensor_size),
                                            3, 2, 1).size())
            self.pool = nn.AvgPool2d(3, 2, 1)

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


# from tensormonk.layers import Convolution
# from tensormonk.activations import Activations
# from tensormonk.regularizations import DropOut
# from tensormonk.layers.utils import check_strides, check_residue,\
#     update_kwargs
# tensor_size = (3, 64, 10, 10)
# x = torch.rand(*tensor_size)
# test = ResidualOriginal(tensor_size, 3, 64, 2, False, "relu", 0.,
#                         "batch", False)
# test(x).size()
# %timeit test(x).size()
# test = ResidualComplex(tensor_size, 3, 64, 2, False, "relu", 0., "batch",
#                        False)
# test(x).size()
# %timeit test(x).size()
# test = ResidualInverted(tensor_size, 3, 96, 1, False, "relu", 0., "batch",
#                         False)
# test(x).size()
# %timeit test(x).size()
# test = ResidualShuffle(tensor_size, 3, 64, 2, False, "relu", 0., "batch",
#                        False)
# test(x).size()
# %timeit test(x).size()
# test = SimpleFire(tensor_size, 3, 64, 2, False, "relu", 0.1, None, False)
# test(x).size()
# %timeit test(x).size()
# test = CarryModular(tensor_size, 3, 128, 2, False, "relu", 0., None, False)
# test(x).size()
# %timeit test(x).size()
# test = SEResidualComplex(tensor_size, 3, 64, 2, False, "relu", 0., "batch",
#                          False)
# test(x).size()
# %timeit test(x).size()
# test = ResidualNeXt(tensor_size, 3, 64, 2, False, "relu", 0., "batch", False)
# test(x).size()
# %timeit test(x).size()
# test = SEResidualNeXt(tensor_size, 3, 64, 2, False, "relu", 0., "batch",
#                       False)
# test(x).size()
# %timeit test(x).size()
# test = DenseBlock(tensor_size, 3, 128, 2, True, "relu", 0., "batch", False)
# test(x).size()
# %timeit test(x).size()
# test = ContextNet_Bottleneck(tensor_size, 3, 128, 1)
# test(x).size()
# %timeit test(x).size()
