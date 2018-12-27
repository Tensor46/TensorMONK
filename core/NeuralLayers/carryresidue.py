""" TensorMONK's :: NeuralLayers :: CarryResidue                            """

__all__ = ["ResidualOriginal", "ResidualComplex", "ResidualInverted",
           "ResidualShuffle", "ResidualNeXt",
           "SEResidualComplex", "SEResidualNeXt",
           "SimpleFire", "CarryModular", "DenseBlock",
           "Stem2", "InceptionA", "InceptionB", "InceptionC",
           "ReductionA", "ReductionB",
           "ContextNet_Bottleneck"]

import torch
import torch.nn as nn
import torch.nn.functional as F
from .convolution import Convolution
from .activations import Activations


def check_strides(strides):
    return (strides > 1 if isinstance(strides, int) else
            (strides[0] > 1 or strides[1] > 1))


def check_residue(strides, t_size, out_channels):
    return check_strides(strides) or t_size[1] != out_channels


def update_kwargs(kwargs, *args):
    if len(args) > 0 and args[0] is not None:
        kwargs["tensor_size"] = args[0]
    if len(args) > 1 and args[1] is not None:
        kwargs["filter_size"] = args[1]
    if len(args) > 2 and args[2] is not None:
        kwargs["out_channels"] = args[2]
    if len(args) > 3 and args[3] is not None:
        kwargs["strides"] = args[3]
    if len(args) > 4 and args[4] is not None:
        kwargs["pad"] = args[4]
    if len(args) > 5 and args[5] is not None:
        kwargs["activation"] = args[5]
    if len(args) > 6 and args[6] is not None:
        kwargs["dropout"] = args[6]
    if len(args) > 7 and args[7] is not None:
        kwargs["normalization"] = args[7]
    if len(args) > 8 and args[8] is not None:
        kwargs["pre_nm"] = args[8]
    if len(args) > 9 and args[9] is not None:
        kwargs["groups"] = args[9]
    if len(args) > 10 and args[10] is not None:
        kwargs["weight_nm"] = args[10]
    if len(args) > 11 and args[11] is not None:
        kwargs["equalized"] = args[11]
    if len(args) > 12 and args[12] is not None:
        kwargs["shift"] = args[12]
    return kwargs


class ResidualOriginal(nn.Module):
    r""" Residual block with two 3x3 convolutions - used in ResNet18 and
    ResNet34. All args are similar to Convolution.
    """
    def __init__(self, tensor_size, filter_size, out_channels, strides=1,
                 pad=True, activation="relu", dropout=0., normalization=None,
                 pre_nm=False, groups=1, weight_nm=False, equalized=False,
                 shift=False, *args, **kwargs):

        super(ResidualOriginal, self).__init__()
        if dropout > 0.:
            self.pre_network = nn.Dropout2d(dropout)

        kwargs = update_kwargs(kwargs, tensor_size, filter_size, out_channels,
                               strides, True, activation, 0., normalization,
                               pre_nm, groups, weight_nm, equalized, shift)
        self.network = nn.Sequential()
        self.network.add_module("Block3x3_1", Convolution(**kwargs))
        kwargs["tensor_size"] = self.network[-1].tensor_size
        kwargs["strides"] = 1
        self.network.add_module("Block3x3_2", Convolution(**kwargs))

        if check_residue(strides, tensor_size, out_channels):
            kwargs["tensor_size"] = tensor_size
            kwargs["filter_size"] = 1
            kwargs["strides"] = strides
            kwargs["activation"] = ""
            self.edit_residue = Convolution(**kwargs)
        self.tensor_size = self.network[-1].tensor_size

    def forward(self, tensor):
        if hasattr(self, "pre_network"):  # for dropout
            tensor = self.pre_network(tensor)
        residue = self.edit_residue(tensor) if hasattr(self, "edit_residue") \
            else tensor
        return self.network(tensor) + residue
# =========================================================================== #


class ResidualComplex(nn.Module):
    r"""Bottleneck Residual block with 1x1 (out_channels//4), 3x3
    (out_channels//4), and 1x1 (out_channels) convolution - used in ResNet50,
    ResNet101, and ResNet152. All args are similar to Convolution.

    Args:
        late_activation: when True, activation is applied after residual
            addition = activation(tensor + residue). And the residue
            filter_size is 3 when strides > 1. default = False
    """
    def __init__(self, tensor_size, filter_size, out_channels, strides=1,
                 pad=True, activation="relu", dropout=0., normalization=None,
                 pre_nm=False, groups=1, weight_nm=False, equalized=False,
                 shift=False, late_activation=False, *args, **kwargs):
        super(ResidualComplex, self).__init__()
        if dropout > 0.:
            self.pre_network = nn.Dropout2d(dropout)

        kwargs = update_kwargs(kwargs, tensor_size, filter_size, out_channels,
                               strides, True, activation, 0., normalization,
                               pre_nm, groups, weight_nm, equalized, shift)

        self.network = nn.Sequential()
        kwargs["filter_size"] = 1
        kwargs["out_channels"] = out_channels//4
        kwargs["strides"] = 1
        self.network.add_module("Block1x1/4", Convolution(**kwargs))
        kwargs["tensor_size"] = self.network[-1].tensor_size
        kwargs["filter_size"] = filter_size
        kwargs["out_channels"] = out_channels//4
        kwargs["strides"] = strides
        self.network.add_module("Block3x3/4", Convolution(**kwargs))
        kwargs["tensor_size"] = self.network[-1].tensor_size
        kwargs["filter_size"] = 1
        kwargs["out_channels"] = out_channels
        kwargs["strides"] = 1
        if late_activation:
            kwargs["activation"] = ""
            self.Activation = Activations("relu" if activation in
                                          ("maxo", "rmxo") else activation)
        self.network.add_module("Block1x1", Convolution(**kwargs))
        if check_residue(strides, tensor_size, out_channels):
            kwargs["tensor_size"] = tensor_size
            kwargs["strides"] = strides
            kwargs["activation"] = ""
            if late_activation:
                kwargs["filter_size"] = 3
            self.edit_residue = Convolution(**kwargs)
        self.tensor_size = self.network[-1].tensor_size

    def forward(self, tensor):
        if hasattr(self, "pre_network"):  # for dropout
            tensor = self.pre_network(tensor)
        residue = self.edit_residue(tensor) if hasattr(self, "edit_residue") \
            else tensor
        if hasattr(self, "Activation"):  # late_activation
            return self.Activation(self.network(tensor) + residue)
        return self.network(tensor) + residue
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
                 shift=False, r=16, *args, **kwargs):
        super(SEResidualComplex, self).__init__()
        if dropout > 0.:
            self.pre_network = nn.Dropout2d(dropout)

        kwargs = update_kwargs(kwargs, tensor_size, filter_size, out_channels,
                               strides, True, activation, 0., normalization,
                               pre_nm, groups, weight_nm, equalized, shift)

        self.network = nn.Sequential()
        kwargs["filter_size"] = 1
        kwargs["out_channels"] = out_channels//4
        kwargs["strides"] = 1
        self.network.add_module("Block1x1/4", Convolution(**kwargs))
        kwargs["tensor_size"] = self.network[-1].tensor_size
        kwargs["filter_size"] = filter_size
        kwargs["out_channels"] = out_channels//4
        kwargs["strides"] = strides
        self.network.add_module("Block3x3/4", Convolution(**kwargs))
        kwargs["tensor_size"] = self.network[-1].tensor_size
        kwargs["filter_size"] = 1
        kwargs["out_channels"] = out_channels
        kwargs["strides"] = 1
        self.network.add_module("Block1x1", Convolution(**kwargs))

        se = [nn.AvgPool2d(self.network[-1].tensor_size[2:], stride=(1, 1)),
              Convolution((1, out_channels, 1, 1), 1, out_channels//r, 1,
                          False, "relu"),
              Convolution((1, out_channels//r, 1, 1), 1, out_channels, 1,
                          False, "sigm")]
        self.SqueezeExcitation = nn.Sequential(*se)

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
        residue = self.edit_residue(tensor) if hasattr(self, "edit_residue") \
            else tensor
        tensor = self.network(tensor)
        return tensor * self.SqueezeExcitation(tensor) + residue
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
                 shift=False, *args, **kwargs):
        super(ResidualNeXt, self).__init__()
        if dropout > 0.:
            self.pre_network = nn.Dropout2d(dropout)

        kwargs = update_kwargs(kwargs, None, None, None, None, True,
                               activation, 0., normalization, pre_nm, None,
                               weight_nm, equalized, shift)

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
        if hasattr(self, "pre_network"):  # for dropout
            tensor = self.pre_network(tensor)
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
                 shift=False, r=16, *args, **kwargs):
        super(SEResidualNeXt, self).__init__()
        if dropout > 0.:
            self.pre_network = nn.Dropout2d(dropout)

        kwargs = update_kwargs(kwargs, None, None, None, None, True,
                               activation, 0., normalization, pre_nm, None,
                               weight_nm, equalized, shift)

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
        if hasattr(self, "pre_network"):  # for dropout
            tensor = self.pre_network(tensor)
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
                 shift=False, t=1, *args, **kwargs):
        super(ResidualInverted, self).__init__()
        if dropout > 0.:
            self.pre_network = nn.Dropout2d(dropout)

        kwargs = update_kwargs(kwargs, None, None, None, None, True,
                               activation, 0., normalization, pre_nm, None,
                               weight_nm, equalized, shift)

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
        if hasattr(self, "pre_network"):  # for dropout
            tensor = self.pre_network(tensor)
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
                 shift=False, *args, **kwargs):
        super(ResidualShuffle, self).__init__()
        if dropout > 0.:
            self.pre_network = nn.Dropout2d(dropout)
        kwargs = update_kwargs(kwargs, None, None, out_channels, None,
                               True, activation, 0., normalization,
                               pre_nm, groups, weight_nm, equalized, shift)

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
                                 Convolution(t_size, 1, out_channels,
                                             **kwargs)]
            self.edit_residue = nn.Sequential(*self.edit_residue)

        self.tensor_size = self.Block3.tensor_size

        if activation in ("maxo", "rmxo"):  # switch to retain out_channels
            activation = "relu"
        self.Activation = Activations(activation)

    def forward(self, tensor):
        if hasattr(self, "pre_network"):  # for dropout
            tensor = self.pre_network(tensor)
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
                 shift=False, *args, **kwargs):
        super(SimpleFire, self).__init__()
        if dropout > 0.:
            self.dropout = nn.Dropout2d(dropout)
        kwargs = update_kwargs(kwargs, None, None, None,
                               None, True, None, 0., normalization,
                               pre_nm, groups, weight_nm, equalized, shift)

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
        if hasattr(self, "dropout"):
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
                 shift=False, growth_rate=32, block=SimpleFire,
                 carry_network="avg", *args, **kwargs):
        super(CarryModular, self).__init__()
        pad = True
        if dropout > 0.:
            self.pre_network = nn.Dropout2d(dropout)
        if tensor_size[1] < out_channels:  # adjusts growth_rate
            growth_rate = out_channels - tensor_size[1]
        else:
            self.dynamic = True

        self.network1 = block \
            if isinstance(block, torch.nn.modules.container.Sequential) else \
            block(tensor_size, filter_size, growth_rate, strides, pad,
                  activation, 0., normalization, pre_nm, groups, weight_nm,
                  equalized, shift, **kwargs)

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
                 shift=False, growth_rate=16, block=Convolution, n_blocks=4,
                 multiplier=4, *args, **kwargs):
        super(DenseBlock, self).__init__()
        assert out_channels == tensor_size[1] + growth_rate * n_blocks, \
            "DenseBlock -- out_channels != tensor_size[1]+growth_rate*n_blocks"
        kwargs = update_kwargs(kwargs, None, None, None,
                               None, True, activation, 0., normalization,
                               pre_nm, groups, weight_nm, equalized, shift)
        if dropout > 0.:
            self.dropout = nn.Dropout2d(dropout)

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
        if hasattr(self, "dropout"):
            tensor = self.dropout(tensor)
        if hasattr(self, "pool"):
            tensor = self.pool(tensor)

        for n in range(1, self.n_blocks+1):
            tensor = torch.cat((tensor,
                                getattr(self, "block"+str(n))(tensor)), 1)
        return tensor
# =========================================================================== #


class Stem2(nn.Module):
    r""" For InceptionV4 - https://arxiv.org/pdf/1602.07261.pdf
    All args are similar to Convolution and shift is disabled. Designed for an
    input tensor_size of (1, 3, 299, 299) to deliver an output of size
    (1, 384, 35, 35)
    """
    def __init__(self, tensor_size=(1, 3, 299, 299), activation="relu",
                 normalization="batch", pre_nm=False, groups=1,
                 weight_nm=False, equalized=False, shift=False,
                 *args, **kwargs):
        super(Stem2, self).__init__()
        kwargs = update_kwargs(kwargs, None, None, None,
                               None, None, activation, 0., normalization,
                               pre_nm, groups, weight_nm, equalized, shift)

        self.C3_32_2 = Convolution(tensor_size, 3, 32, 2, False, **kwargs)
        self.C3_32_1 = Convolution(self.C3_32_2.tensor_size, 3, 32, 1, False,
                                   **kwargs)
        self.C3_64_1 = Convolution(self.C3_32_1.tensor_size, 3, 64, 1, True,
                                   **kwargs)
        self.C160 = CarryModular(self.C3_64_1.tensor_size, 3, 160, 2, False,
                                 block=Convolution, pool="max", **kwargs)

        channel1 = nn.Sequential()
        channel1.add_module("C1_64_1", Convolution(self.C160.tensor_size, 1,
                                                   64, 1,  True, **kwargs))
        channel1.add_module("C17_64_1", Convolution(channel1[-1].tensor_size,
                                                    (1, 7), 64, 1,  True,
                                                    **kwargs))
        channel1.add_module("C71_64_1", Convolution(channel1[-1].tensor_size,
                                                    (7, 1), 64, 1, True,
                                                    **kwargs))
        channel1.add_module("C3_96_1", Convolution(channel1[-1].tensor_size, 3,
                                                   96, 1, False, **kwargs))

        channel2 = nn.Sequential()
        channel2.add_module("C1_64_1", Convolution(self.C160.tensor_size, 1,
                                                   64, 1,  True, **kwargs))
        channel2.add_module("C3_96_1", Convolution(channel2[-1].tensor_size, 3,
                                                   96, 1, False, **kwargs))
        self.C192 = CarryModular(self.C160.tensor_size, 3, 192, 2, False,
                                 block=channel1, carry_network=channel2,
                                 **kwargs)
        self.C384 = CarryModular(self.C192.tensor_size, 3, 384, 2, False,
                                 block=Convolution, pool="max", **kwargs)

        self.tensor_size = self.C384.tensor_size

    def forward(self, tensor):
        tensor = self.C3_64_1(self.C3_32_1(self.C3_32_2(tensor)))
        tensor = self.C160(tensor)
        return self.C384(self.C192(tensor))
# =========================================================================== #


class InceptionA(nn.Module):
    r""" For InceptionV4 - https://arxiv.org/pdf/1602.07261.pdf
    All args are similar to Convolution and shift is disabled. Designed for an
    input tensor_size of (1, 384, 35, 35) to deliver an output of size
    (1, 384, 35, 35)
    """
    def __init__(self, tensor_size=(1, 384, 35, 35), activation="relu",
                 normalization="batch", pre_nm=False, groups=1,
                 weight_nm=False, equalized=False, *args, **kwargs):
        super(InceptionA, self).__init__()
        h, w = tensor_size[2:]

        kwargs = update_kwargs(kwargs, None, None, None,
                               None, True, activation, 0., normalization,
                               pre_nm, groups, weight_nm, equalized, None)

        self.path1 = nn.Sequential(nn.AvgPool2d((3, 3), (1, 1), 1),
                                   Convolution(tensor_size, 1, 96, 1,
                                               **kwargs))
        self.path2 = Convolution(tensor_size, 1, 96, 1, **kwargs)
        path3 = [Convolution(tensor_size, 1, 64, 1, **kwargs),
                 Convolution((1, 64, h, w), 3, 96, 1, **kwargs)]
        self.path3 = nn.Sequential(*path3)
        path4 = [Convolution(tensor_size, 1, 64, 1, **kwargs),
                 Convolution((1, 64, h, w), 3, 96, 1, **kwargs),
                 Convolution((1, 96, h, w), 3, 96, 1, **kwargs)]
        self.path4 = nn.Sequential(*path4)
        self.tensor_size = (1, 96*4, h, w)

    def forward(self, tensor):
        return torch.cat((self.path1(tensor), self.path2(tensor),
                          self.path3(tensor), self.path4(tensor)), 1)
# =========================================================================== #


class ReductionA(nn.Module):
    r""" For InceptionV4 - https://arxiv.org/pdf/1602.07261.pdf
    All args are similar to Convolution and shift is disabled. Designed for an
    input tensor_size of (1, 384, 35, 35) to deliver an output of size
    (1, 1024, 17, 17)
    """
    def __init__(self, tensor_size=(1, 384, 35, 35), activation="relu",
                 normalization="batch", pre_nm=False, groups=1,
                 weight_nm=False, equalized=False, *args, **kwargs):
        super(ReductionA, self).__init__()
        h, w = tensor_size[2:]
        kwargs = update_kwargs(kwargs, None, None, None,
                               None, None, activation, 0., normalization,
                               pre_nm, groups, weight_nm, equalized, None)

        self.path1 = nn.MaxPool2d((3, 3), stride=(2, 2))
        self.path2 = Convolution(tensor_size, 3, 384, 2, False, **kwargs)
        path3 = [Convolution(tensor_size, 1, 192, 1, True, **kwargs),
                 Convolution((1, 192, h, w), 3, 224, 1, True, **kwargs),
                 Convolution((1, 224, h, w), 3, 256, 2, False, **kwargs)]
        self.path3 = nn.Sequential(*path3)

        self.tensor_size = (1, tensor_size[1]+384+256,
                            self.path2.tensor_size[2],
                            self.path2.tensor_size[3])

    def forward(self, tensor):
        return torch.cat((self.path1(tensor), self.path2(tensor),
                          self.path3(tensor)), 1)
# =========================================================================== #


class InceptionB(nn.Module):
    r""" For InceptionV4 - https://arxiv.org/pdf/1602.07261.pdf
    All args are similar to Convolution and shift is disabled. Designed for an
    input tensor_size of (1, 1024, 17, 17) to deliver an output of size
    (1, 1024, 17, 17)
    """
    def __init__(self, tensor_size=(1, 1024, 17, 17), activation="relu",
                 normalization="batch", pre_nm=False, groups=1,
                 weight_nm=False, equalized=False, *args, **kwargs):
        super(InceptionB, self).__init__()
        h, w = tensor_size[2:]
        kwargs = update_kwargs(kwargs, None, None, None,
                               None, True, activation, 0., normalization,
                               pre_nm, groups, weight_nm, equalized, None)

        self.path1 = nn.Sequential(nn.AvgPool2d((3, 3), (1, 1), 1),
                                   Convolution(tensor_size, 1, 128, 1,
                                               **kwargs))
        self.path2 = Convolution(tensor_size, 1, 384, 1, **kwargs)
        path3 = [Convolution(tensor_size, 1, 192, 1, **kwargs),
                 Convolution((1, 192, h, w), (1, 7), 224, 1, **kwargs),
                 Convolution((1, 224, h, w), (1, 7), 256, 1, **kwargs)]
        self.path3 = nn.Sequential(*path3)
        path4 = [Convolution(tensor_size, 1, 192, 1, **kwargs),
                 Convolution((1, 192, h, w), (1, 7), 192, 1, **kwargs),
                 Convolution((1, 192, h, w), (7, 1), 224, 1, **kwargs),
                 Convolution((1, 224, h, w), (1, 7), 224, 1, **kwargs),
                 Convolution((1, 224, h, w), (7, 1), 256, 1, **kwargs)]
        self.path4 = nn.Sequential(*path4)

        self.tensor_size = (1, 128+384+256+256, h, w)

    def forward(self, tensor):
        return torch.cat((self.path1(tensor), self.path2(tensor),
                          self.path3(tensor), self.path4(tensor)), 1)
# =========================================================================== #


class ReductionB(nn.Module):
    r""" For InceptionV4 - https://arxiv.org/pdf/1602.07261.pdf
    All args are similar to Convolution and shift is disabled. Designed for an
    input tensor_size of (1, 1024, 17, 17) to deliver an output of size
    (1, 1536, 8, 8)
    """
    def __init__(self, tensor_size=(1, 1024, 17, 17), activation="relu",
                 normalization="batch", pre_nm=False, groups=1,
                 weight_nm=False, equalized=False, *args, **kwargs):
        super(ReductionB, self).__init__()
        h, w = tensor_size[2:]
        kwargs = update_kwargs(kwargs, None, None, None,
                               None, None, activation, 0., normalization,
                               pre_nm, groups, weight_nm, equalized, None)

        self.path1 = nn.MaxPool2d((3, 3), stride=(2, 2))
        path2 = [Convolution(tensor_size, 1, 192, 1, True, **kwargs),
                 Convolution((1, 192, h, w), 3, 192, 2, False, **kwargs)]
        self.path2 = nn.Sequential(*path2)
        path3 = [Convolution(tensor_size, 1, 256, 1, True, **kwargs),
                 Convolution((1, 256, h, w), (1, 7), 256, 1, True, **kwargs),
                 Convolution((1, 256, h, w), (1, 7), 256, 1, True, **kwargs),
                 Convolution((1, 256, h, w), 3, 320, 2, False, **kwargs)]
        self.path3 = nn.Sequential(*path3)

        self.tensor_size = (1, tensor_size[1]+192+320,
                            self.path2[-1].tensor_size[2],
                            self.path2[-1].tensor_size[3])

    def forward(self, tensor):
        return torch.cat((self.path1(tensor), self.path2(tensor),
                          self.path3(tensor)), 1)
# =========================================================================== #


class InceptionC(nn.Module):
    r""" For InceptionV4 - https://arxiv.org/pdf/1602.07261.pdf
    All args are similar to Convolution and shift is disabled. Designed for an
    input tensor_size of (1, 1536, 8, 8) to deliver an output of size
    (1, 1536, 8, 8)
    """
    def __init__(self, tensor_size=(1, 1536, 8, 8), activation="relu",
                 normalization="batch", pre_nm=False, groups=1,
                 weight_nm=False, equalized=False, *args, **kwargs):
        super(InceptionC, self).__init__()
        h, w = tensor_size[2:]
        kwargs = update_kwargs(kwargs, None, None, None,
                               None, True, activation, 0., normalization,
                               pre_nm, groups, weight_nm, equalized, None)

        self.path1 = nn.Sequential(nn.AvgPool2d((3, 3), (1, 1), 1),
                                   Convolution(tensor_size, 1, 256, 1,
                                               **kwargs))
        self.path2 = Convolution(tensor_size, 1, 256, 1, **kwargs)
        self.path3 = Convolution(tensor_size, 1, 384, 1, **kwargs)
        self.path3a = Convolution(self.path3.tensor_size, (1, 3), 256, 1,
                                  **kwargs)
        self.path3b = Convolution(self.path3.tensor_size, (3, 1), 256, 1,
                                  **kwargs)
        path4 = [Convolution(tensor_size, 1, 384, 1, **kwargs),
                 Convolution((1, 384, h, w), (1, 3), 448, 1, **kwargs),
                 Convolution((1, 448, h, w), (3, 1), 512, 1, **kwargs)]
        self.path4 = nn.Sequential(*path4)
        self.path4a = Convolution(self.path4[-1].tensor_size, (1, 3), 256, 1,
                                  **kwargs)
        self.path4b = Convolution(self.path4[-1].tensor_size, (3, 1), 256, 1,
                                  **kwargs)
        self.tensor_size = (1, 256+256+512+512, h, w)

    def forward(self, tensor):
        path3 = self.path3(tensor)
        path4 = self.path4(tensor)
        return torch.cat((self.path1(tensor), self.path2(tensor),
                          self.path3a(path3), self.path3b(path3),
                          self.path4a(path4), self.path4b(path4)), 1)
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


# from core.NeuralLayers import Convolution, Activations
# tensor_size = (3,3,299,299)
# x = torch.rand(*tensor_size)
# test = Stem2(tensor_size, "relu", "batch", False)
# test(x).size()
# %timeit test(x).size()
# test = InceptionA()
# test(torch.rand(*(1, 384, 35, 35))).size()
# test = ReductionA()
# test(torch.rand(*(1, 384, 35, 35))).size()
# test = InceptionB((1, 1024, 17, 17))
# test(torch.rand(*(1, 1024, 17, 17))).size()
# test = ReductionB((1, 1024, 17, 17))
# test(torch.rand(*(1, 1024, 17, 17))).size()
# test = InceptionC((1, 1536, 8, 8))
# test(torch.rand(*(1, 1536, 8, 8))).size()

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
# test = SEResidualComplex(tensor_size, 3, 64, 1, False, "relu", 0., "batch",
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
