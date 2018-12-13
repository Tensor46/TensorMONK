""" TensorMONK's :: NeuralLayers :: CarryResidue                          """

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
# ============================================================================ #
check_strides = lambda s: (s > 1 if isinstance(s, int) else (s[0] > 1 or s[1] > 1))
check_residue = lambda s, ts, oc: check_strides(s) or ts[1] != oc


def update_kwargs(*args, **kwargs):
    if len(args) >  0: kwargs["tensor_size"] = args[0]
    if len(args) >  1: kwargs["filter_size"] = args[1]
    if len(args) >  2: kwargs["out_channels"] = args[2]
    if len(args) >  3: kwargs["strides"] = args[3]
    if len(args) >  4: kwargs["pad"] = args[4]
    if len(args) >  5: kwargs["activation"] = args[5]
    if len(args) >  6: kwargs["dropout"] = args[6]
    if len(args) >  7: kwargs["normalization"] = args[7]
    if len(args) >  8: kwargs["pre_nm"] = args[8]
    if len(args) >  9: kwargs["groups"] = args[9]
    if len(args) > 10: kwargs["weight_nm"] = args[10]
    if len(args) > 11: kwargs["equalized"] = args[11]
    if len(args) > 12: kwargs["shift"] = args[12]
    return kwargs


class ResidualOriginal(nn.Module):
    def __init__(self, tensor_size, filter_size, out_channels, strides=1,
            pad=True, activation="relu", dropout=0., normalization=None,
            pre_nm=False, groups=1, weight_nm=False, equalized=False,
            shift=False, *args, **kwargs):

        super(ResidualOriginal, self).__init__()
        if dropout > 0.:
            self.pre_network = nn.Dropout2d(dropout)

        kwargs = update_kwargs(tensor_size, filter_size, out_channels,
            strides, True, activation, 0., normalization, pre_nm, groups,
            weight_nm, equalized, shift, **kwargs)

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
        if hasattr(self, "pre_network"): # for dropout
            tensor = self.pre_network(tensor)
        residue = self.edit_residue(tensor) if hasattr(self, "edit_residue") else tensor
        return self.network(tensor) + residue
# ============================================================================ #


class ResidualComplex(nn.Module):
    def __init__(self, tensor_size, filter_size, out_channels, strides=1,
            pad=True, activation="relu", dropout=0., normalization=None,
            pre_nm=False, groups=1, weight_nm=False, equalized=False,
            shift=False, late_activation=False, *args, **kwargs):
        super(ResidualComplex, self).__init__()
        if dropout > 0.:
            self.pre_network = nn.Dropout2d(dropout)

        kwargs = update_kwargs(tensor_size, filter_size, out_channels,
            strides, True, activation, 0., normalization, pre_nm, groups,
            weight_nm, equalized, shift, **kwargs)

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
            """
                Similar to ResidualComplex in paper, other than
                activation(tensor + residue) and residue filter_size is 3
                when strides > 1
            """
            kwargs["activation"] = ""
            self.Activation = Activations("relu" if activation in \
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
        if hasattr(self, "pre_network"): # for dropout
            tensor = self.pre_network(tensor)
        residue = self.edit_residue(tensor) if hasattr(self, "edit_residue") else tensor
        if hasattr(self, "Activation"): # late_activation
            return self.Activation(self.network(tensor) + residue)
        return self.network(tensor) + residue
# ============================================================================ #


class SEResidualComplex(nn.Module):
    """ Squeeze-and-Excitation ResidualComplex -
            https://arxiv.org/pdf/1709.01507.pdf """
    def __init__(self, tensor_size, filter_size, out_channels, strides=1,
            pad=True, activation="relu", dropout=0., normalization=None,
            pre_nm=False, groups=1, weight_nm=False, equalized=False,
            shift=False, r=16, *args, **kwargs):
        super(SEResidualComplex, self).__init__()
        if dropout > 0.:
            self.pre_network = nn.Dropout2d(dropout)

        kwargs = update_kwargs(tensor_size, filter_size, out_channels,
            strides, True, activation, 0., normalization, pre_nm, groups,
            weight_nm, equalized, shift, **kwargs)

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

        self.SqueezeExcitation = nn.Sequential(
            nn.AvgPool2d(self.network[-1].tensor_size[2:], stride=(1, 1)),
            Convolution((1, out_channels, 1, 1), 1, out_channels//r, 1, False, "relu"),
            Convolution((1, out_channels//r, 1, 1), 1, out_channels, 1, False, "sigm"))

        if check_residue(strides, tensor_size, out_channels):
            kwargs["tensor_size"] = tensor_size
            kwargs["filter_size"] = 1
            kwargs["out_channels"] = out_channels
            kwargs["strides"] = strides
            kwargs["activation"] = ""
            self.edit_residue = Convolution(**kwargs)
        self.tensor_size = self.network[-1].tensor_size

    def forward(self, tensor):
        if hasattr(self, "pre_network"): # for dropout
            tensor = self.pre_network(tensor)
        residue = self.edit_residue(tensor) if hasattr(self, "edit_residue") else tensor
        tensor = self.network(tensor)
        return tensor * self.SqueezeExcitation(tensor) + residue
# ============================================================================ #


class ResidualNeXt(nn.Module):
    """ ResNeXt module -- https://arxiv.org/pdf/1611.05431.pdf """
    def __init__(self, tensor_size, filter_size, out_channels, strides=1,
            pad=True, activation="relu", dropout=0., normalization=None,
            pre_nm=False, groups=32, weight_nm=False, equalized=False,
            shift=False, *args, **kwargs):
        super(ResidualNeXt, self).__init__()
        if dropout > 0.:
            self.pre_network = nn.Dropout2d(dropout)

        kwargs = update_kwargs(tensor_size, filter_size, out_channels,
            strides, True, activation, 0., normalization, pre_nm, groups,
            weight_nm, equalized, shift, **kwargs)

        self.network = nn.Sequential()
        kwargs["filter_size"] = 1
        kwargs["out_channels"] = out_channels//2
        kwargs["strides"] = 1
        kwargs["groups"] = 1
        self.network.add_module("Block1x1/2", Convolution(**kwargs))
        kwargs["tensor_size"] = self.network[-1].tensor_size
        kwargs["filter_size"] = filter_size
        kwargs["strides"] = strides
        kwargs["groups"] = groups
        self.network.add_module("Block3x3/2", Convolution(**kwargs))
        kwargs["tensor_size"] = self.network[-1].tensor_size
        kwargs["filter_size"] = 1
        kwargs["out_channels"] = out_channels
        kwargs["strides"] = 1
        kwargs["groups"] = 1
        self.network.add_module("Block1x1", Convolution(**kwargs))
        if check_residue(strides, tensor_size, out_channels):
            kwargs["tensor_size"] = tensor_size
            kwargs["strides"] = strides
            kwargs["activation"] = ""
            self.edit_residue = Convolution(**kwargs)
        self.tensor_size = self.network[-1].tensor_size

    def forward(self, tensor):
        if hasattr(self, "pre_network"): # for dropout
            tensor = self.pre_network(tensor)
        residue = self.edit_residue(tensor) if hasattr(self, "edit_residue") else tensor
        return self.network(tensor) + residue
# ============================================================================ #


class SEResidualNeXt(nn.Module):
    """
        Squeeze-and-Excitation + ResNeXt
        https://arxiv.org/pdf/1709.01507.pdf
        https://arxiv.org/pdf/1611.05431.pdf
    """
    def __init__(self, tensor_size, filter_size, out_channels, strides=1,
            pad=True, activation="relu", dropout=0., normalization=None,
            pre_nm=False, groups=1, weight_nm=False, equalized=False,
            shift=False, r=16, *args, **kwargs):
        super(SEResidualNeXt, self).__init__()
        if dropout > 0.:
            self.pre_network = nn.Dropout2d(dropout)

        kwargs = update_kwargs(tensor_size, filter_size, out_channels,
            strides, True, activation, 0., normalization, pre_nm, groups,
            weight_nm, equalized, shift, **kwargs)

        self.network = nn.Sequential()
        kwargs["filter_size"] = 1
        kwargs["out_channels"] = out_channels//2
        kwargs["strides"] = 1
        kwargs["groups"] = 1
        self.network.add_module("Block1x1/2", Convolution(**kwargs))
        kwargs["tensor_size"] = self.network[-1].tensor_size
        kwargs["filter_size"] = filter_size
        kwargs["strides"] = strides
        kwargs["groups"] = groups
        self.network.add_module("Block3x3/2", Convolution(**kwargs))
        kwargs["tensor_size"] = self.network[-1].tensor_size
        kwargs["filter_size"] = 1
        kwargs["out_channels"] = out_channels
        kwargs["strides"] = 1
        kwargs["groups"] = 1
        self.network.add_module("Block1x1", Convolution(**kwargs))

        self.SqueezeExcitation = nn.Sequential(
            nn.AvgPool2d(self.network[-1].tensor_size[2:], stride=(1, 1)),
            Convolution((1, out_channels, 1, 1), 1, out_channels//r, 1, False, "relu"),
            Convolution((1, out_channels//r, 1, 1), 1, out_channels, 1, False, "sigm"))

        if check_residue(strides, tensor_size, out_channels):
            kwargs["tensor_size"] = tensor_size
            kwargs["strides"] = strides
            kwargs["activation"] = ""
            self.edit_residue = Convolution(**kwargs)
        self.tensor_size = self.network[-1].tensor_size

    def forward(self, tensor):
        if hasattr(self, "pre_network"): # for dropout
            tensor = self.pre_network(tensor)
        residue = self.edit_residue(tensor) if hasattr(self, "edit_residue") else tensor
        tensor = self.network(tensor)
        return tensor * self.SqueezeExcitation(tensor) + residue
# ============================================================================ #


class ResidualInverted(nn.Module):
    """ MobileNetV2 supporting block - https://arxiv.org/pdf/1801.04381.pdf """
    def __init__(self, tensor_size, filter_size, out_channels, strides=1,
            pad=True, activation="relu6", dropout=0., normalization=None,
            pre_nm=False, groups=1, weight_nm=False, equalized=False,
            shift=False, t=1, *args, **kwargs):
        super(ResidualInverted, self).__init__()
        if dropout > 0.:
            self.pre_network = nn.Dropout2d(dropout)

        kwargs = update_kwargs(tensor_size, filter_size, out_channels,
            strides, True, activation, 0., normalization, pre_nm, groups,
            weight_nm, equalized, shift, **kwargs)

        self.network = nn.Sequential()
        kwargs["filter_size"] = 1
        kwargs["out_channels"] = out_channels*t
        kwargs["strides"] = 1
        self.network.add_module("Block1x1pre", Convolution(**kwargs))
        kwargs["tensor_size"] = self.network[-1].tensor_size
        kwargs["filter_size"] = filter_size
        kwargs["out_channels"] = out_channels*t
        kwargs["strides"] = strides
        kwargs["groups"] = out_channels*t
        self.network.add_module("Block3x3", Convolution(**kwargs))
        kwargs["tensor_size"] = self.network[-1].tensor_size
        kwargs["filter_size"] = 1
        kwargs["out_channels"] = out_channels
        kwargs["strides"] = 1
        kwargs["activation"] = ""
        kwargs["groups"] = 1
        self.network.add_module("Block1x1pst", Convolution(**kwargs))
        self.skip_residue = True if check_strides(strides) else False
        if not self.skip_residue and tensor_size[1] != out_channels:
            kwargs["tensor_size"] = tensor_size
            kwargs["strides"] = strides
            self.edit_residue = Convolution(**kwargs)
        self.tensor_size = self.network[-1].tensor_size

    def forward(self, tensor):
        if hasattr(self, "pre_network"): # for dropout
            tensor = self.pre_network(tensor)
        if self.skip_residue: # For strides > 1
            return self.network(tensor)
        residue = self.edit_residue(tensor) if hasattr(self, "edit_residue") else tensor
        return self.network(tensor) + residue
# ============================================================================ #


class ChannelShuffle(nn.Module):
    """ https://arxiv.org/pdf/1707.01083.pdf """
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
    """ ShuffleNet supporting block - https://arxiv.org/pdf/1707.01083.pdf """
    def __init__(self, tensor_size, filter_size, out_channels, strides=1,
            pad=True, activation="relu", dropout=0., normalization=None,
            pre_nm=False, groups=4, weight_nm=False, equalized=False,
            shift=False, *args, **kwargs):
        super(ResidualShuffle, self).__init__()
        if dropout > 0.:
            self.pre_network = nn.Dropout2d(dropout)

        kwargs = update_kwargs(tensor_size, filter_size, out_channels,
            strides, True, activation, 0., normalization, pre_nm, groups,
            weight_nm, equalized, shift, **kwargs)

        self.network = nn.Sequential()
        kwargs["filter_size"] = 1
        kwargs["strides"] = 1
        self.network.add_module("Block1x1pre", Convolution(**kwargs))
        self.network.add_module("Shuffle", ChannelShuffle(groups))
        kwargs["tensor_size"] = self.network[-2].tensor_size
        kwargs["filter_size"] = filter_size
        kwargs["strides"] = strides
        kwargs["activation"] = ""
        self.network.add_module("Block3x3", Convolution(**kwargs))
        kwargs["tensor_size"] = self.network[-1].tensor_size
        kwargs["filter_size"] = 1
        kwargs["strides"] = 1
        self.network.add_module("Block1x1pst", Convolution(**kwargs))
        self.edit_residue = nn.Sequential()
        if check_strides(strides):
            self.edit_residue.add_module("AveragePOOL",
                nn.AvgPool2d((3, 3), stride=(2, 2), padding=1))
        if tensor_size[1] != out_channels:
            kwargs["tensor_size"] = tensor_size
            self.edit_residue.add_module("AdjustDepth", Convolution(**kwargs))
        self.tensor_size = self.network[-1].tensor_size

        self.Activation = Activations("relu" if activation in \
            ("maxo", "rmxo") else activation)

    def forward(self, tensor):
        if hasattr(self, "pre_network"): # for dropout
            tensor = self.pre_network(tensor)
        return self.Activation(self.network(tensor) + self.edit_residue(tensor))
# ============================================================================ #


class SimpleFire(nn.Module):
    """ SqueezeNet supporting block - https://arxiv.org/pdf/1602.07360.pdf """
    def __init__(self, tensor_size, filter_size, out_channels, strides=1,
            pad=True, activation="relu", dropout=0., normalization=None,
            pre_nm=False, groups=1, weight_nm=False, equalized=False,
            shift=False, *args, **kwargs):
        super(SimpleFire, self).__init__()
        self.pre_network = nn.Sequential()
        if dropout > 0.:
            self.pre_network.add_module("DropOut", nn.Dropout2d(dropout))

        kwargs = update_kwargs(tensor_size, filter_size, out_channels,
            strides, True, activation, 0., normalization, pre_nm, groups,
            weight_nm, equalized, shift, **kwargs)

        kwargs["filter_size"] = 1
        kwargs["strides"] = 1
        kwargs["out_channels"] = out_channels//4
        kwargs["activation"] = ""
        self.pre_network.add_module("Block1x1Shrink", Convolution(**kwargs))

        kwargs["tensor_size"] = self.pre_network[-1].tensor_size
        kwargs["filter_size"] = filter_size
        kwargs["out_channels"] = out_channels//2
        kwargs["strides"] = strides
        kwargs["activation"] = activation
        self.network1 = Convolution(**kwargs)
        kwargs["filter_size"] = 1
        self.network2 = Convolution(**kwargs)
        self.tensor_size = (self.network1.tensor_size[0], self.network1.tensor_size[1]*2,
                            self.network1.tensor_size[2], self.network1.tensor_size[3])

    def forward(self, tensor):
        tensor = self.pre_network(tensor)
        return torch.cat((self.network1(tensor), self.network2(tensor)), 1)
# ============================================================================ #


class CarryModular(nn.Module):
    def __init__(self, tensor_size, filter_size, out_channels, strides=1,
            pad=True, activation="relu", dropout=0., normalization=None,
            pre_nm=False, groups=1, weight_nm=False, equalized=False,
            shift=False, growth_rate=32, block=SimpleFire, carry_network="avg",
            *args, **kwargs):
        super(CarryModular, self).__init__()
        pad = True
        if dropout > 0.:
            self.pre_network = nn.Dropout2d(dropout)
        if tensor_size[1] < out_channels: # adjusts growth_rate
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
                self.network2 = nn.AvgPool2d((3, 3), stride=(2, 2), padding=1) \
                    if carry_network.lower() == "avg" else \
                    nn.MaxPool2d((3, 3), stride=(2, 2), padding=1)
            elif isinstance(carry_network, list) or isinstance(carry_network, tuple):
                self.network2 = nn.Sequential(*carry_network)
            elif isinstance(carry_network, torch.nn.modules.container.Sequential):
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
        if hasattr(self, "pre_network"): # for dropout
            tensor = self.pre_network(tensor)
        return torch.cat((self.network1(tensor), self.network2(tensor)), 1)
# ============================================================================ #


class DenseBlock(nn.Module):
    def __init__(self, tensor_size, filter_size, out_channels, strides=1,
            pad=True, activation="relu", dropout=0., normalization=None,
            pre_nm=False, groups=1, weight_nm=False, equalized=False,
            shift=False, growth_rate=16, block=Convolution, n_blocks=4,
            multiplier=4, *args, **kwargs):
        super(DenseBlock, self).__init__()

        assert out_channels == tensor_size[1] + growth_rate * n_blocks, \
            "DenseBlock -- out_channels != tensor_size[1]+growth_rate*n_blocks"
        pad = True
        tensor_size = list(tensor_size)
        if check_strides(strides): # Update tensor_size
            tensor_size[0] = 1
            tensor_size = list(F.avg_pool2d(torch.rand(*tensor_size), 3, 2, 1).size())

        if dropout > 0. and not check_strides(strides):
            self.pre_network = nn.Dropout2d(dropout)
        elif dropout == 0. and check_strides(strides):
            self.pre_network = nn.AvgPool2d(3, 2, padding=1)
        elif dropout > 0. and check_strides(strides):
            self.pre_network = nn.Sequential(nn.Dropout2d(dropout),
                nn.AvgPool2d(3, 2, padding=1))

        self.blocks = nn.ModuleDict()
        for n in range(1, n_blocks+1):
            self.blocks.update({"block-"+str(n):
                nn.Sequential(block(tensor_size, 1, growth_rate*multiplier, 1,
                True, activation, 0., normalization, pre_nm, groups, weight_nm,
                equalized, shift, **kwargs),
                block((1, growth_rate*multiplier, tensor_size[2], tensor_size[3]),
                filter_size, growth_rate, 1, True, activation, 0., normalization,
                pre_nm, groups, weight_nm, equalized, shift, **kwargs))})
            tensor_size[1] += growth_rate
        self.tensor_size = (tensor_size[0], out_channels, tensor_size[2],
            tensor_size[3])

    def forward(self, tensor):
        if hasattr(self, "pre_network"): # for dropout and strides
            tensor = self.pre_network(tensor)

        for module in self.blocks.values():
            tensor = torch.cat((tensor, module(tensor)), 1)
        return tensor
# ============================================================================ #


class Stem2(nn.Module):
    """ For InceptionV4 and InceptionResNetV2 - https://arxiv.org/pdf/1602.07261.pdf """
    def __init__(self, tensor_size=(1, 3, 299, 299), activation="relu", normalization="batch",
                 pre_nm=False, groups=1, weight_nm=False, equalized=False, *args, **kwargs):
        super(Stem2, self).__init__()

        self.C3_32_2 = Convolution(tensor_size, 3, 32, 2, False, activation, 0.,
            normalization, pre_nm, 1, weight_nm, equalized)
        self.C3_32_1 = Convolution(self.C3_32_2.tensor_size, 3, 32, 1, False,
            activation, 0., normalization, pre_nm, 1, weight_nm, equalized)
        self.C3_64_1 = Convolution(self.C3_32_1.tensor_size, 3, 64, 1, True,
            activation, 0., normalization, pre_nm, 1, weight_nm, equalized)

        self.C160 = CarryModular(self.C3_64_1.tensor_size, 3, 160, 2, False,
            activation, 0., normalization, pre_nm, 1, weight_nm,
            block=Convolution, pool="max")

        channel1 = nn.Sequential()
        channel1.add_module("C1_64_1", Convolution(self.C160.tensor_size, 1,
            64, 1,  True, activation, 0.,normalization, pre_nm, 1, weight_nm, equalized))
        channel1.add_module("C17_64_1", Convolution(channel1[-1].tensor_size, (1, 7),
            64, 1,  True, activation, 0., normalization, pre_nm, 1, weight_nm, equalized))
        channel1.add_module("C71_64_1", Convolution(channel1[-1].tensor_size, (7, 1),
            64, 1,  True, activation, 0., normalization, pre_nm, 1, weight_nm, equalized))
        channel1.add_module("C3_96_1", Convolution(channel1[-1].tensor_size, 3,
            96, 1, False, activation, 0., normalization, pre_nm, 1, weight_nm, equalized))

        channel2 = nn.Sequential()
        channel2.add_module("C1_64_1", Convolution(self.C160.tensor_size, 1,
            64, 1,  True, activation, 0., normalization, pre_nm, 1, weight_nm, equalized))
        channel2.add_module("C3_96_1", Convolution(channel2[-1].tensor_size, 3,
            96, 1, False, activation, 0., normalization, pre_nm, 1, weight_nm, equalized))

        self.C192 = CarryModular(self.C160.tensor_size, 3,
            192, 2, False, activation, 0., normalization, pre_nm, 1, weight_nm,
            block=channel1, carry_network=channel2)

        self.C384 = CarryModular(self.C192.tensor_size, 3,
            384, 2, False, activation, 0., normalization, pre_nm, 1, weight_nm,
            block=Convolution, pool="max")

        self.tensor_size = self.C384.tensor_size

    def forward(self, tensor):

        tensor = self.C3_64_1(self.C3_32_1(self.C3_32_2(tensor)))
        tensor = self.C160(tensor)
        return self.C384(self.C192(tensor))
# ============================================================================ #


class InceptionA(nn.Module):
    """ For InceptionV4 - https://arxiv.org/pdf/1602.07261.pdf """
    def __init__(self, tensor_size=(1, 384, 35, 35), activation="relu",
            normalization="batch", pre_nm=False, groups=1, weight_nm=False,
            equalized=False, *args, **kwargs):
        super(InceptionA, self).__init__()
        H, W = tensor_size[2:]

        kwargs["pad"] = True
        kwargs["activation"] = activation
        kwargs["dropout"] = 0.
        kwargs["normalization"] = normalization
        kwargs["pre_nm"] = pre_nm
        kwargs["groups"] = groups
        kwargs["weight_nm"] = weight_nm
        kwargs["equalized"] = equalized

        self.path1 = nn.Sequential(nn.AvgPool2d((3, 3), stride=(1, 1), padding=1),
            Convolution(tensor_size, 1, 96, 1, **kwargs))
        self.path2 = Convolution(tensor_size, 1, 96, 1, **kwargs)
        self.path3 = nn.Sequential(Convolution(tensor_size, 1, 64, 1, **kwargs),
            Convolution((1, 64, H, W), 3, 96, 1, **kwargs))
        self.path4 = nn.Sequential(Convolution(tensor_size, 1, 64, 1, **kwargs),
            Convolution((1, 64, H, W), 3, 96, 1, **kwargs),
            Convolution((1, 96, H, W), 3, 96, 1, **kwargs))

        self.tensor_size = (1, 96*4, H, W)

    def forward(self, tensor):
        return torch.cat((self.path1(tensor), self.path2(tensor), \
            self.path3(tensor), self.path4(tensor)), 1)
# ============================================================================ #


class ReductionA(nn.Module):
    """
        For InceptionV4 - https://arxiv.org/pdf/1602.07261.pdf
        Reduction from 35 to 17
    """
    def __init__(self, tensor_size=(1, 384, 35, 35), activation="relu",
            normalization="batch", pre_nm=False, groups=1, weight_nm=False,
            equalized=False, *args, **kwargs):
        super(ReductionA, self).__init__()
        H, W = tensor_size[2:]

        kwargs["activation"] = activation
        kwargs["dropout"] = 0.
        kwargs["normalization"] = normalization
        kwargs["pre_nm"] = pre_nm
        kwargs["groups"] = groups
        kwargs["weight_nm"] = weight_nm
        kwargs["equalized"] = equalized

        self.path1 = nn.MaxPool2d((3, 3), stride=(2, 2))
        self.path2 = Convolution(tensor_size, 3, 384, 2, False, **kwargs)
        self.path3 = nn.Sequential(Convolution(tensor_size, 1, 192, 1, True, **kwargs),
            Convolution((1, 192, H, W), 3, 224, 1, True, **kwargs),
            Convolution((1, 224, H, W), 3, 256, 2, False, **kwargs))

        self.tensor_size = (1, tensor_size[1]+384+256, \
            self.path2.tensor_size[2], self.path2.tensor_size[3])

    def forward(self, tensor):
        return torch.cat((self.path1(tensor), self.path2(tensor), \
            self.path3(tensor)), 1)
# ============================================================================ #


class InceptionB(nn.Module):
    """ For InceptionV4 - https://arxiv.org/pdf/1602.07261.pdf """
    def __init__(self, tensor_size=(1, 1024, 17, 17), activation="relu",
            normalization="batch", pre_nm=False, groups=1, weight_nm=False,
            equalized=False, *args, **kwargs):
        super(InceptionB, self).__init__()
        H, W = tensor_size[2:]

        kwargs["pad"] = True
        kwargs["activation"] = activation
        kwargs["dropout"] = 0.
        kwargs["normalization"] = normalization
        kwargs["pre_nm"] = pre_nm
        kwargs["groups"] = groups
        kwargs["weight_nm"] = weight_nm
        kwargs["equalized"] = equalized

        self.path1 = nn.Sequential(nn.AvgPool2d((3, 3), stride=(1, 1), padding=1),
            Convolution(tensor_size, 1, 128, 1, **kwargs))
        self.path2 = Convolution(tensor_size, 1, 384, 1, **kwargs)
        self.path3 = nn.Sequential(Convolution(tensor_size, 1, 192, 1, **kwargs),
            Convolution((1, 192, H, W), (1, 7), 224, 1, **kwargs),
            Convolution((1, 224, H, W), (1, 7), 256, 1, **kwargs))
        self.path4 = nn.Sequential(Convolution(tensor_size, 1, 192, 1, **kwargs),
            Convolution((1, 192, H, W), (1, 7), 192, 1, **kwargs),
            Convolution((1, 192, H, W), (7, 1), 224, 1, **kwargs),
            Convolution((1, 224, H, W), (1, 7), 224, 1, **kwargs),
            Convolution((1, 224, H, W), (7, 1), 256, 1, **kwargs))

        self.tensor_size = (1, 128+384+256+256, H, W)

    def forward(self, tensor):
        return torch.cat((self.path1(tensor), self.path2(tensor), \
            self.path3(tensor), self.path4(tensor)), 1)
# ============================================================================ #


class ReductionB(nn.Module):
    """
        For InceptionV4 - https://arxiv.org/pdf/1602.07261.pdf
        Reduction from 17 to 8
    """
    def __init__(self, tensor_size=(1, 1024, 17, 17), activation="relu",
            normalization="batch", pre_nm=False, groups=1, weight_nm=False,
            equalized=False, *args, **kwargs):
        super(ReductionB, self).__init__()
        H, W = tensor_size[2:]

        kwargs["activation"] = activation
        kwargs["dropout"] = 0.
        kwargs["normalization"] = normalization
        kwargs["pre_nm"] = pre_nm
        kwargs["groups"] = groups
        kwargs["weight_nm"] = weight_nm
        kwargs["equalized"] = equalized

        self.path1 = nn.MaxPool2d((3, 3), stride=(2, 2))
        self.path2 = nn.Sequential(Convolution(tensor_size, 1, 192, 1, True, **kwargs),
            Convolution((1, 192, H, W), 3, 192, 2, False, **kwargs))
        self.path3 = nn.Sequential(Convolution(tensor_size, 1, 256, 1, True, **kwargs),
            Convolution((1, 256, H, W), (1, 7), 256, 1, True, **kwargs),
            Convolution((1, 256, H, W), (7, 1), 320, 1, True, **kwargs),
            Convolution((1, 320, H, W), 3, 320, 2, False, **kwargs))

        self.tensor_size = (1, tensor_size[1]+192+320, \
            self.path2[-1].tensor_size[2], self.path2[-1].tensor_size[3])

    def forward(self, tensor):
        return torch.cat((self.path1(tensor), self.path2(tensor), self.path3(tensor)), 1)
# ============================================================================ #


class InceptionC(nn.Module):
    """ For InceptionV4 - https://arxiv.org/pdf/1602.07261.pdf """
    def __init__(self, tensor_size=(1, 1536, 8, 8), activation="relu",
            normalization="batch", pre_nm=False, groups=1, weight_nm=False,
            equalized=False, *args, **kwargs):
        super(InceptionC, self).__init__()
        H, W = tensor_size[2:]

        kwargs["pad"] = True
        kwargs["activation"] = activation
        kwargs["dropout"] = 0.
        kwargs["normalization"] = normalization
        kwargs["pre_nm"] = pre_nm
        kwargs["groups"] = groups
        kwargs["weight_nm"] = weight_nm
        kwargs["equalized"] = equalized

        self.path1 = nn.Sequential(nn.AvgPool2d((3, 3), stride=(1, 1), padding=1),
            Convolution(tensor_size, 1, 256, 1, **kwargs))
        self.path2 = Convolution(tensor_size, 1, 256, 1, **kwargs)
        self.path3 = Convolution(tensor_size, 1, 384, 1, **kwargs)
        self.path3a = Convolution(self.path3.tensor_size, (1, 3), 256, 1, **kwargs)
        self.path3b = Convolution(self.path3.tensor_size, (3, 1), 256, 1, **kwargs)
        self.path4 = nn.Sequential(Convolution(tensor_size, 1, 384, 1, **kwargs),
            Convolution((1, 384, H, W), (1, 3), 448, 1, **kwargs),
            Convolution((1, 448, H, W), (3, 1), 512, 1, **kwargs))
        self.path4a = Convolution(self.path4[-1].tensor_size, (1, 3), 256, 1, **kwargs)
        self.path4b = Convolution(self.path4[-1].tensor_size, (3, 1), 256, 1, **kwargs)

        self.tensor_size = (1, 256+256+512+512, H, W)

    def forward(self, tensor):
        path3 = self.path3(tensor)
        path4 = self.path4(tensor)
        return torch.cat((self.path1(tensor), self.path2(tensor), self.path3a(path3),
                          self.path3b(path3), self.path4a(path4), self.path4b(path4)), 1)
# ============================================================================ #


class ContextNet_Bottleneck(nn.Module):
    """ bottleneck for contextnet - https://arxiv.org/pdf/1805.04554.pdf - Table 1 """
    def __init__(self, tensor_size, filter_size, out_channels, strides=1,
            pad=True, activation="relu", dropout=0., normalization="batch",
            pre_nm=False, groups=1, weight_nm=False, equalized=False,
            shift=False, expansion=1, *args, **kwargs):
        super(ContextNet_Bottleneck, self).__init__()

        if dropout > 0.:
            self.pre_network = nn.Dropout2d(dropout)

        kwargs = update_kwargs(tensor_size, filter_size, out_channels,
            strides, True, activation, 0., normalization, pre_nm, groups,
            weight_nm, equalized, shift, **kwargs)

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
        kwargs["filter_size"] = 1 # cross check -- why name Block3x3_DW12?
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
        if hasattr(self, "pre_network"): # for dropout
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

# tensor_size = (3,64,10,10)
# x = torch.rand(*tensor_size)
# test = ResidualOriginal(tensor_size, 3, 64, 2, False, "relu", 0., "batch", False)
# test(x).size()
# %timeit test(x).size()
# test = ResidualComplex(tensor_size, 3, 64, 2, False, "relu", 0., "batch", False)
# test(x).size()
# %timeit test(x).size()
# test = ResidualInverted(tensor_size, 3, 64, 2, False, "relu", 0., "batch", False)
# test(x).size()
# %timeit test(x).size()
# test = ResidualShuffle(tensor_size, 3, 64, 2, False, "relu", 0., "batch", False)
# test(x).size()
# %timeit test(x).size()
# test = SimpleFire(tensor_size, 3, 64, 2, False, "relu", 0., None, False)
# test(x).size()
# %timeit test(x).size()
# test = CarryModular(tensor_size, 3, 128, 2, False, "relu", 0., None, False)
# test(x).size()
# %timeit test(x).size()
# test = SEResidualComplex(tensor_size, 3, 64, 1, False, "relu", 0., "batch", False)
# test(x).size()
# %timeit test(x).size()
# test = ResidualNeXt(tensor_size, 3, 64, 2, False, "relu", 0., "batch", False)
# test(x).size()
# %timeit test(x).size()
# test = SEResidualNeXt(tensor_size, 3, 64, 2, False, "relu", 0., "batch", False)
# test(x).size()
# %timeit test(x).size()
# test = DenseBlock(tensor_size, 3, 128, 2, True, "relu", 0., "batch", False)
# test(x).size()
# %timeit test(x).size()
# test = ContextNet_Bottleneck(tensor_size, 3, 128, 1)
# test(x).size()
# %timeit test(x).size()
