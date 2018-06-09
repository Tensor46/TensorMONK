""" TensorMONK's :: NeuralLayers :: CarryResidue                          """

__all__ = ["ResidualOriginal", "ResidualComplex", "ResidualComplex2", "ResidualInverted",
           "ResidualShuffle", "SimpleFire", "CarryModular"]

import torch
import torch.nn as nn
from .Convolution import Convolution
# ============================================================================ #


class BaseBlock(nn.Module):
    """ Any residual or carry class forward! """
    def forward(self, tensor):
        if hasattr(self, "pre_network"): # for dropout & SqueezeNet Fire
            tensor = self.pre_network(tensor)

        if hasattr(self, "skip_residue") and self.skip_residue: # For MobileNET, strides>1
            tensor = self.network(tensor)
            if hasattr(self, "Activation"):
                return self.Activation(tensor)
            return tensor

        if hasattr(self, "edit_residue"):
            residue = self.edit_residue(tensor)
        elif hasattr(self, "edit_carry"):
            residue = self.edit_carry(tensor)
        else:
            residue = tensor
        tensor = self.network(tensor)
        if residue.size(1)+tensor.size(1) == self.tensor_size[1] or (hasattr(self, "dynamic") and self.dynamic): # CarryOver
            if hasattr(self, "dynamic") and self.dynamic:
                if hasattr(self, "Activation"):
                    return self.Activation(torch.cat( (residue[:, :-tensor.size(1)], tensor), 1))
                return torch.cat( (residue[:, :-tensor.size(1)], tensor), 1)
            else:
                if hasattr(self, "Activation"):
                    return self.Activation(torch.cat( (residue, tensor), 1))
                return torch.cat( (residue, tensor), 1)
        else: # Residual
            if hasattr(self, "Activation"):
                return self.Activation(tensor + residue)
            return tensor + residue
# ============================================================================ #


class ResidualOriginal(BaseBlock):
    def __init__(self, tensor_size, filter_size, out_channels, strides=(1, 1), pad=True,
                 activation="relu", dropout=0., batch_nm=False, pre_nm=False,
                 groups=1, weight_norm=False, *args, **kwargs):
        super(ResidualOriginal, self).__init__()
        if dropout > 0.:
            self.pre_network = nn.Dropout2d(dropout)
        self.network = nn.Sequential()
        self.network.add_module("Block3x3_1", Convolution(tensor_size, filter_size, out_channels, strides, True, activation, 0., batch_nm, pre_nm, groups, weight_norm))
        self.network.add_module("Block3x3_2", Convolution(self.network[-1].tensor_size, filter_size, out_channels, (1, 1), True, activation, 0., batch_nm, pre_nm, groups, weight_norm))
        if (strides > 1 if isinstance(strides, int) else (strides[0] > 1 or strides[1] > 1)) or tensor_size[1] != out_channels:
            _filter_size = (3, 3) if (strides > 1 if isinstance(strides, int) else (strides[0] > 1 or strides[1] > 1)) else (1, 1)
            self.edit_residue = Convolution(tensor_size, _filter_size, out_channels, strides, True, "", 0., batch_nm, pre_nm, groups, weight_norm)
        self.tensor_size = self.network[-1].tensor_size
# ============================================================================ #


class ResidualComplex(BaseBlock):
    def __init__(self, tensor_size, filter_size, out_channels, strides=(1, 1), pad=True,
                 activation="relu", dropout=0., batch_nm=False, pre_nm=False,
                 groups=1, weight_norm=False, *args, **kwargs):
        super(ResidualComplex, self).__init__()
        if dropout > 0.:
            self.pre_network = nn.Dropout2d(dropout)
        self.network = nn.Sequential()
        self.network.add_module("Block1x1/4", Convolution(tensor_size, (1, 1), out_channels//4, (1, 1), True, activation, 0., batch_nm, pre_nm, groups, weight_norm))
        self.network.add_module("Block3x3/4", Convolution(self.network[-1].tensor_size, filter_size, out_channels//4, strides, True, activation, 0., batch_nm, pre_nm, groups, weight_norm))
        self.network.add_module("Block1x1", Convolution(self.network[-1].tensor_size, (1, 1), out_channels, (1, 1), True, activation, 0., batch_nm, pre_nm, groups, weight_norm))

        if (strides > 1 if isinstance(strides, int) else (strides[0] > 1 or strides[1] > 1)) or tensor_size[1] != out_channels:
            _filter_size = (3, 3) if (strides > 1 if isinstance(strides, int) else (strides[0] > 1 or strides[1] > 1)) else (1, 1)
            self.edit_residue = Convolution(tensor_size, _filter_size, out_channels, strides, True, "", 0., batch_nm, pre_nm, groups, weight_norm)
        self.tensor_size = self.network[-1].tensor_size
# ============================================================================ #


class ResidualComplex2(BaseBlock):
    def __init__(self, tensor_size, filter_size, out_channels, strides=(1, 1), pad=True,
                 activation="relu", dropout=0., batch_nm=False, pre_nm=False,
                 groups=1, weight_norm=False, *args, **kwargs):
        super(ResidualComplex2, self).__init__()
        if dropout > 0.:
            self.pre_network = nn.Dropout2d(dropout)
        self.network = nn.Sequential()
        self.network.add_module("Block1x1/4", Convolution(tensor_size, (1, 1), out_channels//4, (1, 1), True, activation, 0., batch_nm, pre_nm, groups, weight_norm))
        self.network.add_module("Block3x3/4", Convolution(self.network[-1].tensor_size, filter_size, out_channels//4, strides, True, activation, 0., batch_nm, pre_nm, groups, weight_norm))
        self.network.add_module("Block1x1", Convolution(self.network[-1].tensor_size, (1, 1), out_channels, (1, 1), True, "", 0., batch_nm, pre_nm, groups, weight_norm))

        if (strides > 1 if isinstance(strides, int) else (strides[0] > 1 or strides[1] > 1)) or tensor_size[1] != out_channels:
            _filter_size = (3, 3) if (strides > 1 if isinstance(strides, int) else (strides[0] > 1 or strides[1] > 1)) else (1, 1)
            self.edit_residue = Convolution(tensor_size, _filter_size, out_channels, strides, True, "", 0., batch_nm, pre_nm, groups, weight_norm)
        self.tensor_size = self.network[-1].tensor_size

        if activation == "relu":
            self.Activation = nn.ReLU()
        if activation == "lklu":
            self.Activation = nn.LeakyReLU()
        if activation == "tanh":
            self.Activation = nn.Tanh()
        if activation == "sigm":
            self.Activation = nn.Sigmoid()
# ============================================================================ #


class ResidualInverted(BaseBlock):
    """ MobileNetV2 supporting block - https://arxiv.org/pdf/1801.04381.pdf """
    def __init__(self, tensor_size, filter_size, out_channels, strides=(1, 1), pad=True,
                 activation="relu6", dropout=0., batch_nm=False, pre_nm=False,
                 groups=1, weight_norm=False, t=1, *args, **kwargs):
        super(ResidualInverted, self).__init__()
        if dropout > 0.:
            self.pre_network = nn.Dropout2d(dropout)
        self.network = nn.Sequential()
        self.network.add_module("Block1x1pre", Convolution(tensor_size, (1, 1), out_channels*t, (1, 1), True, activation, 0., batch_nm, pre_nm, groups, weight_norm))
        self.network.add_module("Block3x3", Convolution(self.network[-1].tensor_size, filter_size, out_channels*t, strides,
                                                        True, activation, 0., batch_nm, pre_nm, out_channels*t, weight_norm))
        self.network.add_module("Block1x1pst", Convolution(self.network[-1].tensor_size, (1, 1), out_channels, (1, 1), True, "", 0., batch_nm, pre_nm, groups, weight_norm))

        self.skip_residue = False
        if (strides > 1 if isinstance(strides, int) else (strides[0] > 1 or strides[1] > 1)):
            self.skip_residue = True
        if not self.skip_residue and tensor_size[1] != out_channels:
            self.edit_residue = Convolution(tensor_size, (1, 1), out_channels, strides, True, "", 0., batch_nm, pre_nm, groups, weight_norm)
        self.tensor_size = self.network[-1].tensor_size
# ============================================================================ #


class ChannelShuffle(nn.Module):
    """ https://arxiv.org/pdf/1707.01083.pdf """
    def __init__(self, groups, *args, **kwargs):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, tensor):
        tensor_size = tensor.size()
        tensor = tensor.view(tensor_size[0], self.groups, -1, tensor_size[2], tensor_size[3])
        tensor = tensor.transpose_(2, 1).contiguous()
        return tensor.view(tensor_size[0], -1, tensor_size[2], tensor_size[3]).contiguous()


class ResidualShuffle(BaseBlock):
    """ ShuffleNet supporting block - https://arxiv.org/pdf/1707.01083.pdf """
    def __init__(self, tensor_size, filter_size, out_channels, strides=(1, 1), pad=True,
                 activation="relu", dropout=0., batch_nm=False, pre_nm=False,
                 groups=4, weight_norm=False,*args, **kwargs):
        super(ResidualShuffle, self).__init__()
        if dropout > 0.:
            self.pre_network = nn.Dropout2d(dropout)
        self.network = nn.Sequential()
        self.network.add_module("Block1x1pre", Convolution(tensor_size, (1, 1), out_channels, (1, 1), True, activation, 0., batch_nm, pre_nm, groups, weight_norm))
        self.network.add_module("Shuffle", ChannelShuffle(groups))
        self.network.add_module("Block3x3", Convolution(self.network[-2].tensor_size, filter_size, out_channels, strides, True, "", 0., batch_nm, pre_nm, groups, weight_norm))
        self.network.add_module("Block1x1pst", Convolution(self.network[-1].tensor_size, (1, 1), out_channels, (1, 1), True, "", 0., batch_nm, pre_nm, groups, weight_norm))

        self.edit_residue = nn.Sequential()
        if (strides > 1 if isinstance(strides, int) else (strides[0] > 1 or strides[1] > 1)):
            self.edit_residue.add_module("AveragePOOL", nn.AvgPool2d((3, 3), stride=(2, 2), padding=1))
        if tensor_size[1] != out_channels:
            self.edit_residue.add_module("Block1x1AdjustDepth", Convolution(tensor_size, (1, 1), out_channels, (1, 1), True, "", 0., batch_nm, pre_nm, groups, weight_norm))
        self.tensor_size = self.network[-1].tensor_size

        if activation == "relu":
            self.Activation = nn.ReLU()
        if activation == "lklu":
            self.Activation = nn.LeakyReLU()
        if activation == "tanh":
            self.Activation = nn.Tanh()
        if activation == "sigm":
            self.Activation = nn.Sigmoid()
# ============================================================================ #


class SimpleFire(BaseBlock):
    """ SqueezeNet supporting block - https://arxiv.org/pdf/1602.07360.pdf """
    def __init__(self, tensor_size, filter_size, out_channels, strides=(1, 1), pad=True,
                 activation="relu", dropout=0., batch_nm=False, pre_nm=False,
                 groups=1, weight_norm=False, *args, **kwargs):
        super(SimpleFire, self).__init__()
        self.pre_network = nn.Sequential()
        if dropout > 0.:
            self.pre_network.add_module("DropOut", nn.Dropout2d(dropout))
        self.pre_network.add_module("Block1x1Shrink", Convolution(tensor_size, (1, 1), out_channels//4, (1, 1), True, "", 0., batch_nm, pre_nm, groups, weight_norm))

        self.network = Convolution(self.pre_network[-1].tensor_size, filter_size, out_channels//2, strides, True, activation, 0., batch_nm, pre_nm, groups, weight_norm)
        self.edit_carry = Convolution(self.pre_network[-1].tensor_size, (1, 1), out_channels//2, strides, True, activation, 0., batch_nm, pre_nm, groups, weight_norm)

        self.tensor_size = (self.network.tensor_size[0], self.network.tensor_size[1]*2,
                            self.network.tensor_size[2], self.network.tensor_size[3])
# ============================================================================ #


class CarryModular(BaseBlock):
    def __init__(self, tensor_size, filter_size, out_channels, strides=(1, 1), pad=True,
                 activation="relu", dropout=0., batch_nm=False, pre_nm=False,
                 groups=1, weight_norm=False, growth_rate=32,
                 block=SimpleFire, *args, **kwargs):
        super(CarryModular, self).__init__()
        if dropout > 0.:
            self.pre_network = nn.Dropout2d(dropout)
        if tensor_size[1] < out_channels:
            growth_rate = out_channels - tensor_size[1]
        else:
            self.dynamic = True
        self.network = block(tensor_size, filter_size, growth_rate, strides, pad, activation, 0., batch_nm, pre_nm, groups, weight_norm)
        if (strides > 1 if isinstance(strides, int) else (strides[0] > 1 or strides[1] > 1)):
            self.edit_carry = nn.AvgPool2d((3, 3), stride=(2, 2), padding=1)
        self.tensor_size = (self.network.tensor_size[0], out_channels,
                            self.network.tensor_size[2], self.network.tensor_size[3])
# ============================================================================ #



# from core.NeuralLayers import Convolution
# tensor_size = (3,64,10,10)
# x = torch.rand(*tensor_size)
#
# test = ResidualOriginal(tensor_size, 3, 64, 2, False, "relu", 0., True, False)
# test(x).size()
# %timeit test(x).size()
# test = ResidualComplex(tensor_size, 3, 64, 2, False, "relu", 0., True, False)
# test(x).size()
# %timeit test(x).size()
# test = ResidualComplex2(tensor_size, 3, 64, 2, False, "relu", 0., True, False)
# test(x).size()
# %timeit test(x).size()
# test = ResidualInverted(tensor_size, 3, 64, 2, False, "relu", 0., True, False)
# test(x).size()
# %timeit test(x).size()
# test = ResidualShuffle(tensor_size, 3, 64, 2, False, "relu", 0., True, False)
# test(x).size()
# %timeit test(x).size()
# test = SimpleFire(tensor_size, 3, 64, 2, False, "relu", 0., False, False)
# test(x).size()
# %timeit test(x).size()
# test = CarryFire(tensor_size, 3, 128, 2, False, "relu", 0., False, False)
# test(x).size()
# %timeit test(x).size()
#
# test = CarryFire(tensor_size, 3, 128, 1, False, "relu", 0., False, False)
# test

# tensor_size = (3,200,28,28)
# x = torch.rand(*tensor_size)
#
# test = ResidualShuffle(tensor_size, 3, 200, 1, True, "relu", 0., True, False, 2)
# test(x).size()
# %timeit test(x).size()
