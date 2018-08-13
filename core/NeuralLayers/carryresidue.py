""" TensorMONK's :: NeuralLayers :: CarryResidue                          """

__all__ = ["ResidualOriginal", "ResidualComplex", "ResidualComplex2", "ResidualInverted",
           "ResidualShuffle", "ResidualNeXt",
           "SEResidualComplex", "SEResidualNeXt",
           "SimpleFire", "CarryModular",
           "Stem2", "InceptionA", "InceptionB", "InceptionC", "ReductionA", "ReductionB"]

import torch
import torch.nn as nn
from .convolution import Convolution
from .activations import Activations
# ============================================================================ #


class ResidualOriginal(nn.Module):
    def __init__(self, tensor_size, filter_size, out_channels, strides=(1, 1), pad=True,
                 activation="relu", dropout=0., normalization=None, pre_nm=False,
                 groups=1, weight_nm=False, equalized=False, *args, **kwargs):
        super(ResidualOriginal, self).__init__()
        if dropout > 0.:
            self.pre_network = nn.Dropout2d(dropout)
        self.network = nn.Sequential()
        self.network.add_module("Block3x3_1", Convolution(tensor_size, filter_size, out_channels, strides, True, activation,
                                                          0., normalization, pre_nm, groups, weight_nm, equalized, **kwargs))
        self.network.add_module("Block3x3_2", Convolution(self.network[-1].tensor_size, filter_size, out_channels, (1, 1), True, activation,
                                                          0., normalization, pre_nm, groups, weight_nm, equalized, **kwargs))
        if (strides > 1 if isinstance(strides, int) else (strides[0] > 1 or strides[1] > 1)) or tensor_size[1] != out_channels:
            # _filter_size = (3, 3) if (strides > 1 if isinstance(strides, int) else (strides[0] > 1 or strides[1] > 1)) else (1, 1)
            self.edit_residue = Convolution(tensor_size, 1, out_channels, strides, True, "",
                                            0., normalization, pre_nm, groups, weight_nm, equalized, **kwargs)
        self.tensor_size = self.network[-1].tensor_size

    def forward(self, tensor):
        if hasattr(self, "pre_network"): # for dropout
            tensor = self.pre_network(tensor)
        residue = self.edit_residue(tensor) if hasattr(self, "edit_residue") else tensor
        return self.network(tensor) + residue
# ============================================================================ #


class ResidualComplex(nn.Module):
    def __init__(self, tensor_size, filter_size, out_channels, strides=(1, 1), pad=True,
                 activation="relu", dropout=0., normalization=None, pre_nm=False,
                 groups=1, weight_nm=False, equalized=False, *args, **kwargs):
        super(ResidualComplex, self).__init__()
        if dropout > 0.:
            self.pre_network = nn.Dropout2d(dropout)
        self.network = nn.Sequential()
        self.network.add_module("Block1x1/4", Convolution(tensor_size, (1, 1), out_channels//4, (1, 1), True, activation,
                                                          0., normalization, pre_nm, groups, weight_nm, equalized, **kwargs))
        self.network.add_module("Block3x3/4", Convolution(self.network[-1].tensor_size, filter_size, out_channels//4, strides, True, activation,
                                                          0., normalization, pre_nm, groups, weight_nm, equalized, **kwargs))
        self.network.add_module("Block1x1", Convolution(self.network[-1].tensor_size, (1, 1), out_channels, (1, 1), True, activation,
                                                        0., normalization, pre_nm, groups, weight_nm, equalized, **kwargs))

        if (strides > 1 if isinstance(strides, int) else (strides[0] > 1 or strides[1] > 1)) or tensor_size[1] != out_channels:
            # _filter_size = (3, 3) if (strides > 1 if isinstance(strides, int) else (strides[0] > 1 or strides[1] > 1)) else (1, 1)
            self.edit_residue = Convolution(tensor_size, 1, out_channels, strides, True, "",
                                            0., normalization, pre_nm, groups, weight_nm, equalized, **kwargs)
        self.tensor_size = self.network[-1].tensor_size

    def forward(self, tensor):
        if hasattr(self, "pre_network"): # for dropout
            tensor = self.pre_network(tensor)
        residue = self.edit_residue(tensor) if hasattr(self, "edit_residue") else tensor
        return self.network(tensor) + residue
# ============================================================================ #


class SEResidualComplex(nn.Module):
    """ Squeeze-and-Excitation ResidualComplex - https://arxiv.org/pdf/1709.01507.pdf"""
    def __init__(self, tensor_size, filter_size, out_channels, strides=(1, 1), pad=True,
                 activation="relu", dropout=0., normalization=None, pre_nm=False,
                 groups=1, weight_nm=False, equalized=False, r=16, *args, **kwargs):
        super(SEResidualComplex, self).__init__()
        if dropout > 0.:
            self.pre_network = nn.Dropout2d(dropout)
        self.network = nn.Sequential()
        self.network.add_module("Block1x1/4", Convolution(tensor_size, (1, 1), out_channels//4, (1, 1), True, activation,
                                                          0., normalization, pre_nm, groups, weight_nm, equalized, **kwargs))
        self.network.add_module("Block3x3/4", Convolution(self.network[-1].tensor_size, filter_size, out_channels//4, strides, True, activation,
                                                          0., normalization, pre_nm, groups, weight_nm, equalized, **kwargs))
        self.network.add_module("Block1x1", Convolution(self.network[-1].tensor_size, (1, 1), out_channels, (1, 1), True, activation,
                                                        0., normalization, pre_nm, groups, weight_nm, equalized, **kwargs))

        self.SqueezeExcitation = nn.Sequential(nn.AvgPool2d(self.network[-1].tensor_size[2:], stride=(1, 1)),
                                               Convolution((1, out_channels, 1, 1), 1, out_channels//r, 1, False, "relu"),
                                               Convolution((1, out_channels//r, 1, 1), 1, out_channels, 1, False, "sigm"))

        if (strides > 1 if isinstance(strides, int) else (strides[0] > 1 or strides[1] > 1)) or tensor_size[1] != out_channels:
            # _filter_size = (3, 3) if (strides > 1 if isinstance(strides, int) else (strides[0] > 1 or strides[1] > 1)) else (1, 1)
            self.edit_residue = Convolution(tensor_size, 1, out_channels, strides, True, "",
                                            0., normalization, pre_nm, groups, weight_nm, equalized, **kwargs)
        self.tensor_size = self.network[-1].tensor_size

    def forward(self, tensor):
        if hasattr(self, "pre_network"): # for dropout
            tensor = self.pre_network(tensor)
        residue = self.edit_residue(tensor) if hasattr(self, "edit_residue") else tensor
        tensor = self.network(tensor)
        tensor = tensor * self.SqueezeExcitation(tensor)
        return tensor + residue
# ============================================================================ #


class ResidualNeXt(nn.Module):
    """ ResNeXt module -- https://arxiv.org/pdf/1611.05431.pdf """
    def __init__(self, tensor_size, filter_size, out_channels, strides=(1, 1), pad=True,
                 activation="relu", dropout=0., normalization=None, pre_nm=False,
                 groups=1, weight_nm=False, equalized=False, *args, **kwargs):
        super(ResidualNeXt, self).__init__()
        if dropout > 0.:
            self.pre_network = nn.Dropout2d(dropout)
        self.network = nn.Sequential()
        self.network.add_module("Block1x1/2", Convolution(tensor_size, (1, 1), out_channels//2, (1, 1), True, activation,
                                                          0., normalization, pre_nm, groups, weight_nm, equalized, **kwargs))
        self.network.add_module("Block3x3/2", Convolution(self.network[-1].tensor_size, filter_size, out_channels//2, strides, True, activation,
                                                          0., normalization, pre_nm, 32, weight_nm, equalized, **kwargs))
        self.network.add_module("Block1x1", Convolution(self.network[-1].tensor_size, (1, 1), out_channels, (1, 1), True, activation,
                                                        0., normalization, pre_nm, groups, weight_nm, equalized, **kwargs))

        if (strides > 1 if isinstance(strides, int) else (strides[0] > 1 or strides[1] > 1)) or tensor_size[1] != out_channels:
            # _filter_size, grps = (3, 2) if (strides > 1 if isinstance(strides, int) else (strides[0] > 1 or strides[1] > 1)) else (1, 1)
            self.edit_residue = Convolution(tensor_size, 1, out_channels, strides, True, "",
                                            0., normalization, pre_nm, 1, weight_nm, equalized, **kwargs)
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
    def __init__(self, tensor_size, filter_size, out_channels, strides=(1, 1), pad=True,
                 activation="relu", dropout=0., normalization=None, pre_nm=False,
                 groups=1, weight_nm=False, equalized=False, r=16, *args, **kwargs):
        super(SEResidualNeXt, self).__init__()
        if dropout > 0.:
            self.pre_network = nn.Dropout2d(dropout)
        self.network = nn.Sequential()
        self.network.add_module("Block1x1/2", Convolution(tensor_size, (1, 1), out_channels//2, (1, 1), True, activation,
                                                          0., normalization, pre_nm, groups, weight_nm, equalized, **kwargs))
        self.network.add_module("Block3x3/2", Convolution(self.network[-1].tensor_size, filter_size, out_channels//2, strides, True, activation,
                                                          0., normalization, pre_nm, 32, weight_nm, equalized, **kwargs))
        self.network.add_module("Block1x1", Convolution(self.network[-1].tensor_size, (1, 1), out_channels, (1, 1), True, activation,
                                                        0., normalization, pre_nm, groups, weight_nm, equalized, **kwargs))

        self.SqueezeExcitation = nn.Sequential(nn.AvgPool2d(self.network[-1].tensor_size[2:], stride=(1, 1)),
                                               Convolution((1, out_channels, 1, 1), 1, out_channels//r, 1, False, "relu"),
                                               Convolution((1, out_channels//r, 1, 1), 1, out_channels, 1, False, "sigm"))

        if (strides > 1 if isinstance(strides, int) else (strides[0] > 1 or strides[1] > 1)) or tensor_size[1] != out_channels:
            # _filter_size, grps = (3, 2) if (strides > 1 if isinstance(strides, int) else (strides[0] > 1 or strides[1] > 1)) else (1, 1)
            self.edit_residue = Convolution(tensor_size, 1, out_channels, strides, True, "",
                                            0., normalization, pre_nm, 1, weight_nm, equalized, **kwargs)
        self.tensor_size = self.network[-1].tensor_size

    def forward(self, tensor):
        if hasattr(self, "pre_network"): # for dropout
            tensor = self.pre_network(tensor)
        residue = self.edit_residue(tensor) if hasattr(self, "edit_residue") else tensor
        tensor = self.network(tensor)
        tensor = tensor * self.SqueezeExcitation(tensor)
        return tensor + residue
# ============================================================================ #


class ResidualComplex2(nn.Module):
    """
        Similar to ResidualComplex, other than activation(tensor + residue) and
        residue filter_size is 3 when strides > 1
    """
    def __init__(self, tensor_size, filter_size, out_channels, strides=(1, 1), pad=True,
                 activation="relu", dropout=0., normalization=None, pre_nm=False,
                 groups=1, weight_nm=False, equalized=False, *args, **kwargs):
        super(ResidualComplex2, self).__init__()
        if dropout > 0.:
            self.pre_network = nn.Dropout2d(dropout)
        self.network = nn.Sequential()
        self.network.add_module("Block1x1/4", Convolution(tensor_size, (1, 1), out_channels//4, (1, 1), True, activation,
                                                          0., normalization, pre_nm, groups, weight_nm, equalized, **kwargs))
        self.network.add_module("Block3x3/4", Convolution(self.network[-1].tensor_size, filter_size, out_channels//4, strides, True, activation,
                                                          0., normalization, pre_nm, groups, weight_nm, equalized, **kwargs))
        self.network.add_module("Block1x1", Convolution(self.network[-1].tensor_size, (1, 1), out_channels, (1, 1), True, "",
                                                        0., normalization, pre_nm, groups, weight_nm, equalized, **kwargs))

        if (strides > 1 if isinstance(strides, int) else (strides[0] > 1 or strides[1] > 1)) or tensor_size[1] != out_channels:
            _filter_size = (3, 3) if (strides > 1 if isinstance(strides, int) else (strides[0] > 1 or strides[1] > 1)) else (1, 1)
            self.edit_residue = Convolution(tensor_size, _filter_size, out_channels, strides, True, "",
                                            0., normalization, pre_nm, groups, weight_nm, equalized, **kwargs)
        self.tensor_size = self.network[-1].tensor_size
        self.Activation = Activations(activation)

    def forward(self, tensor):
        if hasattr(self, "pre_network"): # for dropout
            tensor = self.pre_network(tensor)
        residue = self.edit_residue(tensor) if hasattr(self, "edit_residue") else tensor
        if hasattr(self, "Activation"):
            return self.Activation(self.network(tensor) + residue)
        return self.network(tensor) + residue
# ============================================================================ #


class ResidualInverted(nn.Module):
    """ MobileNetV2 supporting block - https://arxiv.org/pdf/1801.04381.pdf """
    def __init__(self, tensor_size, filter_size, out_channels, strides=(1, 1), pad=True,
                 activation="relu6", dropout=0., normalization=None, pre_nm=False,
                 groups=1, weight_nm=False, equalized=False, t=1, *args, **kwargs):
        super(ResidualInverted, self).__init__()
        if dropout > 0.:
            self.pre_network = nn.Dropout2d(dropout)
        self.network = nn.Sequential()
        self.network.add_module("Block1x1pre", Convolution(tensor_size, (1, 1), out_channels*t, (1, 1), True, activation,
                                                           0., normalization, pre_nm, groups, weight_nm, equalized, **kwargs))
        self.network.add_module("Block3x3", Convolution(self.network[-1].tensor_size, filter_size, out_channels*t, strides, True,
                                                        activation, 0., normalization, pre_nm, out_channels*t, weight_nm, equalized, **kwargs))
        self.network.add_module("Block1x1pst", Convolution(self.network[-1].tensor_size, (1, 1), out_channels, (1, 1), True, "",
                                                           0., normalization, pre_nm, groups, weight_nm, equalized, **kwargs))

        self.skip_residue = False
        if (strides > 1 if isinstance(strides, int) else (strides[0] > 1 or strides[1] > 1)):
            self.skip_residue = True
        if not self.skip_residue and tensor_size[1] != out_channels:
            self.edit_residue = Convolution(tensor_size, 1, out_channels, strides, True, "",
                                            0., normalization, pre_nm, groups, weight_nm, equalized, **kwargs)
        self.tensor_size = self.network[-1].tensor_size

    def forward(self, tensor):
        if hasattr(self, "pre_network"): # for dropout
            tensor = self.pre_network(tensor)
        if self.skip_residue: # For strides>1
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
        tensor = tensor.view(tensor_size[0], self.groups, -1, tensor_size[2], tensor_size[3])
        tensor = tensor.transpose(2, 1).contiguous()
        return tensor.view(tensor_size[0], -1, tensor_size[2], tensor_size[3]).contiguous()


class ResidualShuffle(nn.Module):
    """ ShuffleNet supporting block - https://arxiv.org/pdf/1707.01083.pdf """
    def __init__(self, tensor_size, filter_size, out_channels, strides=(1, 1), pad=True,
                 activation="relu", dropout=0., normalization=None, pre_nm=False,
                 groups=4, weight_nm=False, equalized=False, *args, **kwargs):
        super(ResidualShuffle, self).__init__()
        if dropout > 0.:
            self.pre_network = nn.Dropout2d(dropout)
        self.network = nn.Sequential()
        self.network.add_module("Block1x1pre", Convolution(tensor_size, (1, 1), out_channels, (1, 1), True, activation,
                                                           0., normalization, pre_nm, groups, weight_nm, equalized, **kwargs))
        self.network.add_module("Shuffle", ChannelShuffle(groups))
        self.network.add_module("Block3x3", Convolution(self.network[-2].tensor_size, filter_size, out_channels, strides, True, "",
                                                        0., normalization, pre_nm, groups, weight_nm, equalized, **kwargs))
        self.network.add_module("Block1x1pst", Convolution(self.network[-1].tensor_size, (1, 1), out_channels, (1, 1), True, "",
                                                           0., normalization, pre_nm, groups, weight_nm, equalized, **kwargs))

        self.edit_residue = nn.Sequential()
        if (strides > 1 if isinstance(strides, int) else (strides[0] > 1 or strides[1] > 1)):
            self.edit_residue.add_module("AveragePOOL", nn.AvgPool2d((3, 3), stride=(2, 2), padding=1))
        if tensor_size[1] != out_channels:
            self.edit_residue.add_module("Block1x1AdjustDepth", Convolution(tensor_size, 1, out_channels, 1, True, "", 0.,
                                                                            normalization, pre_nm, groups, weight_nm, equalized, **kwargs))
        self.tensor_size = self.network[-1].tensor_size
        self.Activation = Activations(activation)

    def forward(self, tensor):
        if hasattr(self, "pre_network"): # for dropout
            tensor = self.pre_network(tensor)
        residue = self.edit_residue(tensor) if hasattr(self, "edit_residue") else tensor
        if hasattr(self, "Activation"):
            return self.Activation(self.network(tensor) + residue)
        return self.network(tensor) + residue
# ============================================================================ #


class SimpleFire(nn.Module):
    """ SqueezeNet supporting block - https://arxiv.org/pdf/1602.07360.pdf """
    def __init__(self, tensor_size, filter_size, out_channels, strides=(1, 1), pad=True,
                 activation="relu", dropout=0., normalization=None, pre_nm=False,
                 groups=1, weight_nm=False, equalized=False, *args, **kwargs):
        super(SimpleFire, self).__init__()
        self.pre_network = nn.Sequential()
        if dropout > 0.:
            self.pre_network.add_module("DropOut", nn.Dropout2d(dropout))
        self.pre_network.add_module("Block1x1Shrink", Convolution(tensor_size, 1, out_channels//4, 1, True, "",
                                                                  0., normalization, pre_nm, groups,
                                                                  weight_nm, equalized, **kwargs))

        self.network1 = Convolution(self.pre_network[-1].tensor_size, filter_size, out_channels//2, strides, True,
                                    activation, 0., normalization, pre_nm, groups, weight_nm, equalized, **kwargs)
        self.network2 = Convolution(self.pre_network[-1].tensor_size, 1, out_channels//2, strides, True,
                                    activation, 0., normalization, pre_nm, groups, weight_nm, equalized, **kwargs)

        self.tensor_size = (self.network1.tensor_size[0], self.network1.tensor_size[1]*2,
                            self.network1.tensor_size[2], self.network1.tensor_size[3])

    def forward(self, tensor):
        tensor = self.pre_network(tensor)
        return torch.cat((self.network1(tensor), self.network2(tensor)), 1)
# ============================================================================ #


class CarryModular(nn.Module):
    def __init__(self, tensor_size, filter_size, out_channels, strides=(1, 1), pad=True,
                 activation="relu", dropout=0., normalization=None, pre_nm=False,
                 groups=1, weight_nm=False, equalized=False, growth_rate=32, block=SimpleFire,
                 carry_network="avg", *args, **kwargs):
        super(CarryModular, self).__init__()
        pad = True
        if dropout > 0.:
            self.pre_network = nn.Dropout2d(dropout)
        if tensor_size[1] < out_channels:
            growth_rate = out_channels - tensor_size[1]
        else:
            self.dynamic = True

        if isinstance(block, torch.nn.modules.container.Sequential):
            # If block is a Sequential container ignore arguments
            self.network1 = block
        else:
            self.network1 = block(tensor_size, filter_size, growth_rate, strides, pad, activation,
                                  0., normalization, pre_nm, groups, weight_nm, equalized, **kwargs)

        if (strides > 1 if isinstance(strides, int) else (strides[0] > 1 or strides[1] > 1)):
            if isinstance(carry_network, str):
                self.network2 = nn.AvgPool2d((3, 3), stride=(2, 2), padding=1 if pad else 0) \
                                  if carry_network.lower() == "avg" else \
                                  nn.MaxPool2d((3, 3), stride=(2, 2), padding=1 if pad else 0)
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
        self.tensor_size = (_tensor_size[0], out_channels, _tensor_size[2], _tensor_size[3])

    def forward(self, tensor):
        if hasattr(self, "pre_network"): # for dropout
            tensor = self.pre_network(tensor)
        return torch.cat((self.network1(tensor), self.network2(tensor)), 1)
# ============================================================================ #


class Stem2(nn.Module):
    """ For InceptionV4 and InceptionResNetV2 - https://arxiv.org/pdf/1602.07261.pdf """
    def __init__(self, tensor_size=(1, 3, 299, 299), activation="relu", normalization="batch",
                 pre_nm=False, groups=1, weight_nm=False, equalized=False, *args, **kwargs):
        super(Stem2, self).__init__()

        self.C3_32_2 = Convolution(tensor_size, 3, 32, 2, False, activation, 0.,
                                   normalization, pre_nm, 1, weight_nm, equalized, **kwargs)
        self.C3_32_1 = Convolution(self.C3_32_2.tensor_size, 3, 32, 1, False, activation, 0.,
                                   normalization, pre_nm, 1, weight_nm, equalized, **kwargs)
        self.C3_64_1 = Convolution(self.C3_32_1.tensor_size, 3, 64, 1, True, activation, 0.,
                                   normalization, pre_nm, 1, weight_nm, equalized, **kwargs)

        self.C160 = CarryModular(self.C3_64_1.tensor_size, 3, 160, 2, False, activation, 0.,
                                 normalization, pre_nm, 1, weight_nm, block=Convolution, pool="max")

        channel1 = nn.Sequential()
        channel1.add_module("C1_64_1", Convolution(self.C160.tensor_size, 1, 64, 1, True, activation, 0.,
                                                   normalization, pre_nm, 1, weight_nm, equalized, **kwargs))
        channel1.add_module("C17_64_1", Convolution(channel1[-1].tensor_size, (1, 7), 64, 1, True, activation, 0.,
                                                    normalization, pre_nm, 1, weight_nm, equalized, **kwargs))
        channel1.add_module("C71_64_1", Convolution(channel1[-1].tensor_size, (7, 1), 64, 1, True, activation, 0.,
                                                    normalization, pre_nm, 1, weight_nm, equalized, **kwargs))
        channel1.add_module("C3_96_1", Convolution(channel1[-1].tensor_size, 3, 96, 1, False, activation, 0.,
                                                   normalization, pre_nm, 1, weight_nm, equalized, **kwargs))

        channel2 = nn.Sequential()
        channel2.add_module("C1_64_1", Convolution(self.C160.tensor_size, 1, 64, 1, True, activation, 0.,
                                                   normalization, pre_nm, 1, weight_nm, equalized, **kwargs))
        channel2.add_module("C3_96_1", Convolution(channel2[-1].tensor_size, 3, 96, 1, False, activation, 0.,
                                                   normalization, pre_nm, 1, weight_nm, equalized, **kwargs))

        self.C192 = CarryModular(self.C160.tensor_size, 3, 192, 2, False, activation, 0.,
                                 normalization, pre_nm, 1, weight_nm, block=channel1, carry_network=channel2)

        self.C384 = CarryModular(self.C192.tensor_size, 3, 384, 2, False, activation, 0.,
                                 normalization, pre_nm, 1, weight_nm, block=Convolution, pool="max")

        self.tensor_size = self.C384.tensor_size

    def forward(self, tensor):

        tensor = self.C3_64_1(self.C3_32_1(self.C3_32_2(tensor)))
        tensor = self.C160(tensor)
        return self.C384(self.C192(tensor))
# ============================================================================ #


class InceptionA(nn.Module):
    """ For InceptionV4 - https://arxiv.org/pdf/1602.07261.pdf """
    def __init__(self, tensor_size=(1, 384, 35, 35), activation="relu", normalization="batch",
                 pre_nm=False, groups=1, weight_nm=False, equalized=False, *args, **kwargs):
        super(InceptionA, self).__init__()
        H, W = tensor_size[2:]

        self.path1 = nn.Sequential(nn.AvgPool2d((3, 3), stride=(1, 1), padding=1),
                                   Convolution(tensor_size, 1, 96, 1, True, activation, 0.,
                                               normalization, pre_nm, groups, weight_nm, equalized, **kwargs))
        self.path2 = Convolution(tensor_size, 1, 96, 1, True, activation, 0.,
                                 normalization, pre_nm, groups, weight_nm, equalized, **kwargs)
        self.path3 = nn.Sequential(Convolution(tensor_size, 1, 64, 1, True, activation, 0.,
                                               normalization, pre_nm, groups, weight_nm, equalized, **kwargs),
                                   Convolution((1, 64, H, W), 3, 96, 1, True, activation, 0.,
                                               normalization, pre_nm, groups, weight_nm, equalized, **kwargs))
        self.path4 = nn.Sequential(Convolution(tensor_size, 1, 64, 1, True, activation, 0.,
                                               normalization, pre_nm, groups, weight_nm, equalized, **kwargs),
                                   Convolution((1, 64, H, W), 3, 96, 1, True, activation, 0.,
                                               normalization, pre_nm, groups, weight_nm, equalized, **kwargs),
                                   Convolution((1, 96, H, W), 3, 96, 1, True, activation, 0.,
                                               normalization, pre_nm, groups, weight_nm, equalized, **kwargs))

        self.tensor_size = (1, 96*4, H, W)

    def forward(self, tensor):
        return torch.cat((self.path1(tensor), self.path2(tensor), self.path3(tensor), self.path4(tensor)), 1)
# ============================================================================ #


class ReductionA(nn.Module):
    """
        For InceptionV4 - https://arxiv.org/pdf/1602.07261.pdf
        Reduction from 35 to 17
    """
    def __init__(self, tensor_size=(1, 384, 35, 35), activation="relu", normalization="batch",
                 pre_nm=False, groups=1, weight_nm=False, equalized=False, *args, **kwargs):
        super(ReductionA, self).__init__()
        H, W = tensor_size[2:]

        self.path1 = nn.MaxPool2d((3, 3), stride=(2, 2))
        self.path2 = Convolution(tensor_size, 3, 384, 2, False, activation, 0.,
                                 normalization, pre_nm, groups, weight_nm, equalized, **kwargs)
        self.path3 = nn.Sequential(Convolution(tensor_size, 1, 192, 1, True, activation, 0., normalization,
                                               pre_nm, groups, weight_nm, equalized, **kwargs),
                                   Convolution((1, 192, H, W), 3, 224, 1, True, activation, 0., normalization,
                                               pre_nm, groups, weight_nm, equalized, **kwargs),
                                   Convolution((1, 224, H, W), 3, 256, 2, False, activation, 0., normalization,
                                               pre_nm, groups, weight_nm, equalized, **kwargs))

        self.tensor_size = (1, tensor_size[1]+384+256, self.path2.tensor_size[2], self.path2.tensor_size[3])

    def forward(self, tensor):
        return torch.cat((self.path1(tensor), self.path2(tensor), self.path3(tensor)), 1)
# ============================================================================ #


class InceptionB(nn.Module):
    """ For InceptionV4 - https://arxiv.org/pdf/1602.07261.pdf """
    def __init__(self, tensor_size=(1, 1024, 17, 17), activation="relu", normalization="batch",
                 pre_nm=False, groups=1, weight_nm=False, equalized=False, *args, **kwargs):
        super(InceptionB, self).__init__()
        H, W = tensor_size[2:]

        self.path1 = nn.Sequential(nn.AvgPool2d((3, 3), stride=(1, 1), padding=1),
                                   Convolution(tensor_size, 1, 128, 1, True, activation, 0., normalization,
                                               pre_nm, groups, weight_nm, equalized, **kwargs))
        self.path2 = Convolution(tensor_size, 1, 384, 1, True, activation, 0., normalization,
                                 pre_nm, groups, weight_nm, equalized, **kwargs)
        self.path3 = nn.Sequential(Convolution(tensor_size, 1, 192, 1, True, activation, 0., normalization,
                                               pre_nm, groups, weight_nm, equalized, **kwargs),
                                   Convolution((1, 192, H, W), (1, 7), 224, 1, True, activation, 0., normalization,
                                               pre_nm, groups, weight_nm, equalized, **kwargs),
                                   Convolution((1, 224, H, W), (1, 7), 256, 1, True, activation, 0., normalization,
                                               pre_nm, groups, weight_nm, equalized, **kwargs))
        self.path4 = nn.Sequential(Convolution(tensor_size, 1, 192, 1, True, activation, 0., normalization,
                                               pre_nm, groups, weight_nm, equalized, **kwargs),
                                   Convolution((1, 192, H, W), (1, 7), 192, 1, True, activation, 0., normalization,
                                               pre_nm, groups, weight_nm, equalized, **kwargs),
                                   Convolution((1, 192, H, W), (7, 1), 224, 1, True, activation, 0., normalization,
                                               pre_nm, groups, weight_nm, equalized, **kwargs),
                                   Convolution((1, 224, H, W), (1, 7), 224, 1, True, activation, 0., normalization,
                                               pre_nm, groups, weight_nm, equalized, **kwargs),
                                   Convolution((1, 224, H, W), (7, 1), 256, 1, True, activation, 0., normalization,
                                               pre_nm, groups, weight_nm, equalized, **kwargs))

        self.tensor_size = (1, 128+384+256+256, H, W)

    def forward(self, tensor):
        return torch.cat((self.path1(tensor), self.path2(tensor), self.path3(tensor), self.path4(tensor)), 1)
# ============================================================================ #


class ReductionB(nn.Module):
    """
        For InceptionV4 - https://arxiv.org/pdf/1602.07261.pdf
        Reduction from 17 to 8
    """
    def __init__(self, tensor_size=(1, 1024, 17, 17), activation="relu", normalization="batch",
                 pre_nm=False, groups=1, weight_nm=False, equalized=False, *args, **kwargs):
        super(ReductionB, self).__init__()
        H, W = tensor_size[2:]

        self.path1 = nn.MaxPool2d((3, 3), stride=(2, 2))
        self.path2 = nn.Sequential(Convolution(tensor_size, 1, 192, 1, True, activation, 0., normalization,
                                               pre_nm, groups, weight_nm, equalized, **kwargs),
                                   Convolution((1, 192, H, W), 3, 192, 2, False, activation, 0., normalization,
                                               pre_nm, groups, weight_nm, equalized, **kwargs))
        self.path3 = nn.Sequential(Convolution(tensor_size, 1, 256, 1, True, activation, 0., normalization,
                                               pre_nm, groups, weight_nm, equalized, **kwargs),
                                   Convolution((1, 256, H, W), (1, 7), 256, 1, True, activation, 0., normalization,
                                               pre_nm, groups, weight_nm, equalized, **kwargs),
                                   Convolution((1, 256, H, W), (7, 1), 320, 1, True, activation, 0., normalization,
                                               pre_nm, groups, weight_nm, equalized, **kwargs),
                                   Convolution((1, 320, H, W), 3, 320, 2, False, activation, 0., normalization,
                                               pre_nm, groups, weight_nm, equalized, **kwargs))

        self.tensor_size = (1, tensor_size[1]+192+320, self.path2[-1].tensor_size[2], self.path2[-1].tensor_size[3])

    def forward(self, tensor):
        return torch.cat((self.path1(tensor), self.path2(tensor), self.path3(tensor)), 1)
# ============================================================================ #


class InceptionC(nn.Module):
    """ For InceptionV4 - https://arxiv.org/pdf/1602.07261.pdf """
    def __init__(self, tensor_size=(1, 1536, 8, 8), activation="relu", normalization="batch",
                 pre_nm=False, groups=1, weight_nm=False, equalized=False, *args, **kwargs):
        super(InceptionC, self).__init__()
        H, W = tensor_size[2:]

        self.path1 = nn.Sequential(nn.AvgPool2d((3, 3), stride=(1, 1), padding=1),
                                   Convolution(tensor_size, 1, 256, 1, True, activation, 0., normalization,
                                               pre_nm, groups, weight_nm, equalized, **kwargs))
        self.path2 = Convolution(tensor_size, 1, 256, 1, True, activation, 0., normalization,
                                 pre_nm, groups, weight_nm, equalized, **kwargs)
        self.path3 = Convolution(tensor_size, 1, 384, 1, True, activation, 0., normalization,
                                 pre_nm, groups, weight_nm, equalized, **kwargs)
        self.path3a = Convolution(self.path3.tensor_size, (1, 3), 256, 1, True, activation, 0., normalization,
                                  pre_nm, groups, weight_nm, equalized, **kwargs)
        self.path3b = Convolution(self.path3.tensor_size, (3, 1), 256, 1, True, activation, 0., normalization,
                                  pre_nm, groups, weight_nm, equalized, **kwargs)
        self.path4 = nn.Sequential(Convolution(tensor_size, 1, 384, 1, True, activation, 0., normalization,
                                               pre_nm, groups, weight_nm, equalized, **kwargs),
                                   Convolution((1, 384, H, W), (1, 3), 448, 1, True, activation, 0., normalization, 
                                               pre_nm, groups, weight_nm, equalized, **kwargs),
                                   Convolution((1, 448, H, W), (3, 1), 512, 1, True, activation, 0., normalization,
                                               pre_nm, groups, weight_nm, equalized, **kwargs))
        self.path4a = Convolution(self.path4[-1].tensor_size, (1, 3), 256, 1, True, activation, 0., normalization,
                                  pre_nm, groups, weight_nm, equalized, **kwargs)
        self.path4b = Convolution(self.path4[-1].tensor_size, (3, 1), 256, 1, True, activation, 0., normalization,
                                  pre_nm, groups, weight_nm, equalized, **kwargs)

        self.tensor_size = (1, 256+256+512+512, H, W)

    def forward(self, tensor):
        path3 = self.path3(tensor)
        path4 = self.path4(tensor)
        return torch.cat((self.path1(tensor), self.path2(tensor), self.path3a(path3),
                          self.path3b(path3), self.path4a(path4), self.path4b(path4)), 1)


# from core.NeuralLayers import Convolution, Activations
# tensor_size = (3,3,299,299)
# x = torch.rand(*tensor_size)
# test = Stem2(tensor_size, "relu", "batch", False)
# test(x).size()
# test.C384.tensor_size
# %timeit test(x).size()
# tensor_size = (3,1536,8,8)
# x = torch.rand(*tensor_size)
# test = InceptionC(tensor_size, "relu", "batch", False)
# test(x).size()
# %timeit test(x).size()
#
#
# tensor_size = (3,64,10,10)
# x = torch.rand(*tensor_size)
# test = ResidualOriginal(tensor_size, 3, 64, 2, False, "relu", 0., "batch", False)
# test(x).size()
# %timeit test(x).size()
# test = ResidualComplex(tensor_size, 3, 64, 2, False, "relu", 0., "batch", False)
# test(x).size()
# %timeit test(x).size()
# test = ResidualComplex2(tensor_size, 3, 64, 2, False, "relu", 0., "batch", False)
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
