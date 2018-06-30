""" TensorMONK's :: NeuralLayers :: CarryResidue                          """

__all__ = ["ResidualOriginal", "ResidualComplex", "ResidualComplex2", "ResidualInverted",
           "ResidualShuffle", "ResidualNeXt",
           "SimpleFire", "CarryModular",
           "Stem2", "InceptionA", "InceptionB", "InceptionC", "ReductionA", "ReductionB"]

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
                 groups=1, weight_nm=False, *args, **kwargs):
        super(ResidualOriginal, self).__init__()
        if dropout > 0.:
            self.pre_network = nn.Dropout2d(dropout)
        self.network = nn.Sequential()
        self.network.add_module("Block3x3_1", Convolution(tensor_size, filter_size, out_channels, strides, True, activation, 0., batch_nm, pre_nm, groups, weight_nm))
        self.network.add_module("Block3x3_2", Convolution(self.network[-1].tensor_size, filter_size, out_channels, (1, 1), True, activation, 0., batch_nm, pre_nm, groups, weight_nm))
        if (strides > 1 if isinstance(strides, int) else (strides[0] > 1 or strides[1] > 1)) or tensor_size[1] != out_channels:
            _filter_size = (3, 3) if (strides > 1 if isinstance(strides, int) else (strides[0] > 1 or strides[1] > 1)) else (1, 1)
            self.edit_residue = Convolution(tensor_size, _filter_size, out_channels, strides, True, "", 0., batch_nm, pre_nm, groups, weight_nm)
        self.tensor_size = self.network[-1].tensor_size
# ============================================================================ #


class ResidualComplex(BaseBlock):
    def __init__(self, tensor_size, filter_size, out_channels, strides=(1, 1), pad=True,
                 activation="relu", dropout=0., batch_nm=False, pre_nm=False,
                 groups=1, weight_nm=False, *args, **kwargs):
        super(ResidualComplex, self).__init__()
        if dropout > 0.:
            self.pre_network = nn.Dropout2d(dropout)
        self.network = nn.Sequential()
        self.network.add_module("Block1x1/4", Convolution(tensor_size, (1, 1), out_channels//4, (1, 1), True, activation, 0., batch_nm, pre_nm, groups, weight_nm))
        self.network.add_module("Block3x3/4", Convolution(self.network[-1].tensor_size, filter_size, out_channels//4, strides, True, activation, 0., batch_nm, pre_nm, groups, weight_nm))
        self.network.add_module("Block1x1", Convolution(self.network[-1].tensor_size, (1, 1), out_channels, (1, 1), True, activation, 0., batch_nm, pre_nm, groups, weight_nm))

        if (strides > 1 if isinstance(strides, int) else (strides[0] > 1 or strides[1] > 1)) or tensor_size[1] != out_channels:
            _filter_size = (3, 3) if (strides > 1 if isinstance(strides, int) else (strides[0] > 1 or strides[1] > 1)) else (1, 1)
            self.edit_residue = Convolution(tensor_size, _filter_size, out_channels, strides, True, "", 0., batch_nm, pre_nm, groups, weight_nm)
        self.tensor_size = self.network[-1].tensor_size
# ============================================================================ #


class ResidualNeXt(BaseBlock):
    """ ResNeXt module -- https://arxiv.org/pdf/1611.05431.pdf """
    def __init__(self, tensor_size, filter_size, out_channels, strides=(1, 1), pad=True,
                 activation="relu", dropout=0., batch_nm=False, pre_nm=False,
                 groups=1, weight_nm=False, *args, **kwargs):
        super(ResidualNeXt, self).__init__()
        if dropout > 0.:
            self.pre_network = nn.Dropout2d(dropout)
        self.network = nn.Sequential()
        self.network.add_module("Block1x1/2", Convolution(tensor_size, (1, 1), out_channels//2, (1, 1), True, activation, 0., batch_nm, pre_nm, groups, weight_nm))
        self.network.add_module("Block3x3/2", Convolution(self.network[-1].tensor_size, filter_size, out_channels//2, strides, True, activation, 0., batch_nm, pre_nm, 32, weight_nm))
        self.network.add_module("Block1x1", Convolution(self.network[-1].tensor_size, (1, 1), out_channels, (1, 1), True, activation, 0., batch_nm, pre_nm, groups, weight_nm))

        if (strides > 1 if isinstance(strides, int) else (strides[0] > 1 or strides[1] > 1)) or tensor_size[1] != out_channels:
            _filter_size, grps = (3, 2) if (strides > 1 if isinstance(strides, int) else (strides[0] > 1 or strides[1] > 1)) else (1, 1)
            self.edit_residue = Convolution(tensor_size, _filter_size, out_channels, strides, True, "", 0., batch_nm, pre_nm, grps, weight_nm)
        self.tensor_size = self.network[-1].tensor_size
# ============================================================================ #


class ResidualComplex2(BaseBlock):
    def __init__(self, tensor_size, filter_size, out_channels, strides=(1, 1), pad=True,
                 activation="relu", dropout=0., batch_nm=False, pre_nm=False,
                 groups=1, weight_nm=False, *args, **kwargs):
        super(ResidualComplex2, self).__init__()
        if dropout > 0.:
            self.pre_network = nn.Dropout2d(dropout)
        self.network = nn.Sequential()
        self.network.add_module("Block1x1/4", Convolution(tensor_size, (1, 1), out_channels//4, (1, 1), True, activation, 0., batch_nm, pre_nm, groups, weight_nm))
        self.network.add_module("Block3x3/4", Convolution(self.network[-1].tensor_size, filter_size, out_channels//4, strides, True, activation, 0., batch_nm, pre_nm, groups, weight_nm))
        self.network.add_module("Block1x1", Convolution(self.network[-1].tensor_size, (1, 1), out_channels, (1, 1), True, "", 0., batch_nm, pre_nm, groups, weight_nm))

        if (strides > 1 if isinstance(strides, int) else (strides[0] > 1 or strides[1] > 1)) or tensor_size[1] != out_channels:
            _filter_size = (3, 3) if (strides > 1 if isinstance(strides, int) else (strides[0] > 1 or strides[1] > 1)) else (1, 1)
            self.edit_residue = Convolution(tensor_size, _filter_size, out_channels, strides, True, "", 0., batch_nm, pre_nm, groups, weight_nm)
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
                 groups=1, weight_nm=False, t=1, *args, **kwargs):
        super(ResidualInverted, self).__init__()
        if dropout > 0.:
            self.pre_network = nn.Dropout2d(dropout)
        self.network = nn.Sequential()
        self.network.add_module("Block1x1pre", Convolution(tensor_size, (1, 1), out_channels*t, (1, 1), True, activation, 0., batch_nm, pre_nm, groups, weight_nm))
        self.network.add_module("Block3x3", Convolution(self.network[-1].tensor_size, filter_size, out_channels*t, strides,
                                                        True, activation, 0., batch_nm, pre_nm, out_channels*t, weight_nm))
        self.network.add_module("Block1x1pst", Convolution(self.network[-1].tensor_size, (1, 1), out_channels, (1, 1), True, "", 0., batch_nm, pre_nm, groups, weight_nm))

        self.skip_residue = False
        if (strides > 1 if isinstance(strides, int) else (strides[0] > 1 or strides[1] > 1)):
            self.skip_residue = True
        if not self.skip_residue and tensor_size[1] != out_channels:
            self.edit_residue = Convolution(tensor_size, (1, 1), out_channels, strides, True, "", 0., batch_nm, pre_nm, groups, weight_nm)
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
                 groups=4, weight_nm=False,*args, **kwargs):
        super(ResidualShuffle, self).__init__()
        if dropout > 0.:
            self.pre_network = nn.Dropout2d(dropout)
        self.network = nn.Sequential()
        self.network.add_module("Block1x1pre", Convolution(tensor_size, (1, 1), out_channels, (1, 1), True, activation, 0., batch_nm, pre_nm, groups, weight_nm))
        self.network.add_module("Shuffle", ChannelShuffle(groups))
        self.network.add_module("Block3x3", Convolution(self.network[-2].tensor_size, filter_size, out_channels, strides, True, "", 0., batch_nm, pre_nm, groups, weight_nm))
        self.network.add_module("Block1x1pst", Convolution(self.network[-1].tensor_size, (1, 1), out_channels, (1, 1), True, "", 0., batch_nm, pre_nm, groups, weight_nm))

        self.edit_residue = nn.Sequential()
        if (strides > 1 if isinstance(strides, int) else (strides[0] > 1 or strides[1] > 1)):
            self.edit_residue.add_module("AveragePOOL", nn.AvgPool2d((3, 3), stride=(2, 2), padding=1))
        if tensor_size[1] != out_channels:
            self.edit_residue.add_module("Block1x1AdjustDepth", Convolution(tensor_size, (1, 1), out_channels, (1, 1), True, "", 0., batch_nm, pre_nm, groups, weight_nm))
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
                 groups=1, weight_nm=False, *args, **kwargs):
        super(SimpleFire, self).__init__()
        self.pre_network = nn.Sequential()
        if dropout > 0.:
            self.pre_network.add_module("DropOut", nn.Dropout2d(dropout))
        self.pre_network.add_module("Block1x1Shrink", Convolution(tensor_size, (1, 1), out_channels//4, (1, 1), True, "", 0., batch_nm, pre_nm, groups, weight_nm))

        self.network = Convolution(self.pre_network[-1].tensor_size, filter_size, out_channels//2, strides, True, activation, 0., batch_nm, pre_nm, groups, weight_nm)
        self.edit_carry = Convolution(self.pre_network[-1].tensor_size, (1, 1), out_channels//2, strides, True, activation, 0., batch_nm, pre_nm, groups, weight_nm)

        self.tensor_size = (self.network.tensor_size[0], self.network.tensor_size[1]*2,
                            self.network.tensor_size[2], self.network.tensor_size[3])
# ============================================================================ #


class CarryModular(BaseBlock):
    def __init__(self, tensor_size, filter_size, out_channels, strides=(1, 1), pad=True,
                 activation="relu", dropout=0., batch_nm=False, pre_nm=False,
                 groups=1, weight_nm=False, growth_rate=32, block=SimpleFire,
                 carry_network="avg", *args, **kwargs):
        super(CarryModular, self).__init__()
        if dropout > 0.:
            self.pre_network = nn.Dropout2d(dropout)
        if tensor_size[1] < out_channels:
            growth_rate = out_channels - tensor_size[1]
        else:
            self.dynamic = True

        if isinstance(block, torch.nn.modules.container.Sequential):
            # If block is an Sequential container ignore parameters
            self.network = block
        else:
            self.network = block(tensor_size, filter_size, growth_rate, strides, pad, activation, 0., batch_nm, pre_nm, groups, weight_nm)

        if (strides > 1 if isinstance(strides, int) else (strides[0] > 1 or strides[1] > 1)):
            if isinstance(carry_network, str):
                self.edit_carry = nn.AvgPool2d((3, 3), stride=(2, 2), padding=1 if pad else 0) \
                                  if carry_network.lower() == "avg" else \
                                  nn.MaxPool2d((3, 3), stride=(2, 2), padding=1 if pad else 0)
            elif isinstance(carry_network, list) or isinstance(carry_network, tuple):
                self.edit_carry = nn.Sequential(*carry_network)
            elif isinstance(carry_network, torch.nn.modules.container.Sequential):
                self.edit_carry = carry_network
            else:
                raise NotImplementedError

        if isinstance(block, torch.nn.modules.container.Sequential):
            _tensor_size = self.network[-1].tensor_size
        else:
            _tensor_size = self.network.tensor_size

        self.tensor_size = (_tensor_size[0], out_channels, _tensor_size[2], _tensor_size[3])
# ============================================================================ #


class Stem2(nn.Module):
    """ For InceptionV4 and InceptionResNetV2 - https://arxiv.org/pdf/1602.07261.pdf """
    def __init__(self, tensor_size=(1, 3, 299, 299), activation="relu", batch_nm=True,
                 pre_nm=False, groups=1, weight_nm=False, *args, **kwargs):
        super(Stem2, self).__init__()

        self.C3_32_2 = Convolution(tensor_size, 3, 32, 2, False, activation, 0., batch_nm, pre_nm, 1, weight_nm)
        self.C3_32_1 = Convolution(self.C3_32_2.tensor_size, 3, 32, 1, False, activation, 0., batch_nm, pre_nm, 1, weight_nm)
        self.C3_64_1 = Convolution(self.C3_32_1.tensor_size, 3, 64, 1, True, activation, 0., batch_nm, pre_nm, 1, weight_nm)

        self.C160 = CarryModular(self.C3_64_1.tensor_size, 3, 160, 2, False, activation, 0., batch_nm, pre_nm, 1, weight_nm, block=Convolution, pool="max")

        channel1 = nn.Sequential()
        channel1.add_module("C1_64_1", Convolution(self.C160.tensor_size, 1, 64, 1, True, activation, 0., batch_nm, pre_nm, 1, weight_nm))
        channel1.add_module("C17_64_1", Convolution(channel1[-1].tensor_size, (1, 7), 64, 1, True, activation, 0., batch_nm, pre_nm, 1, weight_nm))
        channel1.add_module("C71_64_1", Convolution(channel1[-1].tensor_size, (7, 1), 64, 1, True, activation, 0., batch_nm, pre_nm, 1, weight_nm))
        channel1.add_module("C3_96_1", Convolution(channel1[-1].tensor_size, 3, 96, 1, False, activation, 0., batch_nm, pre_nm, 1, weight_nm))

        channel2 = nn.Sequential()
        channel2.add_module("C1_64_1", Convolution(self.C160.tensor_size, 1, 64, 1, True, activation, 0., batch_nm, pre_nm, 1, weight_nm))
        channel2.add_module("C3_96_1", Convolution(channel2[-1].tensor_size, 3, 96, 1, False, activation, 0., batch_nm, pre_nm, 1, weight_nm))

        self.C192 = CarryModular(self.C160.tensor_size, 3, 192, 2, False, activation, 0., batch_nm, pre_nm, 1, weight_nm, block=channel1, carry_network=channel2)

        self.C384 = CarryModular(self.C192.tensor_size, 3, 384, 2, False, activation, 0., batch_nm, pre_nm, 1, weight_nm, block=Convolution, pool="max")

        self.tensor_size = self.C384.tensor_size

    def forward(self, tensor):

        tensor = self.C3_64_1(self.C3_32_1(self.C3_32_2(tensor)))
        tensor = self.C160(tensor)
        return self.C384(self.C192(tensor))
# ============================================================================ #


class InceptionA(nn.Module):
    """ For InceptionV4 - https://arxiv.org/pdf/1602.07261.pdf """
    def __init__(self, tensor_size=(1, 384, 35, 35), activation="relu", batch_nm=True,
                 pre_nm=False, groups=1, weight_nm=False, *args, **kwargs):
        super(InceptionA, self).__init__()
        H, W = tensor_size[2:]

        self.path1 = nn.Sequential(nn.AvgPool2d((3, 3), stride=(1, 1), padding=1),
                                   Convolution(tensor_size, 1, 96, 1, True, activation, 0., batch_nm, pre_nm, groups, weight_nm))
        self.path2 = Convolution(tensor_size, 1, 96, 1, True, activation, 0., batch_nm, pre_nm, groups, weight_nm)
        self.path3 = nn.Sequential(Convolution(tensor_size, 1, 64, 1, True, activation, 0., batch_nm, pre_nm, groups, weight_nm),
                                   Convolution((1, 64, H, W), 3, 96, 1, True, activation, 0., batch_nm, pre_nm, groups, weight_nm))
        self.path4 = nn.Sequential(Convolution(tensor_size, 1, 64, 1, True, activation, 0., batch_nm, pre_nm, groups, weight_nm),
                                   Convolution((1, 64, H, W), 3, 96, 1, True, activation, 0., batch_nm, pre_nm, groups, weight_nm),
                                   Convolution((1, 96, H, W), 3, 96, 1, True, activation, 0., batch_nm, pre_nm, groups, weight_nm))

        self.tensor_size = (1, 96*4, H, W)

    def forward(self, tensor):
        return torch.cat((self.path1(tensor), self.path2(tensor), self.path3(tensor), self.path4(tensor)), 1)
# ============================================================================ #


class ReductionA(nn.Module):
    """
        For InceptionV4 - https://arxiv.org/pdf/1602.07261.pdf
        Reduction from 35 to 17
    """
    def __init__(self, tensor_size=(1, 384, 35, 35), activation="relu", batch_nm=True,
                 pre_nm=False, groups=1, weight_nm=False, *args, **kwargs):
        super(ReductionA, self).__init__()
        H, W = tensor_size[2:]

        self.path1 = nn.MaxPool2d((3, 3), stride=(2, 2))
        self.path2 = Convolution(tensor_size, 3, 384, 2, False, activation, 0., batch_nm, pre_nm, groups, weight_nm)
        self.path3 = nn.Sequential(Convolution(tensor_size, 1, 192, 1, True, activation, 0., batch_nm, pre_nm, groups, weight_nm),
                                   Convolution((1, 192, H, W), 3, 224, 1, True, activation, 0., batch_nm, pre_nm, groups, weight_nm),
                                   Convolution((1, 224, H, W), 3, 256, 2, False, activation, 0., batch_nm, pre_nm, groups, weight_nm))

        self.tensor_size = (1, tensor_size[1]+384+256, self.path2.tensor_size[2], self.path2.tensor_size[3])

    def forward(self, tensor):
        return torch.cat((self.path1(tensor), self.path2(tensor), self.path3(tensor)), 1)
# ============================================================================ #


class InceptionB(nn.Module):
    """ For InceptionV4 - https://arxiv.org/pdf/1602.07261.pdf """
    def __init__(self, tensor_size=(1, 1024, 17, 17), activation="relu", batch_nm=True,
                 pre_nm=False, groups=1, weight_nm=False, *args, **kwargs):
        super(InceptionB, self).__init__()
        H, W = tensor_size[2:]

        self.path1 = nn.Sequential(nn.AvgPool2d((3, 3), stride=(1, 1), padding=1),
                                   Convolution(tensor_size, 1, 128, 1, True, activation, 0., batch_nm, pre_nm, groups, weight_nm))
        self.path2 = Convolution(tensor_size, 1, 384, 1, True, activation, 0., batch_nm, pre_nm, groups, weight_nm)
        self.path3 = nn.Sequential(Convolution(tensor_size, 1, 192, 1, True, activation, 0., batch_nm, pre_nm, groups, weight_nm),
                                   Convolution((1, 192, H, W), (1, 7), 224, 1, True, activation, 0., batch_nm, pre_nm, groups, weight_nm),
                                   Convolution((1, 224, H, W), (1, 7), 256, 1, True, activation, 0., batch_nm, pre_nm, groups, weight_nm))
        self.path4 = nn.Sequential(Convolution(tensor_size, 1, 192, 1, True, activation, 0., batch_nm, pre_nm, groups, weight_nm),
                                   Convolution((1, 192, H, W), (1, 7), 192, 1, True, activation, 0., batch_nm, pre_nm, groups, weight_nm),
                                   Convolution((1, 192, H, W), (7, 1), 224, 1, True, activation, 0., batch_nm, pre_nm, groups, weight_nm),
                                   Convolution((1, 224, H, W), (1, 7), 224, 1, True, activation, 0., batch_nm, pre_nm, groups, weight_nm),
                                   Convolution((1, 224, H, W), (7, 1), 256, 1, True, activation, 0., batch_nm, pre_nm, groups, weight_nm))

        self.tensor_size = (1, 128+384+256+256, H, W)

    def forward(self, tensor):
        return torch.cat((self.path1(tensor), self.path2(tensor), self.path3(tensor), self.path4(tensor)), 1)
# ============================================================================ #


class ReductionB(nn.Module):
    """
        For InceptionV4 - https://arxiv.org/pdf/1602.07261.pdf
        Reduction from 17 to 8
    """
    def __init__(self, tensor_size=(1, 1024, 17, 17), activation="relu", batch_nm=True,
                 pre_nm=False, groups=1, weight_nm=False, *args, **kwargs):
        super(ReductionB, self).__init__()
        H, W = tensor_size[2:]

        self.path1 = nn.MaxPool2d((3, 3), stride=(2, 2))
        self.path2 = nn.Sequential(Convolution(tensor_size, 1, 192, 1, True, activation, 0., batch_nm, pre_nm, groups, weight_nm),
                                   Convolution((1, 192, H, W), 3, 192, 2, False, activation, 0., batch_nm, pre_nm, groups, weight_nm))
        self.path3 = nn.Sequential(Convolution(tensor_size, 1, 256, 1, True, activation, 0., batch_nm, pre_nm, groups, weight_nm),
                                   Convolution((1, 256, H, W), (1, 7), 256, 1, True, activation, 0., batch_nm, pre_nm, groups, weight_nm),
                                   Convolution((1, 256, H, W), (7, 1), 320, 1, True, activation, 0., batch_nm, pre_nm, groups, weight_nm),
                                   Convolution((1, 320, H, W), 3, 320, 2, False, activation, 0., batch_nm, pre_nm, groups, weight_nm))

        self.tensor_size = (1, tensor_size[1]+192+320, self.path2[-1].tensor_size[2], self.path2[-1].tensor_size[3])

    def forward(self, tensor):
        return torch.cat((self.path1(tensor), self.path2(tensor), self.path3(tensor)), 1)
# ============================================================================ #


class InceptionC(nn.Module):
    """ For InceptionV4 - https://arxiv.org/pdf/1602.07261.pdf """
    def __init__(self, tensor_size=(1, 1536, 8, 8), activation="relu", batch_nm=True,
                 pre_nm=False, groups=1, weight_nm=False, *args, **kwargs):
        super(InceptionC, self).__init__()
        H, W = tensor_size[2:]

        self.path1 = nn.Sequential(nn.AvgPool2d((3, 3), stride=(1, 1), padding=1),
                                   Convolution(tensor_size, 1, 256, 1, True, activation, 0., batch_nm, pre_nm, groups, weight_nm))
        self.path2 = Convolution(tensor_size, 1, 256, 1, True, activation, 0., batch_nm, pre_nm, groups, weight_nm)
        self.path3 = Convolution(tensor_size, 1, 384, 1, True, activation, 0., batch_nm, pre_nm, groups, weight_nm)
        self.path3a = Convolution(self.path3.tensor_size, (1, 3), 256, 1, True, activation, 0., batch_nm, pre_nm, groups, weight_nm)
        self.path3b = Convolution(self.path3.tensor_size, (3, 1), 256, 1, True, activation, 0., batch_nm, pre_nm, groups, weight_nm)
        self.path4 = nn.Sequential(Convolution(tensor_size, 1, 384, 1, True, activation, 0., batch_nm, pre_nm, groups, weight_nm),
                                   Convolution((1, 384, H, W), (1, 3), 448, 1, True, activation, 0., batch_nm, pre_nm, groups, weight_nm),
                                   Convolution((1, 448, H, W), (3, 1), 512, 1, True, activation, 0., batch_nm, pre_nm, groups, weight_nm))
        self.path4a = Convolution(self.path4[-1].tensor_size, (1, 3), 256, 1, True, activation, 0., batch_nm, pre_nm, groups, weight_nm)
        self.path4b = Convolution(self.path4[-1].tensor_size, (3, 1), 256, 1, True, activation, 0., batch_nm, pre_nm, groups, weight_nm)

        self.tensor_size = (1, 256+256+512+512, H, W)

    def forward(self, tensor):
        path3 = self.path3(tensor)
        path4 = self.path4(tensor)
        return torch.cat((self.path1(tensor), self.path2(tensor), self.path3a(path3),
                          self.path3b(path3), self.path4a(path4), self.path4b(path4)), 1)


# from core.NeuralLayers import Convolution
# tensor_size = (3,3,299,299)
# x = torch.rand(*tensor_size)
# test = Stem2(tensor_size, "relu", True, False)
# test(x).size()
# test.C384.tensor_size
# %timeit test(x).size()
# tensor_size = (3,1536,8,8)
# x = torch.rand(*tensor_size)
# test = InceptionC(tensor_size, "relu", True, False)
# test(x).size()
# %timeit test(x).size()
#
#
# tensor_size = (3,64,10,10)
# x = torch.rand(*tensor_size)
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
