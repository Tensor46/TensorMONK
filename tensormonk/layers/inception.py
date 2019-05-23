""" TensorMONK :: layers :: inception """

__all__ = ["Stem2", "InceptionA", "InceptionB", "InceptionC",
           "ReductionA", "ReductionB"]

import torch
import torch.nn as nn
from .convolution import Convolution
from .utils import update_kwargs, compute_flops
from .carryresidue import CarryModular


class Stem2(nn.Module):
    r""" For InceptionV4 - https://arxiv.org/pdf/1602.07261.pdf
    All args are similar to Convolution and shift is disabled. Designed for an
    input tensor_size of (1, 3, 299, 299) to deliver an output of size
    (1, 384, 35, 35)
    """
    def __init__(self, tensor_size=(1, 3, 299, 299), activation="relu",
                 dropout=0., normalization="batch", pre_nm=False, groups=1,
                 weight_nm=False, equalized=False, shift=False,
                 bias=False, dropblock=True, **kwargs):
        super(Stem2, self).__init__()
        kwargs = update_kwargs(kwargs, *([None]*5), activation, dropout,
                               normalization, pre_nm, groups, weight_nm,
                               equalized, shift, bias, dropblock)

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

    def flops(self):
        return compute_flops(self)
# =========================================================================== #


class InceptionA(nn.Module):
    r""" For InceptionV4 - https://arxiv.org/pdf/1602.07261.pdf
    All args are similar to Convolution and shift is disabled. Designed for an
    input tensor_size of (1, 384, 35, 35) to deliver an output of size
    (1, 384, 35, 35)
    """
    def __init__(self, tensor_size=(1, 384, 35, 35), activation="relu",
                 dropout=0., normalization="batch", pre_nm=False, groups=1,
                 weight_nm=False, equalized=False, shift=False,
                 bias=False, dropblock=True, **kwargs):
        super(InceptionA, self).__init__()
        h, w = tensor_size[2:]
        kwargs = update_kwargs(kwargs, *([None]*4), True, activation, dropout,
                               normalization, pre_nm, groups, weight_nm,
                               equalized, shift, bias, dropblock)

        self._flops = tensor_size[1] * (h//2) * (w//2) * (3*3+1)
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

    def flops(self):
        return compute_flops(self) + self._flops
# =========================================================================== #


class ReductionA(nn.Module):
    r""" For InceptionV4 - https://arxiv.org/pdf/1602.07261.pdf
    All args are similar to Convolution and shift is disabled. Designed for an
    input tensor_size of (1, 384, 35, 35) to deliver an output of size
    (1, 1024, 17, 17)
    """
    def __init__(self, tensor_size=(1, 384, 35, 35), activation="relu",
                 dropout=0., normalization="batch", pre_nm=False, groups=1,
                 weight_nm=False, equalized=False, shift=False,
                 bias=False, dropblock=True, **kwargs):
        super(ReductionA, self).__init__()
        h, w = tensor_size[2:]
        kwargs = update_kwargs(kwargs, *([None]*5), activation, dropout,
                               normalization, pre_nm, groups, weight_nm,
                               equalized, shift, bias, dropblock)
        self._flops = tensor_size[1] * (h//2) * (w//2) * (3*3+1)
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

    def flops(self):
        return compute_flops(self) + self._flops
# =========================================================================== #


class InceptionB(nn.Module):
    r""" For InceptionV4 - https://arxiv.org/pdf/1602.07261.pdf
    All args are similar to Convolution and shift is disabled. Designed for an
    input tensor_size of (1, 1024, 17, 17) to deliver an output of size
    (1, 1024, 17, 17)
    """
    def __init__(self, tensor_size=(1, 1024, 17, 17), activation="relu",
                 dropout=0., normalization="batch", pre_nm=False, groups=1,
                 weight_nm=False, equalized=False, shift=False,
                 bias=False, dropblock=True, **kwargs):
        super(InceptionB, self).__init__()
        h, w = tensor_size[2:]
        kwargs = update_kwargs(kwargs, *([None]*4), True, activation, dropout,
                               normalization, pre_nm, groups, weight_nm,
                               equalized, shift, bias, dropblock)
        self._flops = tensor_size[1] * (h//2) * (w//2) * (3*3+1)
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

    def flops(self):
        return compute_flops(self) + self._flops
# =========================================================================== #


class ReductionB(nn.Module):
    r""" For InceptionV4 - https://arxiv.org/pdf/1602.07261.pdf
    All args are similar to Convolution and shift is disabled. Designed for an
    input tensor_size of (1, 1024, 17, 17) to deliver an output of size
    (1, 1536, 8, 8)
    """
    def __init__(self, tensor_size=(1, 1024, 17, 17), activation="relu",
                 dropout=0., normalization="batch", pre_nm=False, groups=1,
                 weight_nm=False, equalized=False, shift=False,
                 bias=False, dropblock=True, **kwargs):
        super(ReductionB, self).__init__()
        h, w = tensor_size[2:]
        kwargs = update_kwargs(kwargs, *([None]*5), activation, dropout,
                               normalization, pre_nm, groups, weight_nm,
                               equalized, shift, bias, dropblock)
        self._flops = tensor_size[1] * (h//2) * (w//2) * (3*3+1)
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

    def flops(self):
        return compute_flops(self) + self._flops
# =========================================================================== #


class InceptionC(nn.Module):
    r""" For InceptionV4 - https://arxiv.org/pdf/1602.07261.pdf
    All args are similar to Convolution and shift is disabled. Designed for an
    input tensor_size of (1, 1536, 8, 8) to deliver an output of size
    (1, 1536, 8, 8)
    """
    def __init__(self, tensor_size=(1, 1536, 8, 8), activation="relu",
                 dropout=0., normalization="batch", pre_nm=False, groups=1,
                 weight_nm=False, equalized=False, shift=False,
                 bias=False, dropblock=True, **kwargs):
        super(InceptionC, self).__init__()
        h, w = tensor_size[2:]
        kwargs = update_kwargs(kwargs, *([None]*4), True, activation, dropout,
                               normalization, pre_nm, groups, weight_nm,
                               equalized, shift, bias, dropblock)
        self._flops = tensor_size[1] * (h//2) * (w//2) * (3*3+1)
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

    def flops(self):
        return compute_flops(self) + self._flops


# from tensormonk.layers import Convolution, CarryModular
# from tensormonk.layers.utils import update_kwargs, compute_flops
# tensor_size = (3, 3, 299, 299)
# x = torch.rand(*tensor_size)
# test = Stem2(tensor_size, "relu", 0., "batch", False)
# test(x).size()
# test.flops()
# %timeit test(x).size()
# test = InceptionA()
# test(torch.rand(*(1, 384, 35, 35))).size()
# test.flops()
# test = ReductionA()
# test(torch.rand(*(1, 384, 35, 35))).size()
# test.flops()
# test = InceptionB((1, 1024, 17, 17))
# test(torch.rand(*(1, 1024, 17, 17))).size()
# test.flops()
# test = ReductionB((1, 1024, 17, 17))
# test(torch.rand(*(1, 1024, 17, 17))).size()
# test.flops()
# test = InceptionC((1, 1536, 8, 8))
# test(torch.rand(*(1, 1536, 8, 8))).size()
# test.flops()
