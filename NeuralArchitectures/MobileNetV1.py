""" tensorMONK's :: NeuralArchitectures                                      """

import torch
import torch.nn as nn
import numpy as np
from ..NeuralLayers import *
#==============================================================================#


class MobileNetV1(nn.Module):
    """ Implemented - https://arxiv.org/pdf/1707.01083.pdf """
    def __init__(self, tensor_size=(6, 3, 224, 224), groups=4, *args, **kwargs):
        super(MobileNetV1, self).__init__()
        activation, batch_nm, pre_nm = "relu", True, False

        self.Net46 = nn.Sequential()
        print(tensor_size)
        self.Net46.add_module("Mobile0", Convolution(tensor_size, 3, 32, 2, True, activation, 0., batch_nm, False))
        print(self.Net46[-1].tensor_size)
        self.Net46.add_module("Mobile1", Convolution(self.Net46[-1].tensor_size, 3, 32, 1, True, activation, 0., batch_nm, False, 32))
        print(self.Net46[-1].tensor_size)
        self.Net46.add_module("Mobile2", Convolution(self.Net46[-1].tensor_size, 1, 64, 1, True, activation, 0., batch_nm, False, 1))
        print(self.Net46[-1].tensor_size)
        self.Net46.add_module("Mobile3", Convolution(self.Net46[-1].tensor_size, 3, 64, 2, True, activation, 0., batch_nm, False, 64))
        print(self.Net46[-1].tensor_size)
        self.Net46.add_module("Mobile4", Convolution(self.Net46[-1].tensor_size, 1, 128, 1, True, activation, 0., batch_nm, False, 1))
        print(self.Net46[-1].tensor_size)
        self.Net46.add_module("Mobile5", Convolution(self.Net46[-1].tensor_size, 3, 128, 1, True, activation, 0., batch_nm, False, 128))
        print(self.Net46[-1].tensor_size)
        self.Net46.add_module("Mobile6", Convolution(self.Net46[-1].tensor_size, 1, 128, 1, True, activation, 0., batch_nm, False, 1))
        print(self.Net46[-1].tensor_size)
        self.Net46.add_module("Mobile7", Convolution(self.Net46[-1].tensor_size, 3, 128, 2, True, activation, 0., batch_nm, False, 128))
        print(self.Net46[-1].tensor_size)
        self.Net46.add_module("Mobile8", Convolution(self.Net46[-1].tensor_size, 1, 256, 1, True, activation, 0., batch_nm, False, 1))
        print(self.Net46[-1].tensor_size)
        self.Net46.add_module("Mobile9", Convolution(self.Net46[-1].tensor_size, 3, 256, 1, True, activation, 0., batch_nm, False, 256))
        print(self.Net46[-1].tensor_size)
        self.Net46.add_module("Mobile10", Convolution(self.Net46[-1].tensor_size, 1, 256, 1, True, activation, 0., batch_nm, False, 1))
        print(self.Net46[-1].tensor_size)
        self.Net46.add_module("Mobile11", Convolution(self.Net46[-1].tensor_size, 3, 256, 2, True, activation, 0., batch_nm, False, 256))
        print(self.Net46[-1].tensor_size)
        self.Net46.add_module("Mobile12", Convolution(self.Net46[-1].tensor_size, 1, 512, 1, True, activation, 0., batch_nm, False, 1))
        print(self.Net46[-1].tensor_size)
        self.Net46.add_module("Mobile13", Convolution(self.Net46[-1].tensor_size, 3, 512, 1, True, activation, 0., batch_nm, False, 512))
        print(self.Net46[-1].tensor_size)
        self.Net46.add_module("Mobile14", Convolution(self.Net46[-1].tensor_size, 1, 512, 1, True, activation, 0., batch_nm, False, 1))
        print(self.Net46[-1].tensor_size)
        self.Net46.add_module("Mobile15", Convolution(self.Net46[-1].tensor_size, 3, 512, 1, True, activation, 0., batch_nm, False, 512))
        print(self.Net46[-1].tensor_size)
        self.Net46.add_module("Mobile16", Convolution(self.Net46[-1].tensor_size, 1, 512, 1, True, activation, 0., batch_nm, False, 1))
        print(self.Net46[-1].tensor_size)
        self.Net46.add_module("Mobile17", Convolution(self.Net46[-1].tensor_size, 3, 512, 1, True, activation, 0., batch_nm, False, 512))
        print(self.Net46[-1].tensor_size)
        self.Net46.add_module("Mobile18", Convolution(self.Net46[-1].tensor_size, 1, 512, 1, True, activation, 0., batch_nm, False, 1))
        print(self.Net46[-1].tensor_size)
        self.Net46.add_module("Mobile19", Convolution(self.Net46[-1].tensor_size, 3, 512, 1, True, activation, 0., batch_nm, False, 512))
        print(self.Net46[-1].tensor_size)
        self.Net46.add_module("Mobile20", Convolution(self.Net46[-1].tensor_size, 1, 512, 1, True, activation, 0., batch_nm, False, 1))
        print(self.Net46[-1].tensor_size)
        self.Net46.add_module("Mobile21", Convolution(self.Net46[-1].tensor_size, 3, 512, 1, True, activation, 0., batch_nm, False, 512))
        print(self.Net46[-1].tensor_size)
        self.Net46.add_module("Mobile22", Convolution(self.Net46[-1].tensor_size, 1, 512, 1, True, activation, 0., batch_nm, False, 1))
        print(self.Net46[-1].tensor_size)
        self.Net46.add_module("Mobile23", Convolution(self.Net46[-1].tensor_size, 3, 512, 2, True, activation, 0., batch_nm, False, 512))
        print(self.Net46[-1].tensor_size)
        self.Net46.add_module("Mobile24", Convolution(self.Net46[-1].tensor_size, 1, 1024, 1, True, activation, 0., batch_nm, False, 1))
        print(self.Net46[-1].tensor_size)
        self.Net46.add_module("Mobile25", Convolution(self.Net46[-1].tensor_size, 3, 1024, 1, True, activation, 0., batch_nm, False, 1024))
        print(self.Net46[-1].tensor_size)
        self.Net46.add_module("Mobile26", Convolution(self.Net46[-1].tensor_size, 1, 1024, 1, True, activation, 0., batch_nm, False, 1))
        print(self.Net46[-1].tensor_size)
        self.Net46.add_module("Mobile26", nn.AvgPool2d(self.Net46[-1].tensor_size[2:]))
        self.tensor_size = (6, 1024)
        print(1024)

    def forward(self, tensor):
        return self.Net46(tensor).view(tensor.size(0), -1)


# from tensorMONK.NeuralLayers import *
# tensor = torch.rand(1,3,224,224)
# test = MobileNetV1((1,3,224,224))
# test(tensor).size()
