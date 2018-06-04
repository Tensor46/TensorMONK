""" tensorMONK's :: NeuralArchitectures                                      """

import torch
import torch.nn as nn
import numpy as np
from ..NeuralLayers import *
#==============================================================================#


class MobileNetV2(nn.Module):
    """ Implemented https://arxiv.org/pdf/1801.04381.pdf

        To replicate the paper, use tensor_size = (1, 3, 224, 224)
        Works fairly well, for tensor_size's of min(height, width) >= 128
    """
    def __init__(self, tensor_size=(6, 3, 224, 224), *args, **kwargs):
        super(MobileNetV2, self).__init__()
        activation, batch_nm, pre_nm = "relu", True, False

        self.Net46 = nn.Sequential()
        print(tensor_size)
        self.Net46.add_module("Mobile0", Convolution(tensor_size, 3, 32, 2, True, activation, 0., batch_nm, False))
        print(self.Net46[-1].tensor_size)
        self.Net46.add_module("Mobile1", ResidualInverted(self.Net46[-1].tensor_size, 3, 16, 1, True, activation, 0., batch_nm, pre_nm, t=1))
        print(self.Net46[-1].tensor_size)
        self.Net46.add_module("Mobile2", ResidualInverted(self.Net46[-1].tensor_size, 3, 24, 2, True, activation, 0., batch_nm, pre_nm, t=6))
        print(self.Net46[-1].tensor_size)
        self.Net46.add_module("Mobile3", ResidualInverted(self.Net46[-1].tensor_size, 3, 24, 1, True, activation, 0., batch_nm, pre_nm, t=6))
        print(self.Net46[-1].tensor_size)
        self.Net46.add_module("Mobile4", ResidualInverted(self.Net46[-1].tensor_size, 3, 32, 2, True, activation, 0., batch_nm, pre_nm, t=6))
        print(self.Net46[-1].tensor_size)
        self.Net46.add_module("Mobile5", ResidualInverted(self.Net46[-1].tensor_size, 3, 32, 1, True, activation, 0., batch_nm, pre_nm, t=6))
        print(self.Net46[-1].tensor_size)
        self.Net46.add_module("Mobile6", ResidualInverted(self.Net46[-1].tensor_size, 3, 32, 1, True, activation, 0., batch_nm, pre_nm, t=6))
        print(self.Net46[-1].tensor_size)
        self.Net46.add_module("Mobile7", ResidualInverted(self.Net46[-1].tensor_size, 3, 64, 2, True, activation, 0., batch_nm, pre_nm, t=6))
        print(self.Net46[-1].tensor_size)
        self.Net46.add_module("Mobile8", ResidualInverted(self.Net46[-1].tensor_size, 3, 64, 1, True, activation, 0., batch_nm, pre_nm, t=6))
        print(self.Net46[-1].tensor_size)
        self.Net46.add_module("Mobile9", ResidualInverted(self.Net46[-1].tensor_size, 3, 64, 1, True, activation, 0., batch_nm, pre_nm, t=6))
        print(self.Net46[-1].tensor_size)
        self.Net46.add_module("Mobile10", ResidualInverted(self.Net46[-1].tensor_size, 3, 64, 1, True, activation, 0., batch_nm, pre_nm, t=6))
        print(self.Net46[-1].tensor_size)
        self.Net46.add_module("Mobile11", ResidualInverted(self.Net46[-1].tensor_size, 3, 96, 1, True, activation, 0., batch_nm, pre_nm, t=6))
        print(self.Net46[-1].tensor_size)
        self.Net46.add_module("Mobile12", ResidualInverted(self.Net46[-1].tensor_size, 3, 96, 1, True, activation, 0., batch_nm, pre_nm, t=6))
        print(self.Net46[-1].tensor_size)
        self.Net46.add_module("Mobile13", ResidualInverted(self.Net46[-1].tensor_size, 3, 96, 1, True, activation, 0., batch_nm, pre_nm, t=6))
        print(self.Net46[-1].tensor_size)
        self.Net46.add_module("Mobile14", ResidualInverted(self.Net46[-1].tensor_size, 3, 160, 2, True, activation, 0., batch_nm, pre_nm, t=6))
        print(self.Net46[-1].tensor_size)
        self.Net46.add_module("Mobile15", ResidualInverted(self.Net46[-1].tensor_size, 3, 160, 1, True, activation, 0., batch_nm, pre_nm, t=6))
        print(self.Net46[-1].tensor_size)
        self.Net46.add_module("Mobile16", ResidualInverted(self.Net46[-1].tensor_size, 3, 160, 1, True, activation, 0., batch_nm, pre_nm, t=6))
        print(self.Net46[-1].tensor_size)
        self.Net46.add_module("Mobile17", ResidualInverted(self.Net46[-1].tensor_size, 3, 320, 1, True, activation, 0., batch_nm, pre_nm, t=6))
        print(self.Net46[-1].tensor_size)
        self.Net46.add_module("Mobile18", Convolution(self.Net46[-1].tensor_size, 1, 1280, 1, True, activation, 0., batch_nm, pre_nm))
        print(self.Net46[-1].tensor_size)
        self.Net46.add_module("Pool", nn.AvgPool2d(self.Net46[-1].tensor_size[2:]))
        self.tensor_size = (6, 1280)
        print(1280)

    def forward(self, tensor):
        return self.Net46(tensor).view(tensor.size(0), -1)


# from core.NeuralLayers import *
# tensor = torch.rand(1,3,224,224)
# test = MobileNetV2((1,3,224,224))
# test(tensor).size()
