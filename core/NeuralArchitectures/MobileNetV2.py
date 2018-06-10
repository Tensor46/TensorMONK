""" TensorMONK's :: NeuralArchitectures                                      """

import torch
import torch.nn as nn
import numpy as np
from ..NeuralLayers import *
#==============================================================================#

class MobileNetV2(nn.Module):
    """
        Implemented https://arxiv.org/pdf/1801.04381.pdf

        To replicate the paper, use default parameters
        Works fairly well, for tensor_size's of min(height, width) >= 128
    """
    def __init__(self, tensor_size=(6, 3, 224, 224), activation="relu",
                 batch_nm=True, pre_nm=False, weight_norm=False, *args, **kwargs):
        super(MobileNetV2, self).__init__()

        self.Net46 = nn.Sequential()

        block_params = [(16, 1, 1), (24, 2, 6), (24, 1, 6), (32, 2, 6),
                        (32, 1, 6), (32, 1, 6), (64, 2, 6), (64, 1, 6),
                        (64, 1, 6), (64, 1, 6), (96, 1, 6), (96, 1, 6),
                        (96, 1, 6), (160, 2, 6), (160, 1, 6), (160, 1, 6),
                        (320, 1, 6),]

        print("Input", tensor_size)
        self.Net46 = nn.Sequential()
        self.Net46.add_module("Convolution", Convolution(tensor_size, 3, 32, 2, True, activation, 0., batch_nm, False, 1, weight_norm))
        print("Convolution", self.Net46[-1].tensor_size)

        for i, (oc, s, t) in enumerate(block_params):
            self.Net46.add_module("ResidualInverted"+str(i), ResidualInverted(self.Net46[-1].tensor_size, 3, oc, s, True, activation, 0., batch_nm, pre_nm, 1, weight_norm, t=t))
            print("ResidualInverted"+str(i), self.Net46[-1].tensor_size)

        self.Net46.add_module("ConvolutionLast", Convolution(self.Net46[-1].tensor_size, 1, 1280, 1, True, activation, 0., batch_nm, pre_nm, 1, weight_norm))
        print("ConvolutionLast", self.Net46[-1].tensor_size)
        self.Net46.add_module("AveragePool", nn.AvgPool2d(self.Net46[-1].tensor_size[2:]))
        print("AveragePool", (1, 1280, 1, 1))

        self.tensor_size = (6, 1280)

    def forward(self, tensor):
        return self.Net46(tensor).view(tensor.size(0), -1)

# from core.NeuralLayers import *
# tensor_size = (1, 3, 224, 224)
# tensor = torch.rand(*tensor_size)
# test = MobileNetV2(tensor_size)
# test(tensor).size()
