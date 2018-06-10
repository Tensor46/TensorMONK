""" TensorMONK's :: NeuralArchitectures                                      """

import torch
import torch.nn as nn
import numpy as np
from ..NeuralLayers import *
#==============================================================================#


class MobileNetV1(nn.Module):
    """
        Implemented https://arxiv.org/pdf/1704.04861.pdf

        To replicate the paper, use default parameters
        Works fairly well, for tensor_size of min(height, width) >= 128
    """
    def __init__(self, tensor_size=(6, 3, 224, 224), activation="relu",
                 batch_nm=True, pre_nm=False, weight_norm=False, *args, **kwargs):
        super(MobileNetV1, self).__init__()

        self.Net46 = nn.Sequential()

        block_params = [(3, 32, 2, 1), (3, 32, 1, 32), (1, 64, 1, 1), (3, 64, 2, 64),
                        (1, 128, 1, 1), (3, 128, 1, 128), (1, 128, 1, 1), (3, 128, 2, 128),
                        (1, 256, 1, 1), (3, 256, 1, 256), (1, 256, 1, 1), (3, 256, 2, 256),
                        (1, 512, 1, 1), (3, 512, 1, 512), (1, 512, 1, 1), (3, 512, 1, 512),
                        (1, 512, 1, 1), (3, 512, 1, 512), (1, 512, 1, 1), (3, 512, 1, 512),
                        (1, 512, 1, 1), (3, 512, 1, 512), (1, 512, 1, 1), (3, 512, 2, 512),
                        (1, 1024, 1, 1), (3, 1024, 1, 1024), (1, 1024, 1, 1), ]

        print("Input", tensor_size)
        _tensor_size = tensor_size
        for i, (k, oc, s, g) in enumerate(block_params):
            self.Net46.add_module("Mobile"+str(i), Convolution(_tensor_size, k, oc, s,
                                  True, activation, 0., batch_nm, False if i == 0 else pre_nm, g, weight_norm))
            _tensor_size = self.Net46[-1].tensor_size
            print("Mobile"+str(i), _tensor_size)

        self.Net46.add_module("AveragePool", nn.AvgPool2d(self.Net46[-1].tensor_size[2:]))
        print("AveragePool", (1, 1024, 1, 1))

        self.tensor_size = (6, 1024)

    def forward(self, tensor):
        return self.Net46(tensor).view(tensor.size(0), -1)


# from core.NeuralLayers import *
# tensor_size = (1, 3, 224, 224)
# tensor = torch.rand(*tensor_size)
# test = MobileNetV1(tensor_size)
# test(tensor).size()
