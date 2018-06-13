""" TensorMONK's :: NeuralArchitectures                                      """


import torch
import torch.nn as nn
import numpy as np
from ..NeuralLayers import *
#==============================================================================#


class ResidualNet(nn.Module):
    """
        Implemented https://arxiv.org/pdf/1512.03385.pdf

        Works for all the min(height, width) >= 32
        To replicate the paper, use default parameters (and select type)
    """

    def __init__(self, tensor_size=(6,3,128,128), type="r18",
                 activation="relu", batch_nm=True, pre_nm=False, weight_nm=False,
                 *args, **kwargs):
        super(ResidualNet, self).__init__()

        assert type.lower() in ("r18", "r34", "r50", "r101", "r152"), "ResidualNet -- type must be r18/r34/r50/r101/r152"

        if type.lower() == "r18":
            BaseBlock = ResidualOriginal
            # 2x 64; 2x 128; 2x 256; 2x 512
            block_params = [(64, 1), (64, 1), (256, 2), (256, 1),
                            (128, 2), (128, 1), (512, 2), (512, 1)]
        elif type.lower() == "r34":
            BaseBlock = ResidualOriginal
            # 3x 64; 4x 128; 6x 256; 3x 512
            block_params = [(64, 1)]*3 + \
                           [(128, 2)] + [(128, 1)]*3 + \
                           [(256, 2)] + [(256, 1)]*5 + \
                           [(512, 2)] + [(512, 1)]*2
        elif type.lower() == "r50":
            BaseBlock = ResidualComplex
            # 3x 256; 4x 512; 6x 1024; 3x 2048
            block_params = [(256, 1)]*3 + \
                           [(512, 2)] + [(512, 1)]*3 + \
                           [(1024, 2)] + [(1024, 1)]*5 + \
                           [(2048, 2)] + [(2048, 1)]*2
        elif type.lower() == "r101":
            BaseBlock = ResidualComplex
            # 3x 256; 4x 512; 23x 1024; 3x 2048
            block_params = [(256, 1)]*3 + \
                           [(512, 2)] + [(512, 1)]*3 + \
                           [(1024, 2)] + [(1024, 1)]*22 + \
                           [(2048, 2)] + [(2048, 1)]*2
        elif type.lower() == "r152":
            BaseBlock = ResidualComplex
            # 3x 256; 8x 512; 36x 1024; 3x 2048
            block_params = [(256, 1)]*3 + \
                           [(512, 2)] + [(512, 1)]*7 + \
                           [(1024, 2)] + [(1024, 1)]*35 + \
                           [(2048, 2)] + [(2048, 1)]*2
        else:
            raise NotImplementedError

        self.Net46 = nn.Sequential()
        print("Input", tensor_size)
        s = 2
        if min(tensor_size[2], tensor_size[3]) < 64: # Addon -- To make it flexible for other tensor_size's
            s = 1
            print("Initial convolution strides changed from 2 to 1, as min(tensor_size[2], tensor_size[3]) <  64")
        self.Net46.add_module("Convolution", Convolution(tensor_size, 7, 64, s, True, activation, 0., batch_nm, False, 1, weight_nm))
        print("Convolution", self.Net46[-1].tensor_size)

        if min(tensor_size[2], tensor_size[3]) > 128:
            self.Net46.add_module("MaxPool", nn.MaxPool2d((3, 3), stride=(2, 2), padding=1))
            _tensor_size = (self.Net46[-2].tensor_size[0], self.Net46[-2].tensor_size[1],
                            self.Net46[-2].tensor_size[2]//2, self.Net46[-2].tensor_size[3]//2)
            print("MaxPool", _tensor_size)
        else: # Addon -- To make it flexible for other tensor_size's
            print("MaxPool is ignored if min(tensor_size[2], tensor_size[3]) <=  128")
            _tensor_size = self.Net46[-1].tensor_size

        for i, (oc, s) in enumerate(block_params):
            self.Net46.add_module("Residual"+str(i), BaseBlock(_tensor_size, 3, oc, s, True, activation, 0., batch_nm, pre_nm, 1, weight_nm))
            _tensor_size = self.Net46[-1].tensor_size
            print("Residual"+str(i), _tensor_size)

        self.Net46.add_module("AveragePool", nn.AvgPool2d(self.Net46[-1].tensor_size[2:]))
        print("AveragePool", (1, oc, 1, 1))

        self.tensor_size = (6, oc)

    def forward(self, tensor):
        return self.Net46(tensor).view(tensor.size(0), -1)


# from core.NeuralLayers import *
# tensor_size = (1, 3, 224, 224)
# tensor = torch.rand(*tensor_size)
# test = ResidualNet(tensor_size, "r18")
# test(tensor).size()
