
""" TensorMONK's :: NeuralArchitectures                                      """

import torch
import torch.nn as nn
import numpy as np
from ..NeuralLayers import *
#==============================================================================#


class ContextNet(nn.Module):
    """
        Implemented https://arxiv.org/pdf/1805.04554.pdf
    """
    def __init__(self, tensor_size=(1, 3, 1024, 2048), *args, **kwargs):
        super(ContextNet, self).__init__()
        Strides         = [2, 1, 1]
        bottleneck      = ContextNet_Bottleneck
        normalization   = "batch"
        self.DeepNET    = nn.Sequential()
        self.ShallowNET = nn.Sequential()
        self.DeepNET.add_module("AVGPL", nn.AvgPool2d((5,5), (4,4), 2)) # 1, 1, 256, 512
        self.DeepNET.add_module("DN_CNV1", Convolution(tensor_size, 3, 32, 2, True, "relu", 0., normalization, False, 1)) # 1, 1, 128, 256
        self.DeepNET.add_module("DN_BN10", bottleneck(self.DeepNET[-1].tensor_size, 3, 32, 1, expansion=1))
        self.DeepNET.add_module("DN_BN20", bottleneck(self.DeepNET[-1].tensor_size, 3, 32, 1, expansion=6)) # 1, 1, 128, 256
        for i in range(3): self.DeepNET.add_module("DN_BN3"+str(i), bottleneck(self.DeepNET[-1].tensor_size, 3, 48, Strides[i], expansion=6)) # 1, 1, 64, 128
        for i in range(3): self.DeepNET.add_module("DN_BN4"+str(i), bottleneck(self.DeepNET[-1].tensor_size, 3, 64, Strides[i], expansion=6)) # 1, 1, 32, 64
        for i in range(2): self.DeepNET.add_module("DN_BN5"+str(i), bottleneck(self.DeepNET[-1].tensor_size, 3, 96, 1, expansion=6))
        for i in range(2): self.DeepNET.add_module("DN_BN6"+str(i), bottleneck(self.DeepNET[-1].tensor_size, 3, 128, 1, expansion=6)) # 1, 1, 32, 64
        self.DeepNET.add_module("DN_CNV2", Convolution(self.DeepNET[-1].tensor_size, 3, 128, 1, True, "relu", 0., normalization, False, 1)) # 1, 1, 32, 64
        self.DeepNET.add_module("UPSMPLE", nn.Upsample(scale_factor = 4, mode = 'bilinear')) # 1, 1, 128, 256
        _tensor_size = (1, 128, self.DeepNET[-2].tensor_size[2]*4, self.DeepNET[-2].tensor_size[3]*4)
        self.DeepNET.add_module("DN_DW11", Convolution(_tensor_size, 3, _tensor_size[1], 1, True, "relu", 0., None, False, groups =_tensor_size[1], dilation = 4))
        self.DeepNET.add_module("DN_DW12", Convolution(self.DeepNET[-1].tensor_size, 1, 128, 1, True, "relu", 0., normalization, False, 1))
        self.DeepNET.add_module("DN_CNV3", Convolution(self.DeepNET[-1].tensor_size, 1, 128, 1, True, "relu", 0., normalization, False, 1)) # 128, 256

        activation, pre_nm, groups = "relu", False, 1
        self.ShallowNET.add_module("SM_CNV1", Convolution(tensor_size, 3, 32, 2, True, "relu", 0.,True, False, 1)) # 512 x 1024
        self.ShallowNET.add_module("SM_DW11", Convolution(self.ShallowNET[-1].tensor_size, 3, 32, 2, True, activation, 0., None, pre_nm, groups = tensor_size[1])) # 256, 512
        self.ShallowNET.add_module("SM_DW12", Convolution(self.ShallowNET[-1].tensor_size, 1, 64, 1,True, activation, 0., normalization, pre_nm, groups))
        self.ShallowNET.add_module("SM_DW21", Convolution(self.ShallowNET[-1].tensor_size, 3, 64, 2, True, activation, 0., None, pre_nm, groups = tensor_size[1])) # 128, 256
        self.ShallowNET.add_module("SM_DW22", Convolution(self.ShallowNET[-1].tensor_size, 1, 128, 1,True, activation, 0., normalization, pre_nm, groups))
        self.ShallowNET.add_module("SM_DW31", Convolution(self.ShallowNET[-1].tensor_size, 3, 128, 1, True, activation, 0., None, pre_nm, groups = tensor_size[1]))
        self.ShallowNET.add_module("SM_DW32", Convolution(self.ShallowNET[-1].tensor_size, 1, 128, 1,True, activation, 0., normalization, pre_nm, groups))
        self.ShallowNET.add_module("SM_CNV2", Convolution(self.ShallowNET[-1].tensor_size, 1, 128, 1,True, activation, 0., normalization, pre_nm, groups)) # 128, 256
        self.tensor_size = self.ShallowNET[-1].tensor_size

        self.FuseNET = Convolution(self.ShallowNET[-1].tensor_size, 1, self.ShallowNET[-1].tensor_size[1], 1, True)

    def forward(self, tensor):
        return self.FuseNET(self.DeepNET(tensor)+self.ShallowNET(tensor))

# from core.NeuralLayers import *
# tensor_size = (1, 1, 1024, 2048)
# test = ContextNet(tensor_size)
