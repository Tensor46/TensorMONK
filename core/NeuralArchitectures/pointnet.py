
""" TensorMONK's :: NeuralArchitectures                                      """

import torch
import torch.nn as nn
import numpy as np
from ..NeuralLayers import *
#==============================================================================#
class PointNet(nn.Module):
    """
        Implemented http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhang_Learning_Discriminative_and_CVPR_2017_paper.pdf
    """
    def __init__(self, tensor_size=(1, 1, 32, 32), out_channels=2, *args, **kwargs):
        super(PointNet, self).__init__()
        normalization = "batch"
        self.PointNET    = nn.Sequential()
        self.PointNET.add_module("CONV1", Convolution(tensor_size, 5, 32, 1, False, "relu", 0., None))
        self.PointNET.add_module("POOL1", nn.MaxPool2d(2))
        _tensor_size = self.PointNET[-2].tensor_size
        _tensor_size = (_tensor_size[0], _tensor_size[1], _tensor_size[2]//2, _tensor_size[3]//2)
        #print(_tensor_size)
        self.PointNET.add_module("CONV2", Convolution(_tensor_size, 3, 128,1, False, "relu", 0.,normalization))
        self.PointNET.add_module("POOL2", nn.MaxPool2d(2))
        _tensor_size = self.PointNET[-2].tensor_size
        _tensor_size = (_tensor_size[0], _tensor_size[1], _tensor_size[2]//2, _tensor_size[3]//2)
        #print(_tensor_size)
        self.PointNET.add_module("CONV3", Convolution(_tensor_size, 3, 128, 1, False, "relu", 0.,normalization))
        #print(self.PointNET[-1].tensor_size)
        self.PointNET.add_module("CONV4", Convolution(self.PointNET[-1].tensor_size, 3, 256, 1, False, "relu", 0.,normalization))
        #print(self.PointNET[-1].tensor_size)
        self.PointNET.add_module("CONV5", Convolution(self.PointNET[-1].tensor_size, 2, out_channels, 1, False, "relu", 0.,None))
        print(self.PointNET[-1].tensor_size)
        self.tensor_size = self.PointNET[-1].tensor_size
    def forward(self, tensor):
        return self.PointNET(tensor).squeeze(2).squeeze(2)

# from core.NeuralLayers import *
# tensor_size = (1, 1, 32,32)
# test = PointNet(tensor_size)
# test(torch.rand(1,1,32,32))
