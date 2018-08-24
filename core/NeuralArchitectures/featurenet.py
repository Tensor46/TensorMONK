
""" TensorMONK's :: NeuralArchitectures                                      """

import torch
import torch.nn as nn
import numpy as np
import sys
from ..NeuralLayers import *
#==============================================================================#
class FeatureNet(nn.Module):
    """
    Generic idea of exploiting multi scale features. try and explicity use different scales for information boosting.
    """
    def __init__(self, tensor_size=(1, 1, 32, 32), out_channels = 256, *args, **kwargs):
        super(FeatureNet, self).__init__()

        self.FeatureNET_HR1    = nn.Sequential()
        self.FeatureNET_HR2    = nn.Sequential()
        self.FeatureNET_HR3    = nn.Sequential()

        self.FeatureNET_LR1    = nn.Sequential()
        self.FeatureNET_LR2    = nn.Sequential()
        self.FeatureNET_LR3    = nn.Sequential()

        padding, batchnm, stride = False, "batch", 1
        self.FeatureNET_HR1.add_module("HR_CONV1", Convolution(tensor_size, 3, 16, stride, padding, "relu6", 0.,batchnm)) #16, 1, 30, 30
        self.FeatureNET_HR1.add_module("HR_CONV2", Convolution(self.FeatureNET_HR1[-1].tensor_size, 3, 32, stride, padding, "relu6", 0.,batchnm)) #32, 1, 28, 28
        self.FeatureNET_HR1.add_module("HR_POO1", nn.MaxPool2d(2)) #32, 1, 14, 14
        _tensor_size1 = self.FeatureNET_HR1[-2].tensor_size
        _tensor_size1 = (_tensor_size1[0], _tensor_size1[1], _tensor_size1[2]//2, _tensor_size1[3]//2)
        self.FeatureNET_HR2.add_module("HR_CONV3", Convolution(_tensor_size1, 3, 64, stride, padding, "relu6", 0.,batchnm)) #64, 1, 12, 12
        self.FeatureNET_HR2.add_module("HR_POO2", nn.MaxPool2d(2)) #64, 1, 6, 6
        _tensor_size2 = self.FeatureNET_HR2[-2].tensor_size
        _tensor_size2 = (_tensor_size2[0], _tensor_size2[1], _tensor_size2[2]//2, _tensor_size2[3]//2)
        self.FeatureNET_HR3.add_module("HR_CONV4", Convolution(_tensor_size2, 3, 64, stride, padding, "relu6", 0.,batchnm)) #64, 1, 4, 4
        self.FeatureNET_HR3.add_module("HR_CONV5", Convolution(self.FeatureNET_HR3[-1].tensor_size, 3, 128, stride, padding, "relu6", 0.,batchnm)) #128, 1, 2, 2
        self.FeatureNET_HR3.add_module("HR_POOL3", nn.MaxPool2d(2)) #128, 1, 1, 1
        _tensor_size3 = self.FeatureNET_HR3[-2].tensor_size
        _tensor_size3 = (_tensor_size3[0], _tensor_size3[1], _tensor_size3[2]//2, _tensor_size3[3]//2)

        self.FeatureNET_LR1.add_module("LR_POO1", nn.AvgPool2d(2)) #1, 1, 16, 16
        _tensor_size = tensor_size
        _tensor_size = (_tensor_size[0], _tensor_size[1], _tensor_size[2]//2, _tensor_size[3]//2)
        self.FeatureNET_LR1.add_module("LR_CONV1", Convolution(_tensor_size, 3, 16, stride, padding, "relu6", 0.,batchnm)) #16, 1, 14, 14
        _tensor_size = self.FeatureNET_LR1[-1].tensor_size
        _tensor_size = (_tensor_size[0], _tensor_size[1]+_tensor_size1[1], _tensor_size[2], _tensor_size[3])
        self.FeatureNET_LR2.add_module("LR_CONV2", Convolution(_tensor_size, 3, 32, stride, padding, "relu6", 0.,batchnm)) #16, 1, 12, 12
        self.FeatureNET_LR2.add_module("LR_POO2", nn.AvgPool2d(2)) #1, 1, 6, 6
        _tensor_size = self.FeatureNET_LR2[-2].tensor_size
        _tensor_size = (_tensor_size[0], _tensor_size[1], _tensor_size[2]//2, _tensor_size[3]//2)

        _tensor_size = (_tensor_size[0], _tensor_size[1]+_tensor_size2[1], _tensor_size[2]//2, _tensor_size[3]//2)
        self.FeatureNET_LR3.add_module("LR_CONV3", Convolution(_tensor_size, 3, 64, stride, padding, "relu6", 0.,batchnm)) #16, 1, 4, 4
        self.FeatureNET_LR3.add_module("LR_CONV4", Convolution(self.FeatureNET_LR3[-1].tensor_size, 3, 128, stride, padding, "relu6", 0.,batchnm)) #16, 1, 4, 4
        self.FeatureNET_LR3.add_module("LR_POO3", nn.AvgPool2d(2)) #1, 1, 6, 6
        _tensor_size = self.FeatureNET_LR3[-2].tensor_size
        _tensor_size = (_tensor_size[0], _tensor_size[1]+_tensor_size3[1], _tensor_size[2]//2, _tensor_size[3]//2)

        self.mergeNET = Convolution(_tensor_size, 1, out_channels, 1)

        self.norm01 = lambda x: x.div(x.max(2, True)[0].max(3, True)[0] + 1e-8)
        self.tensor_size = _tensor_size
    def forward(self, tensor):
        y1 = torch.tanh(self.FeatureNET_HR1(tensor))
        y2 = torch.tanh(self.FeatureNET_HR2(y1))
        y3 = torch.tanh(self.FeatureNET_HR3(y2))

        x1 = torch.tanh(self.FeatureNET_LR1(tensor))
        x1 = x1+torch.rand(*x1.size()).to(x1.device).div(10.).add(-0.05)
        x1 = x1.clamp(0)
        x2 = torch.tanh(self.FeatureNET_LR2(torch.cat((x1, y1.detach()), 1)))
        x3 = torch.tanh(self.FeatureNET_LR3(torch.cat((x2, y2.detach()), 1)))

        merge = self.mergeNET(torch.cat((y3, x3),1))
        return y3, x3, torch.tanh(merge)

class FeatureCapNet(nn.Module):
    def __init__(self, tensor_size, *args,**kwargs):
        super(FeatureCapNet,self).__init__()
        self.FeatureNET = nn.Sequential()
        normalization = "batch"
        self.FeatureNET.add_module("CONV1", Convolution(tensor_size, 3, 32, 2, False,"relu",0.,normalization)) #15
        self.FeatureNET.add_module("RESN1", ResidualComplex(self.FeatureNET[-1].tensor_size, 3, 64, 1, False, "relu", 0., normalization)) #15
        self.FeatureNET.add_module("RESN2", ResidualComplex(self.FeatureNET[-1].tensor_size, 3, 96, 1, False, "relu", 0., normalization)) #15

        primary_n_capsules      = 8
        primary_capsule_length  = 16

        routing_capsule_length  = 16
        n_labels                = 2
        routing_iterations      = 3
        block                   = Convolution

        self.PrimaryCapsule = PrimaryCapsule(self.FeatureNET[-1].tensor_size,
                                             filter_size=5, out_channels=128, strides=1,
                                             pad=False, activation="", dropout=0.,
                                             normalization=None, pre_nm=False, groups=1,
                                             block=block, n_capsules=primary_n_capsules,
                                             capsule_length=primary_capsule_length)

        self.RoutingCapsule = RoutingCapsule(self.PrimaryCapsule.tensor_size,
                                             n_capsules=n_labels, capsule_length=routing_capsule_length,
                                             iterations=routing_iterations)
        self.tensor_size = self.RoutingCapsule.tensor_size

    def forward(self, tensor):
        initial_tensor = self.FeatureNET(tensor)
        primary_tensor = self.PrimaryCapsule(initial_tensor)
        routing_tensor = self.RoutingCapsule(primary_tensor)
        return routing_tensor.view(tensor.size(0),-1)

# tSize = (1,1,32,32)
# capnet = FeatureCapNet(tSize)
# res = capnet(torch.rand(tSize))
#
# splitnet = FeatureNet(tSize)
# res = splitnet(torch.rand(tSize))
