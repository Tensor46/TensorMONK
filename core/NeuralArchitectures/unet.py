import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..NeuralLayers import *

class ConvBase(nn.Module):
    ''' (CONV + RELU)* 2'''
    def __init__(self, tensor_size, out_channels, PAD, *args, **kwargs):
        super(ConvBase, self).__init__()
        self.BaseCONV = nn.Sequential()
        self.BaseCONV.add_module("BC1", Convolution(tensor_size, 3, out_channels//2, 1, PAD, "relu", 0.2, None))
        self.BaseCONV.add_module("BC2", Convolution(self.BaseCONV[-1].tensor_size, 3, out_channels, 1, PAD, "relu", 0.2, None))
        self.norm = nn.InstanceNorm2d(self.BaseCONV[-1].tensor_size[1])
    def forward(self, tensor):
        return self.norm(self.BaseCONV(tensor))

class Down(nn.Module):
    def __init__(self, tensor_size, downsample, in_channels, out_channels, PAD = False, *args, **kwargs):
        super(Down, self).__init__()
        self.Down = nn.Sequential()
        self.Down.add_module("MXPL", nn.MaxPool2d((2,2), 2, 0))
        _tensor_size = (tensor_size[0],in_channels,tensor_size[2]//(2*downsample),tensor_size[3]//(2*downsample))
        self.Down.add_module("dCNV", ConvBase(_tensor_size, out_channels, PAD))
    def forward(self, tensor):
        return self.Down(tensor)

class Up(nn.Module):
    def __init__(self, tensor_size, out_channels, PAD = False, *args, **kwargs):
        super(Up, self).__init__()
        self.norm = nn.InstanceNorm2d(tensor_size[1])
        self.Up   = ConvolutionTranspose(tensor_size, 2, tensor_size[1]//2, 2, False)
        _ts = self.Up.tensor_size
        print(_ts)
        _tensor_size = (_ts[0], _ts[1]*2, _ts[2], _ts[3])
        self.Conv = ConvBase(_tensor_size, out_channels, PAD)

    def forward(self, tensor1,tensor2):
        tensor1 = self.Up(tensor1)
        diffX = tensor1.size(2) - tensor2.size(2)
        diffY = tensor1.size(3) - tensor2.size(3)
        tensor2 = F.pad(tensor2, (diffX//2, diffX//2, diffY//2, diffY//2))
        tensor = torch.cat([tensor2, tensor1], dim=1)
        return self.Conv(tensor)

class UNet(nn.Module):
    def __init__(self, tensor_size, out_channels, n_classes = 10, *args, **kwargs):
        super(UNet, self).__init__()
        self.dNet1 = ConvBase(tensor_size, out_channels, False)
        self.dNet2 = Down(self.dNet1.BaseCONV[-1].tensor_size, 1, out_channels, out_channels*2)
        self.dNet3 = Down(self.dNet2.Down[-1].BaseCONV[-1].tensor_size, 1, out_channels*2, out_channels*4)
        self.dNet4 = Down(self.dNet3.Down[-1].BaseCONV[-1].tensor_size, 1, out_channels*4, out_channels*8)
        self.dNet5 = Down(self.dNet4.Down[-1].BaseCONV[-1].tensor_size, 1, out_channels*8, out_channels*16)

        self.uNet1 = Up(self.dNet5.Down[-1].BaseCONV[-1].tensor_size, self.dNet4.Down[-1].BaseCONV[-1].tensor_size[1])
        self.uNet2 = Up(self.uNet1.Conv.BaseCONV[-1].tensor_size, self.dNet3.Down[-1].BaseCONV[-1].tensor_size[1])
        self.uNet3 = Up(self.uNet2.Conv.BaseCONV[-1].tensor_size, self.dNet2.Down[-1].BaseCONV[-1].tensor_size[1])
        self.uNet4 = Up(self.uNet3.Conv.BaseCONV[-1].tensor_size, self.dNet1.BaseCONV[-1].tensor_size[1])
        self.BasicConv = Convolution(self.uNet4.Conv.BaseCONV[-1].tensor_size, 1, n_classes)
        self.tensor_size = self.BasicConv.tensor_size
    def forward(self, tensor):
        d1 = self.dNet1(tensor)
        d2 = self.dNet2(d1)
        d3 = self.dNet3(d2)
        d4 = self.dNet4(d3)
        d5 = self.dNet5(d4)
        u1 = self.uNet1(d5, d4)
        u2 = self.uNet2(u1, d3)
        u3 = self.uNet3(u2, d2)
        u4 = self.uNet4(u3, d1)
        return self.BasicConv(u4)

class UNetPatch(nn.Module):
    def __init__(self, tensor_size, out_channels, n_classes = 2, *args, **kwargs):
        super(UNetPatch, self).__init__()
        self.dNet1 = ConvBase(tensor_size, out_channels, True)
        self.dNet2 = Down(self.dNet1.BaseCONV[-1].tensor_size, 1, out_channels, out_channels*2, True)
        self.dNet3 = Down(self.dNet2.Down[-1].BaseCONV[-1].tensor_size, 1, out_channels*2, out_channels*4, True)

        self.uNet1 = Up(self.dNet3.Down[-1].BaseCONV[-1].tensor_size, self.dNet2.Down[-1].BaseCONV[-1].tensor_size[1], True)
        self.uNet2 = Up(self.uNet1.Conv.BaseCONV[-1].tensor_size, self.dNet1.BaseCONV[-1].tensor_size[1], True)
        self.BasicConv = Convolution(self.uNet2.Conv.BaseCONV[-1].tensor_size, 1, n_classes)
        self.tensor_size = self.BasicConv.tensor_size
    def forward(self, tensor):
        if len(tensor.size())!=4:
            tensor = tensor.unsqueeze(0)
        d1 = self.dNet1(tensor)
        d2 = self.dNet2(d1)
        d3 = self.dNet3(d2)
        u1 = self.uNet1(d3, d2)
        u2 = self.uNet2(u1, d1)
        return self.BasicConv(u2), (d1, d2, d3, u1, u2)

# from core.NeuralLayers import *
# tSize = (1,1,572,572)
# unet = UNet(tSize, 64)
# res = unet(torch.rand(tSize))
#
# tSize = (1,1,48,48)
# unet = UNetPatch(tSize, 64)
# res = unet(torch.rand(tSize))
