""" TensorMONK's :: NeuralArchitectures                                     """
import torch.nn as nn
from core.NeuralLayers import Convolution
# =========================================================================== #


class PointNet(nn.Module):
    r"""
        Implemented from paper: Learning Discriminative and Transformation
        Covariant Local Feature Detectors
        Args:
        tensor_size: shape of tensor in BCHW
                     (None/any integer >0, channels, height, width)
        out_channels: depth of output feature channels.
        activation: None/relu/relu6/lklu/elu/prelu/tanh/sigm/maxo/rmxo/swish
        normalization: None/batch/group/instance/layer/pixelwise
    """
    def __init__(self, tensor_size=(1, 1, 32, 32), out_channels=2,
                 *args, **kwargs):
        super(PointNet, self).__init__()
        normalization = "batch"
        activation = "relu"
        self.PointNET = nn.Sequential()
        self.PointNET.add_module("CONV1",
                                 Convolution(tensor_size, 5, 32, 1, False,
                                             activation, 0., None))
        self.PointNET.add_module("POOL1", nn.MaxPool2d(2))
        _tensor_size = self.PointNET[-2].tensor_size
        _tensor_size = (_tensor_size[0], _tensor_size[1],
                        _tensor_size[2]//2, _tensor_size[3]//2)
        self.PointNET.add_module("CONV2",
                                 Convolution(_tensor_size, 3, 128, 1, False,
                                             activation, 0., normalization))
        self.PointNET.add_module("POOL2", nn.MaxPool2d(2))
        _tensor_size = self.PointNET[-2].tensor_size
        _tensor_size = (_tensor_size[0], _tensor_size[1],
                        _tensor_size[2]//2, _tensor_size[3]//2)
        self.PointNET.add_module("CONV3",
                                 Convolution(_tensor_size, 3, 128, 1, False,
                                             activation, 0., normalization))
        self.PointNET.add_module("CONV4",
                                 Convolution(self.PointNET[-1].tensor_size, 3,
                                             256, 1, False, activation, 0.,
                                             normalization))
        self.PointNET.add_module("CONV5",
                                 Convolution(self.PointNET[-1].tensor_size, 2,
                                             out_channels, 1, False,
                                             activation, 0., None))
        self.tensor_size = self.PointNET[-1].tensor_size

    def forward(self, tensor):
        return self.PointNET(tensor).squeeze(2).squeeze(2)


# import torch
# tensor_size = (1, 1, 32,32)
# test = PointNet(tensor_size)
# test(torch.rand(1,1,32,32))
