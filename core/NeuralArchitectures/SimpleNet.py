""" TensorMONK's :: NeuralArchitectures                                      """

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from ..NeuralLayers import *
#==============================================================================#


class SimpleNet(nn.Module):
    """
        For MNIST testing
    """
    def __init__(self, tensor_size=(6, 1, 28, 28), *args, **kwargs):
        super(SimpleNet, self).__init__()
        activation, batch_nm, pre_nm = "relu", False, False

        self.Net46 = nn.Sequential()
        self.Net46.add_module("conv1", Convolution(tensor_size, 5, 16, 2, True, activation, 0., batch_nm, False))
        self.Net46.add_module("conv2", Convolution(self.Net46[-1].tensor_size, 5, 32, 2, True, activation, 0., batch_nm, False))
        self.Net46.add_module("conv3", Convolution(self.Net46[-1].tensor_size, 3, 64, 2, True, activation, 0., batch_nm, False))
        self.linear = nn.Linear(np.prod(self.Net46[-1].tensor_size), 64, bias=True)

        self.tensor_size = (6, 64)

    def forward(self, tensor):
        return F.relu(self.linear(self.Net46(tensor).view(tensor.size(0), -1)))


# from core.NeuralLayers import *
# tensor_size = (1, 1, 28, 28)
# tensor = torch.rand(*tensor_size)
# test = SimpleNet(tensor_size)
# test(tensor).size()
