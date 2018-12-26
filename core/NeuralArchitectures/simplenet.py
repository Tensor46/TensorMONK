""" TensorMONK's :: NeuralArchitectures                                     """

import torch
from ..NeuralLayers import Convolution, Linear


class SimpleNet(torch.nn.Sequential):
    """
        For MNIST testing
    """
    def __init__(self, tensor_size=(6, 1, 28, 28), *args, **kwargs):
        super(SimpleNet, self).__init__()

        kwargs = {"pad": True, "activation": "relu",
                  "normalization": None, "pre_nm": False}

        self.add_module("conv1", Convolution(tensor_size, 5, 16, 2, **kwargs))
        self.add_module("conv2", Convolution(self.conv1.tensor_size, 5, 32, 2,
                                             **kwargs))
        self.add_module("conv3", Convolution(self.conv2.tensor_size, 3, 64, 2,
                                             **kwargs))
        self.add_module("linear", Linear(self.conv3.tensor_size, 64, "relu"))
        self.tensor_size = (1, 64)


# from core.NeuralLayers import Convolution, Linear
# tensor_size = (1, 1, 28, 28)
# tensor = torch.rand(*tensor_size)
# test = SimpleNet(tensor_size)
# test(tensor).size()
