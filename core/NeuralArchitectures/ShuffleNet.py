""" TensorMONK's :: NeuralArchitectures                                      """

import torch
import torch.nn as nn
import numpy as np
from ..NeuralLayers import *
#==============================================================================#


class ShuffleNet(nn.Module):
    """
        Implemented https://arxiv.org/pdf/1707.01083.pdf
        See table 1, to better understand type parameter!

        Works for all the min(height, width) >= 32
        To replicate the paper, use default parameters
    """
    def __init__(self, tensor_size=(6, 3, 224, 224), type="g4",
                 activation="relu", batch_nm=True, pre_nm=False, weight_norm=False,
                 embedding=False, n_embedding=256, *args, **kwargs):
        super(ShuffleNet, self).__init__()

        assert type.lower() in ("g1", "g2", "g3", "g4", "g8"), "ShuffleNet -- type must be g1/g2/g3/g4/g8"

        if type.lower() == "g1":
            groups = 1
            block_params = [(144, 2)] + [(144, 1)]*3 + \
                           [(288, 2)] + [(288, 1)]*7 + \
                           [(576, 2)] + [(576, 1)]*3
        elif type.lower() == "g2":
            groups = 2
            block_params = [(200, 2)] + [(200, 1)]*3 + \
                           [(400, 2)] + [(400, 1)]*7 + \
                           [(800, 2)] + [(800, 1)]*3
        elif type.lower() == "g3":
            groups = 3
            block_params = [(240, 2)] + [(240, 1)]*3 + \
                           [(480, 2)] + [(480, 1)]*7 + \
                           [(960, 2)] + [(960, 1)]*3
        elif type.lower() == "g4":
            groups = 4
            block_params = [(272, 2)] + [(272, 1)]*3 + \
                           [(544, 2)] + [(544, 1)]*7 + \
                           [(1088, 2)] + [(1088, 1)]*3
        elif type.lower() == "g8":
            groups = 8
            block_params = [(384, 2)] + [(384, 1)]*3 + \
                           [(768, 2)] + [(768, 1)]*7 + \
                           [(1536, 2)] + [(1536, 1)]*3
        else:
            raise NotImplementedError

        self.Net46 = nn.Sequential()
        print("Input", tensor_size)
        s = 2
        if min(tensor_size[2], tensor_size[3]) < 64: # Addon -- To make it flexible for other tensor_size's
            s = 1
            print("Initial convolution strides changed from 2 to 1, as min(tensor_size[2], tensor_size[3]) <  64")
        self.Net46.add_module("Convolution", Convolution(tensor_size, 3, 24, s, True, activation, 0., batch_nm, False, 1, weight_norm))
        print("Convolution",self.Net46[-1].tensor_size)

        if min(tensor_size[2], tensor_size[3]) > 128:
            self.Net46.add_module("MaxPool", nn.MaxPool2d((3, 3), stride=(2, 2), padding=1))
            _tensor_size = (self.Net46[-2].tensor_size[0], self.Net46[-2].tensor_size[1],
                            self.Net46[-2].tensor_size[2]//2, self.Net46[-2].tensor_size[3]//2)
            print("MaxPool", _tensor_size)
        else: # Addon -- To make it flexible for other tensor_size's
            print("MaxPool is ignored if min(tensor_size[2], tensor_size[3]) <=  128")
            _tensor_size = self.Net46[-1].tensor_size

        for i, (oc, s) in enumerate(block_params):
            self.Net46.add_module("Shuffle"+str(i), ResidualShuffle(_tensor_size, 3, oc, s, True, activation, 0., batch_nm, pre_nm, groups, weight_norm))
            _tensor_size = self.Net46[-1].tensor_size
            print("Shuffle"+str(i), _tensor_size)

        self.Net46.add_module("AveragePool", nn.AvgPool2d(self.Net46[-1].tensor_size[2:]))
        print("AveragePool", (1, oc, 1, 1))
        self.tensor_size = (6, oc)

        if embedding:
            self.embedding = nn.Linear(oc, n_embedding, bias=False)
            self.tensor_size = (6, n_embedding)
            print("Linear", (1, n_embedding))

    def forward(self, tensor):
        if hasattr(self, "embedding"):
            return self.embedding(self.Net46(tensor).view(tensor.size(0), -1))
        return self.Net46(tensor).view(tensor.size(0), -1)


# from core.NeuralLayers import *
# tensor_size = (1, 3, 224, 224)
# tensor = torch.rand(*tensor_size)
# test = ShuffleNet(tensor_size, "g1")
# test(tensor).size()
