""" tensorMONK's :: NeuralArchitectures                                      """

import torch
import torch.nn as nn
import numpy as np
from ..NeuralLayers import *
#==============================================================================#


class ShuffleNet(nn.Module):
    """ Implemented https://arxiv.org/pdf/1707.01083.pdf """
    def __init__(self, tensor_size=(6, 3, 224, 224), groups=4, *args, **kwargs):
        super(ShuffleNet, self).__init__()
        activation, batch_nm, pre_nm = "relu", True, False
        if groups == 1:
            c1, c2, c3, c4 = 24, 144, 288, 576
        elif groups == 2:
            c1, c2, c3, c4 = 24, 200, 400, 800
        elif groups == 3:
            c1, c2, c3, c4 = 24, 240, 480, 960
        elif groups == 4:
            c1, c2, c3, c4 = 24, 272, 544, 1088
        elif groups == 8:
            c1, c2, c3, c4 = 24, 384, 768, 1536
        else:
            raise NotImplementedError

        self.Net46 = nn.Sequential()
        print(tensor_size)
        self.Net46.add_module("Shuffle0", Convolution(tensor_size, 3, c1, 2, True, activation, 0., batch_nm, False))
        print(self.Net46[-1].tensor_size)
        self.Net46.add_module("Shuffle1", nn.AvgPool2d((3, 3), stride=(2, 2), padding=1))
        _tensor_size = (self.Net46[-2].tensor_size[0], self.Net46[-2].tensor_size[1],
                        self.Net46[-2].tensor_size[2]//2, self.Net46[-2].tensor_size[3]//2)
        print(_tensor_size)
        self.Net46.add_module("Shuffle2", ResidualShuffle(_tensor_size, 3, c2, 2, True, activation, 0., batch_nm, False))
        print(self.Net46[-1].tensor_size)
        self.Net46.add_module("Shuffle3", ResidualShuffle(self.Net46[-1].tensor_size, 3, c2, 1, True, activation, 0., batch_nm, False))
        print(self.Net46[-1].tensor_size)
        self.Net46.add_module("Shuffle4", ResidualShuffle(self.Net46[-1].tensor_size, 3, c2, 1, True, activation, 0., batch_nm, False))
        print(self.Net46[-1].tensor_size)
        self.Net46.add_module("Shuffle5", ResidualShuffle(self.Net46[-1].tensor_size, 3, c2, 1, True, activation, 0., batch_nm, False))
        print(self.Net46[-1].tensor_size)
        self.Net46.add_module("Shuffle6", ResidualShuffle(self.Net46[-1].tensor_size, 3, c3, 2, True, activation, 0., batch_nm, False))
        print(self.Net46[-1].tensor_size)
        self.Net46.add_module("Shuffle7", ResidualShuffle(self.Net46[-1].tensor_size, 3, c3, 1, True, activation, 0., batch_nm, False))
        print(self.Net46[-1].tensor_size)
        self.Net46.add_module("Shuffle8", ResidualShuffle(self.Net46[-1].tensor_size, 3, c3, 1, True, activation, 0., batch_nm, False))
        print(self.Net46[-1].tensor_size)
        self.Net46.add_module("Shuffle9", ResidualShuffle(self.Net46[-1].tensor_size, 3, c3, 1, True, activation, 0., batch_nm, False))
        print(self.Net46[-1].tensor_size)
        self.Net46.add_module("Shuffle10", ResidualShuffle(self.Net46[-1].tensor_size, 3, c3, 1, True, activation, 0., batch_nm, False))
        print(self.Net46[-1].tensor_size)
        self.Net46.add_module("Shuffle11", ResidualShuffle(self.Net46[-1].tensor_size, 3, c3, 1, True, activation, 0., batch_nm, False))
        print(self.Net46[-1].tensor_size)
        self.Net46.add_module("Shuffle12", ResidualShuffle(self.Net46[-1].tensor_size, 3, c3, 1, True, activation, 0., batch_nm, False))
        print(self.Net46[-1].tensor_size)
        self.Net46.add_module("Shuffle13", ResidualShuffle(self.Net46[-1].tensor_size, 3, c3, 1, True, activation, 0., batch_nm, False))
        print(self.Net46[-1].tensor_size)
        self.Net46.add_module("Shuffle14", ResidualShuffle(self.Net46[-1].tensor_size, 3, c4, 2, True, activation, 0., batch_nm, False))
        print(self.Net46[-1].tensor_size)
        self.Net46.add_module("Shuffle15", ResidualShuffle(self.Net46[-1].tensor_size, 3, c4, 1, True, activation, 0., batch_nm, False))
        print(self.Net46[-1].tensor_size)
        self.Net46.add_module("Shuffle16", ResidualShuffle(self.Net46[-1].tensor_size, 3, c4, 1, True, activation, 0., batch_nm, False))
        print(self.Net46[-1].tensor_size)
        self.Net46.add_module("Shuffle17", ResidualShuffle(self.Net46[-1].tensor_size, 3, c4, 1, True, activation, 0., batch_nm, False))
        print(self.Net46[-1].tensor_size)
        self.Net46.add_module("Shuffle18", nn.AvgPool2d(self.Net46[-1].tensor_size[2:]))
        self.tensor_size = (6, c4)
        print(c4)

    def forward(self, tensor):
        return self.Net46(tensor).view(tensor.size(0), -1)


# from tensorMONK.NeuralLayers import *
# tensor_size = (1, 3, 160, 128)
# tensor = torch.rand(*tensor_size)
# test = ShuffleNet(tensor_size)
# test(tensor).size()
