""" TensorMONK's :: NeuralArchitectures                                      """


import torch
import torch.nn as nn
import numpy as np
from ..NeuralLayers import *
#==============================================================================#


class InceptionV4(nn.Module):
    """
        Implemented https://arxiv.org/pdf/1602.07261.pdf
    """

    def __init__(self, tensor_size=(6, 3, 299, 299),
                 activation="relu", batch_nm=True, pre_nm=False, groups=1,
                 weight_nm=False, embedding=False, n_embedding=256, *args, **kwargs):
        super(InceptionV4, self).__init__()

        self.Net46 = nn.Sequential()
        print("Input", tensor_size)
        self.Net46.add_module("Stem", Stem2(tensor_size, activation, batch_nm, False, groups, weight_nm))
        print("Stem", self.Net46[-1].tensor_size)
        for i in range(4):
            self.Net46.add_module("InceptionA"+str(i), InceptionA(self.Net46[-1].tensor_size, activation, batch_nm, False, groups, weight_nm))
            print("InceptionA", self.Net46[-1].tensor_size)
        self.Net46.add_module("ReductionA", ReductionA(self.Net46[-1].tensor_size, activation, batch_nm, False, groups, weight_nm))
        print("ReductionA", self.Net46[-1].tensor_size)
        for i in range(7):
            self.Net46.add_module("InceptionB"+str(i), InceptionB(self.Net46[-1].tensor_size, activation, batch_nm, False, groups, weight_nm))
            print("InceptionB", self.Net46[-1].tensor_size)
        self.Net46.add_module("ReductionB", ReductionB(self.Net46[-1].tensor_size, activation, batch_nm, False, groups, weight_nm))
        print("ReductionB", self.Net46[-1].tensor_size)
        for i in range(3):
            self.Net46.add_module("InceptionC"+str(i), InceptionC(self.Net46[-1].tensor_size, activation, batch_nm, False, groups, weight_nm))
            print("InceptionC", self.Net46[-1].tensor_size)

        self.Net46.add_module("AveragePool", nn.AvgPool2d(self.Net46[-1].tensor_size[2:]))
        print("AveragePool", (1, self.Net46[-2].tensor_size, 1, 1))
        self.tensor_size = (6, self.Net46[-2].tensor_size)

        if embedding:
            self.embedding = nn.Linear(self.Net46[-2].tensor_size, n_embedding, bias=False)
            self.tensor_size = (6, n_embedding)
            print("Linear", (1, n_embedding))

    def forward(self, tensor):
        if hasattr(self, "embedding"):
            return self.embedding(self.Net46(tensor).view(tensor.size(0), -1))
        return self.Net46(tensor).view(tensor.size(0), -1)


# from core.NeuralLayers import *
# tensor_size = (1, 3, 299, 299)
# tensor = torch.rand(*tensor_size)
# test = InceptionV4(tensor_size)
# %timeit test(tensor).size()
