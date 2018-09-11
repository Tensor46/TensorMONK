""" TensorMONK's :: NeuralArchitectures                                      """

import os
import wget
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..NeuralLayers import Convolution, DenseBlock
# ============================================================================ #


def map_pretrained(state_dict, type):
    # no fully connected
    if type == "d121":
        url = "https://download.pytorch.org/models/densenet121-a639ec97.pth"
    elif type == "d169":
        url = "https://download.pytorch.org/models/densenet169-b2777c0a.pth"
    elif type == "d201":
        url = "https://download.pytorch.org/models/densenet201-c1103571.pth"
    else:
        print(" ... pretrained weights are not avaiable for {}".format(type))
        return state_dict

    # download is not in models
    filename = os.path.join(".../models" if os.path.isdir(".../models") else
        "./models", url.split("/")[-1])
    if not os.path.isfile(filename):
        print(" ... downloading pretrained")
        wget.download(url, filename)
        print(" ... downloaded")

    # relate keys from TensorMONK's model to pretrained
    labels = state_dict.keys()
    prestate_dict = torch.load(filename)
    prelabels = prestate_dict.keys()

    pairs = []
    _labels = [x for x in labels if "num_batches_tracked" not in x and "embedding" not in x]
    _prelabels = [x for x in prelabels if "fc" not in x]
    for x, y in zip(_labels, _prelabels):
        pairs.append((x, y.replace(y.split(".")[-1], x.split(".")[-1])))

    # update the state_dict
    for x, y in pairs:
        if state_dict[x].size() == prestate_dict[y].size():
            state_dict[x] = prestate_dict[y]

    del prestate_dict
    return state_dict
# ============================================================================ #


class DenseNet(nn.Module):
    """
        Implemented from https://arxiv.org/pdf/1608.06993.pdf

            Available models        type
            ================================
            DenseNet-121            d121
            DenseNet-169            d169
            DenseNet-201            d201
            DenseNet-264            d264

        Works for all the min(height, width) >= 32
        To replicate the paper, use default parameters (and select type)
    """

    def __init__(self,
                 tensor_size = (6, 3, 224, 224),
                 type = "d121",
                 activation = "relu",
                 normalization = "batch",
                 pre_nm = True,
                 groups = 1,
                 weight_nm = False,
                 equalized = False,
                 embedding = False,
                 n_embedding = 256,
                 pretrained = False,
                 *args, **kwargs):
        super(DenseNet, self).__init__()

        type = type.lower()
        assert type in ("d121", "d169", "d201", "d264"),\
            """DenseNet :: type must be d121/d169/d201/d264"""

        self.pretrained = pretrained
        if self.pretrained:
            assert tensor_size[1] == 1 or tensor_size[1] == 3, """DenseNet ::
                rgb(preferred)/grey image is required for pretrained"""
            activation, normalization, pre_nm = "relu", "batch", True
            groups, weight_nm, equalized = 1, False, False

        self.type = type
        self.in_tensor_size = tensor_size

        if type == "d121":
            block_params, k = [6, 12, 24, 16], 32
        elif type == "d169":
            block_params, k = [6, 12, 32, 32], 32
        elif type == "d201":
            block_params, k = [6, 12, 48, 32], 32
        elif type == "d264":
            block_params, k = [6, 12, 64, 48], 32
        else:
            raise NotImplementedError

        self.network = nn.Sequential()
        print("Input", tensor_size)
        s = 2
        if min(tensor_size[2], tensor_size[3]) < 64: # Addon -- To make it flexible for other tensor_size's
            s = 1
            print("""DenseNet :: Initial convolution strides changed from 2 to 1,
                as min(tensor_size[2], tensor_size[3]) <  64""")
        self.network.add_module("Convolution",
            Convolution(tensor_size, 7, 64, s, True, activation, 0., normalization,
                        False, 1, weight_nm, equalized, **kwargs))
        print("Convolution", self.network[-1].tensor_size)

        if min(tensor_size[2], tensor_size[3]) > 128:
            self.network.add_module("MaxPool", nn.MaxPool2d((3, 3), stride=(2, 2), padding=1))
            h, w = self.network[-2].tensor_size[2:]
            _tensor_size = (1, 64, h//2+(1 if h%2 == 1 else 0), w//2+(1 if w%2 == 1 else 0))
            print("MaxPool", _tensor_size)
        else: # Addon -- To make it flexible for other tensor_size's
            print("""DenseNet :: MaxPool is ignored if min(tensor_size[2],
                tensor_size[3]) <=  128""")
            _tensor_size = self.network[-1].tensor_size

        for i, n_blocks in enumerate(block_params):
            self.network.add_module("DenseBlock-"+str(i),
                DenseBlock(_tensor_size, 3, _tensor_size[1]+n_blocks*k, 1, True,
                activation, 0., normalization, pre_nm, groups, weight_nm, equalized,
                growth_rate=k, n_blocks=n_blocks, multiplier=4))
            _tensor_size = self.network[-1].tensor_size
            print("DenseBlock-"+str(i), _tensor_size)

            if i+1 != len(block_params):
                self.network.add_module("Transition-"+str(i),
                    nn.Sequential(Convolution(_tensor_size, 1, _tensor_size[1]//2, 1, True,
                    activation, 0., normalization, pre_nm, groups, weight_nm, equalized),
                    nn.AvgPool2d(2, 2)))
                _tensor_size = self.network[-1][-2].tensor_size
                h, w = _tensor_size[2:]
                _tensor_size = (1, _tensor_size[1], h//2, w//2)
                print("Transition-"+str(i), _tensor_size)

        self.network.add_module("AveragePool", nn.AvgPool2d(_tensor_size[2:]))
        print("AveragePool", (1, _tensor_size[1], 1, 1))
        self.tensor_size = (6, _tensor_size[1])

        if embedding:
            self.embedding = nn.Linear(_tensor_size[1], n_embedding, bias=False)
            self.tensor_size = (6, n_embedding)
            print("Linear", self.tensor_size)

        if self.pretrained:
            self.load_pretrained()

    def forward(self, tensor):
        if self.pretrained:
            if self.in_tensor_size[1] == 1: # convert to rgb
                tensor = torch.cat((tensor, tensor, tensor), 1)
            if tensor.min() >= 0: # do imagenet normalization
                tensor[:, 0].add_(-0.485).div_(0.229)
                tensor[:, 1].add_(-0.456).div_(0.224)
                tensor[:, 2].add_(-0.406).div_(0.225)

        if hasattr(self, "embedding"):
            return self.embedding(self.network(tensor).view(tensor.size(0), -1))
        return self.network(tensor).view(tensor.size(0), -1)

    def load_pretrained(self):
        if self.in_tensor_size[1] == 1 or self.in_tensor_size[1] == 3:
            self.load_state_dict(map_pretrained(self.state_dict(), self.type))
        else:
            print(" ... pretrained not available")
            self.pretrained = False


# from core.NeuralLayers import Convolution, DenseBlock
# tensor_size = (1, 3, 224, 224)
# tensor = torch.rand(*tensor_size)
# test = DenseNet(tensor_size, "d121")
# %timeit test(torch.rand(*tensor_size)).shape
# import torchvision.utils as tutils
# test.load_pretrained()
# tutils.save_image(test.state_dict()['network.Convolution.Convolution.weight'],
#     "./models/ws.png")
