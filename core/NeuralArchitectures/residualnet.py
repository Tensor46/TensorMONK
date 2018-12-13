""" TensorMONK's :: NeuralArchitectures                                      """

import os
import wget
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..NeuralLayers import *
# ============================================================================ #


def map_pretrained(state_dict, type):
    # no fully connected
    if type == "r18":
        url = r'https://download.pytorch.org/models/resnet18-5c106cde.pth'
    elif type == "r34":
        url = r'https://download.pytorch.org/models/resnet34-333f7ec4.pth'
    elif type == "r50":
        url = r'https://download.pytorch.org/models/resnet50-19c8e357.pth'
    elif type == "r101":
        url = r'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'
    elif type == "r152":
        url = r'https://download.pytorch.org/models/resnet152-b121ed2d.pth'
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
    _labels = [x for x in labels if "num_batches_tracked" not in x and
        "edit_residue" not in x and "embedding" not in x]
    _prelabels = [x for x in prelabels if "fc" not in x and
        "downsample" not in x]
    for x, y in zip(_labels, _prelabels):
        pairs.append((x, y.replace(y.split(".")[-1], x.split(".")[-1])))

    _labels = [x for x in labels if "num_batches_tracked" not in x and
        "edit_residue" in x]
    _prelabels = [x for x in prelabels if "fc" not in x and
        "downsample" in x]
    for x, y in zip(_labels, _prelabels):
        pairs.append((x, y.replace(y.split(".")[-1], x.split(".")[-1])))

    # update the state_dict
    for x, y in pairs:
        if state_dict[x].size() == prestate_dict[y].size():
            state_dict[x] = prestate_dict[y]

    del prestate_dict
    return state_dict
# ============================================================================ #


class ResidualNet(nn.Module):
    """
        Implemented
        ResNet*   from https://arxiv.org/pdf/1512.03385.pdf
        ResNeXt*  from https://arxiv.org/pdf/1611.05431.pdf
        SEResNet* from https://arxiv.org/pdf/1709.01507.pdf
        SEResNeXt* --  Squeeze-and-Excitation + ResNeXt

            Available models        type
            ================================
            ResNet18                r18
            ResNet34                r34
            ResNet50                r50
            ResNet101               r101
            ResNet152               r152
            ResNeXt50               rn50
            ResNeXt101              rn101
            ResNeXt152              rn152
            SEResNet50              ser50
            SEResNet101             ser101
            SEResNet152             ser152
            SEResNeXt50             sern50
            SEResNeXt101            sern101
            SEResNeXt152            sern152

        * SE = Squeeze-and-Excitation
        Works for all the min(height, width) >= 32
        To replicate the paper, use default parameters (and select type)
    """

    def __init__(self,
                 tensor_size = (6, 3, 128, 128),
                 type = "r18",
                 activation = "relu",
                 normalization = "batch",
                 pre_nm = False,
                 groups = 1,
                 weight_nm = False,
                 equalized = False,
                 embedding = False,
                 shift = False,
                 n_embedding = 256,
                 pretrained = False,
                 *args, **kwargs):
        super(ResidualNet, self).__init__()

        type = type.lower()
        assert type in ("r18", "r34", "r50", "r101", "r152", "rn50", "rn101",
            "rn152", "ser50", "ser101", "ser152", "sern50", "sern101", "sern152"),\
            """ResidualNet -- type must be r18/r34/r50/r101/r152/rn50/rn101/
            rn152/ser50/ser101/ser152/sern50/sern101/sern152"""

        self.pretrained = pretrained
        if self.pretrained:
            assert tensor_size[1] == 1 or tensor_size[1] == 3, """ResidualNet ::
                rgb(preferred)/grey image is required for pretrained"""
            activation, normalization, pre_nm = "relu", "batch", True
            groups, weight_nm, equalized, shift = 1, False, False, False
        self.type = type
        self.in_tensor_size = tensor_size

        if type == "r18":
            BaseBlock = ResidualOriginal
            # 2x 64; 2x 128; 2x 256; 2x 512
            block_params = [(64, 1), (64, 1), (128, 2), (128, 1),
                            (256, 2), (256, 1), (512, 2), (512, 1)]
        elif type == "r34":
            BaseBlock = ResidualOriginal
            # 3x 64; 4x 128; 6x 256; 3x 512
            block_params = [(64, 1)]*3 + \
                           [(128, 2)] + [(128, 1)]*3 + \
                           [(256, 2)] + [(256, 1)]*5 + \
                           [(512, 2)] + [(512, 1)]*2
        elif type in ("r50", "rn50", "ser50", "sern50"):
            if type.startswith("se"):
                BaseBlock = SEResidualNeXt if type == "sern50" else SEResidualComplex
            else:
                BaseBlock = ResidualNeXt if type == "rn50" else ResidualComplex
            # 3x 256; 4x 512; 6x 1024; 3x 2048
            block_params = [(256, 1)]*3 + \
                           [(512, 2)] + [(512, 1)]*3 + \
                           [(1024, 2)] + [(1024, 1)]*5 + \
                           [(2048, 2)] + [(2048, 1)]*2
        elif type in ("r101", "rn101", "ser101", "sern101"):
            if type.startswith("se"):
                BaseBlock = SEResidualNeXt if type == "sern101" else SEResidualComplex
            else:
                BaseBlock = ResidualNeXt if type == "rn101" else ResidualComplex
            # 3x 256; 4x 512; 23x 1024; 3x 2048
            block_params = [(256, 1)]*3 + \
                           [(512, 2)] + [(512, 1)]*3 + \
                           [(1024, 2)] + [(1024, 1)]*22 + \
                           [(2048, 2)] + [(2048, 1)]*2
        elif type in ("r152", "rn152", "ser152", "sern152"):
            if type.startswith("se"):
                BaseBlock = SEResidualNeXt if type == "sern152" else SEResidualComplex
            else:
                BaseBlock = ResidualNeXt if type == "rn152" else ResidualComplex
            # 3x 256; 8x 512; 36x 1024; 3x 2048
            block_params = [(256, 1)]*3 + \
                           [(512, 2)] + [(512, 1)]*7 + \
                           [(1024, 2)] + [(1024, 1)]*35 + \
                           [(2048, 2)] + [(2048, 1)]*2
        else:
            raise NotImplementedError

        self.Net46 = nn.Sequential()
        print("Input", tensor_size)
        s = 2
        if min(tensor_size[2], tensor_size[3]) < 64: # Addon -- To make it flexible for other tensor_size's
            s = 1
            print("Initial convolution strides changed from 2 to 1, as min(tensor_size[2], tensor_size[3]) <  64")
        self.Net46.add_module("Convolution",
            Convolution(tensor_size, 7, 64, s, True, activation, 0., normalization,
                        False, 1, weight_nm, equalized, **kwargs))
        print("Convolution", self.Net46[-1].tensor_size)

        if min(tensor_size[2], tensor_size[3]) > 128:
            self.Net46.add_module("MaxPool", nn.MaxPool2d((3, 3), stride=(2, 2), padding=1))
            _tensor_size = (self.Net46[-2].tensor_size[0], self.Net46[-2].tensor_size[1],
                            self.Net46[-2].tensor_size[2]//2, self.Net46[-2].tensor_size[3]//2)
            print("MaxPool", _tensor_size)
        else: # Addon -- To make it flexible for other tensor_size's
            print("MaxPool is ignored if min(tensor_size[2], tensor_size[3]) <=  128")
            _tensor_size = self.Net46[-1].tensor_size

        for i, (oc, s) in enumerate(block_params):
            self.Net46.add_module("Residual"+str(i),
                BaseBlock(_tensor_size, 3, oc, s, True, activation, 0., normalization,
                          pre_nm, groups, weight_nm, equalized, **kwargs))
            _tensor_size = self.Net46[-1].tensor_size
            print("Residual"+str(i), _tensor_size)

        self.Net46.add_module("AveragePool", nn.AvgPool2d(self.Net46[-1].tensor_size[2:]))
        print("AveragePool", (1, oc, 1, 1))
        self.tensor_size = (6, oc)

        if embedding:
            self.embedding = nn.Linear(oc, n_embedding, bias=False)
            self.tensor_size = (6, n_embedding)
            print("Linear", (1, n_embedding))

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
            return self.embedding(self.Net46(tensor).view(tensor.size(0), -1))
        return self.Net46(tensor).view(tensor.size(0), -1)

    def load_pretrained(self):
        if self.in_tensor_size[1] == 1 or self.in_tensor_size[1] == 3:
            self.load_state_dict(map_pretrained(self.state_dict(), self.type))
        else:
            print(" ... pretrained not available")
            self.pretrained = False

# from core.NeuralLayers import *
# tensor_size = (1, 3, 224, 224)
# tensor = torch.rand(*tensor_size)
# test = ResidualNet(tensor_size, "ser50")
# test(tensor).size()
# import torchvision.utils as tutils
# test = ResidualNet(tensor_size, "r34")
# test.load_pretrained()
# tutils.save_image(test.state_dict()['Net46.Convolution.Convolution.weight'],
#     "./models/ws.png")
