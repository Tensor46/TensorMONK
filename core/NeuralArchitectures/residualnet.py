""" TensorMONK's :: NeuralArchitectures                                     """

import os
import wget
import torch
import torch.nn as nn
from ..NeuralLayers import Convolution, ResidualOriginal, ResidualComplex,\
    ResidualNeXt, SEResidualComplex, SEResidualNeXt
from ..utils import ImageNetNorm
# =========================================================================== #


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


class ResidualNet(nn.Sequential):
    r"""Versions of residual networks. With the ability to change the strides of
    initial convolution and remove max pool the models works for all the
    min(height, width) >= 32. To replicate the paper, use default parameters
    (and select type)
        Implemented
        ResNet*   from https://arxiv.org/pdf/1512.03385.pdf
        ResNeXt*  from https://arxiv.org/pdf/1611.05431.pdf
        SEResNet* from https://arxiv.org/pdf/1709.01507.pdf
        SEResNeXt* --  Squeeze-and-Excitation + ResNeXt

    Args:
        tensor_size: shape of tensor in BCHW
            (None/any integer >0, channels, height, width)
        type (string): model type
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

        activation: None/relu/relu6/lklu/elu/prelu/tanh/sigm/maxo/rmxo/swish
        normalization: None/batch/group/instance/layer/pixelwise
        pre_nm: if True, normalization -> activation -> convolution else
            convolution -> normalization -> activation
        groups: grouped convolution, value must be divisble by tensor_size[1]
            and out_channels, default = 1
        weight_nm: True/False, default = False
        equalized: True/False, default = False
        shift: True/False, default = False
            Shift replaces 3x3 convolution with pointwise convs after shifting.
            Requires tensor_size[1] >= 9 and filter_size = 3
        n_embedding: when not None and > 0, adds a linear layer to the network
            and returns a torch.Tensor of shape (None, n_embedding)
        pretrained: downloads and updates the weights with pretrained weights
    """

    def __init__(self,
                 tensor_size=(6, 3, 128, 128),
                 type: str = "r18",
                 activation: str = "relu",
                 normalization: str = "batch",
                 pre_nm: bool = False,
                 groups: int = 1,
                 weight_nm: bool = False,
                 equalized: bool = False,
                 shift: bool = False,
                 n_embedding: int = None,
                 pretrained: bool = False,
                 *args, **kwargs):
        super(ResidualNet, self).__init__()

        type = type.lower()
        assert type in ("r18", "r34", "r50", "r101", "r152", "rn50", "rn101",
                        "rn152", "ser50", "ser101", "ser152", "sern50",
                        "sern101", "sern152"),\
            """ResidualNet -- type must be r18/r34/r50/r101/r152/rn50/rn101/
            rn152/ser50/ser101/ser152/sern50/sern101/sern152"""

        self.pretrained = pretrained
        if self.pretrained:
            assert tensor_size[1] == 1 or tensor_size[1] == 3, """ResidualNet ::
                rgb(preferred)/grey image is required for pretrained"""
            activation, normalization, pre_nm = "relu", "batch", True
            groups, weight_nm, equalized, shift = 1, False, False, False
        self.model_type = type
        self.in_tensor_size = tensor_size

        if type in ("r18", "r34"):
            BaseBlock = ResidualOriginal
            if type == "r18":
                # 2x 64; 2x 128; 2x 256; 2x 512
                block_params = [(64, 1), (64, 1), (128, 2), (128, 1),
                                (256, 2), (256, 1), (512, 2), (512, 1)]
            else:
                # 3x 64; 4x 128; 6x 256; 3x 512
                block_params = [(64, 1)]*3 + \
                               [(128, 2)] + [(128, 1)]*3 + \
                               [(256, 2)] + [(256, 1)]*5 + \
                               [(512, 2)] + [(512, 1)]*2
        else:
            if type in ("r50", "r101", "r152"):
                BaseBlock = ResidualComplex
            elif type in ("rn50", "rn101", "rn152"):
                BaseBlock = ResidualNeXt
            elif type in ("ser50", "ser101", "ser152"):
                BaseBlock = SEResidualComplex
            elif type in ("sern50", "sern101", "sern152"):
                BaseBlock = SEResidualNeXt

            if type.endswith("50"):
                # 3x 256; 4x 512; 6x 1024; 3x 2048
                block_params = [(256, 1)]*3 + \
                               [(512, 2)] + [(512, 1)]*3 + \
                               [(1024, 2)] + [(1024, 1)]*5 + \
                               [(2048, 2)] + [(2048, 1)]*2
            elif type.endswith("101"):
                # 3x 256; 4x 512; 23x 1024; 3x 2048
                block_params = [(256, 1)]*3 + \
                               [(512, 2)] + [(512, 1)]*3 + \
                               [(1024, 2)] + [(1024, 1)]*22 + \
                               [(2048, 2)] + [(2048, 1)]*2
            elif type.endswith("152"):
                # 3x 256; 8x 512; 36x 1024; 3x 2048
                block_params = [(256, 1)]*3 + \
                               [(512, 2)] + [(512, 1)]*7 + \
                               [(1024, 2)] + [(1024, 1)]*35 + \
                               [(2048, 2)] + [(2048, 1)]*2

        if pretrained:
            print("ImageNetNorm = ON")
            self.add_module("ImageNetNorm", ImageNetNorm())
        print("Input", tensor_size)
        s = 2
        if min(tensor_size[2], tensor_size[3]) < 64:
            # Addon -- To make it flexible for other tensor_size's
            s = 1
            print("Initial convolution strides changed from 2 to 1, " +
                  "as min(height, width) <  64")
        self.add_module("InitialConvolution",
                        Convolution(tensor_size, 7, 64, s, True,
                                    activation, 0., normalization, False,
                                    1, weight_nm, equalized, **kwargs))
        t_size = self.InitialConvolution.tensor_size
        print("InitialConvolution", t_size)

        if min(tensor_size[2], tensor_size[3]) > 128:
            self.add_module("MaxPool", nn.MaxPool2d(3, 2, 1))
            t_size = (1, 64, t_size[2]//2, t_size[3]//2)
            print("MaxPool", t_size)
        else:  # Addon -- To make it flexible for other tensor_size's
            print("MaxPool is ignored if min(height, width) <= 128")

        # Residual blocks
        for i, (oc, s) in enumerate(block_params):
            nm = "Residual" + str(i)
            self.add_module(nm, BaseBlock(t_size, 3, oc, s, True,
                                          activation, 0., normalization,
                                          pre_nm, groups, weight_nm,
                                          equalized, shift, **kwargs))
            t_size = getattr(self, nm).tensor_size
            print(nm, t_size)

        self.add_module("AveragePool",
                        nn.AvgPool2d(t_size[2:]))
        print("AveragePool", (1, oc, 1, 1))
        self.tensor_size = (1, oc)

        if n_embedding is not None and n_embedding > 0:
            self.add_module("embedding", nn.Linear(oc, n_embedding,
                                                   bias=False))
            self.tensor_size = (1, n_embedding)
            print("Linear", (1, n_embedding))

        if self.pretrained:
            self.load_pretrained()

    def load_pretrained(self):
        if self.in_tensor_size[1] == 1 or self.in_tensor_size[1] == 3:
            self.load_state_dict(map_pretrained(self.state_dict(),
                                                self.model_type))
        else:
            print(" ... pretrained not available")
            self.pretrained = False


# from core.NeuralLayers import ResidualOriginal, ResidualComplex,\
#     ResidualNeXt, SEResidualComplex, SEResidualNeXt, Convolution
# from core.utils import ImageNetNorm
# tensor_size = (1, 3, 224, 224)
# tensor = torch.rand(*tensor_size)
# test = ResidualNet(tensor_size, "r18", pretrained=True)
# test
# test(tensor).size()
# import torchvision.utils as tutils
# tutils.save_image(test.state_dict()['InitialConvolution.Convolution.weight'],
#     "./models/test_ws.png")
