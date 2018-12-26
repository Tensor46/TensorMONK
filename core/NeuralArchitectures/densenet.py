""" TensorMONK's :: NeuralArchitectures                                     """

import os
import wget
import torch
from ..NeuralLayers import Convolution, DenseBlock, Linear
from ..utils import ImageNetNorm


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

    # get list of unique modules (conv + norm)
    modules = []
    checks = [".Convolution.weight", ".Normalization.weight",
              ".Normalization.bias", ".Normalization.running_mean",
              ".Normalization.running_var"]
    for x in list(labels):
        if "num_batches_tracked" not in x and "embedding" not in x:
            for y in checks:
                if x.endswith(y):
                    x = x[:-len(y)]
            if len(modules) == 0:
                modules += [x]
            else:
                if x != modules[-1]:
                    modules += [x]

    pre_modules = []
    for x in list(prelabels):
        if x.endswith((".weight", ".bias")) and "conv" in x:
            for y in (".weight", ".bias"):
                if x.endswith(y):
                    x = x[:-len(y)]
            if len(pre_modules) == 0:
                pre_modules += [x]
            else:
                if x != pre_modules[-1]:
                    pre_modules += [x]

    # mapped parameters to pretrained network
    pairs = []
    for x in list(state_dict.keys()):
        if "num_batches_tracked" not in x and "embedding" not in x:
            idx = [i for i, y in enumerate(modules) if x.startswith(y)][0]
            tmp = pre_modules[idx]
            if ".Normalization" in x:
                tmp = tmp.replace(".conv", ".norm")
            pairs += [(x, tmp + "." + x.split(".")[-1])]
            assert tmp + "." + x.split(".")[-1] in prestate_dict.keys()

    # update the state_dict
    for x, y in pairs:
        if state_dict[x].size() == prestate_dict[y].size():
            state_dict[x] = prestate_dict[y]

    del prestate_dict
    return state_dict


class DenseNet(torch.nn.Sequential):
    r"""Versions of DenseNets. With the ability to change the strides of
    initial convolution and remove max pool the models works for all the
    min(height, width) >= 32. To replicate the paper, use default parameters
    (and select type). Implemented from https://arxiv.org/pdf/1608.06993.pdf

    Args:
        tensor_size: shape of tensor in BCHW
            (None/any integer >0, channels, height, width)
        type (string): model type
            Available models        type
            ============================
            DenseNet-121            d121
            DenseNet-169            d169
            DenseNet-201            d201
            DenseNet-264            d264

        activation: None/relu/relu6/lklu/elu/prelu/tanh/sigm/maxo/rmxo/swish
        normalization: None/batch/group/instance/layer/pixelwise
        pre_nm: if True, normalization -> activation -> convolution else
            convolution -> normalization -> activation
        groups: grouped convolution, value must be divisble by tensor_size[1]
            and out_channels, default = 1
        weight_nm: True/False, default = False
        shift: True/False, default = False
            Shift replaces 3x3 convolution with pointwise convs after shifting.
            Requires tensor_size[1] >= 9 and filter_size = 3
        n_embedding: when not None and > 0, adds a linear layer to the network
            and returns a torch.Tensor of shape (None, n_embedding)
        pretrained: downloads and updates the weights with pretrained weights
    """
    def __init__(self,
                 tensor_size=(6, 3, 224, 224),
                 type: str = "d121",
                 activation: str = "relu",
                 normalization: str = "batch",
                 pre_nm: bool = True,
                 groups: int = 1,
                 weight_nm: bool = False,
                 equalized: bool = False,
                 shift: bool = False,
                 n_embedding: int = None,
                 pretrained: bool = False,
                 *args, **kwargs):
        super(DenseNet, self).__init__()

        type = type.lower()
        assert type in ("d121", "d169", "d201", "d264"),\
            "DenseNet: type must be d121/d169/d201/d264"

        self.pretrained = pretrained
        if self.pretrained:
            assert tensor_size[1] == 3, \
                "DenseNet: tensor_size[1] == 3 is required for pretrained"
            activation, normalization, pre_nm = "relu", "batch", True
            groups, weight_nm, equalized, shift = 1, False, False, False

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

        print("Input", tensor_size)
        s = 2
        if min(tensor_size[2], tensor_size[3]) < 64:
            # Addon -- To make it flexible for other tensor_size's
            s = 1
            print("""DenseNet: Initial convolution strides changed from 2 to 1,
                as min(tensor_size[2], tensor_size[3]) <  64""")
        if pretrained:
            self.add_module("ImageNetNorm", ImageNetNorm())

        kwargs = {"activation": activation, "normalization": normalization,
                  "weight_nm": weight_nm, "equalized": equalized,
                  "shift": shift, "pad": True, "groups": groups}
        self.add_module("InitialConvolution",
                        Convolution(tensor_size, 7, 64, s,
                                    pre_nm=False, **kwargs))
        t_size = self.InitialConvolution.tensor_size
        print("InitialConvolution", t_size)

        if min(tensor_size[2], tensor_size[3]) > 128:
            self.add_module("MaxPool", torch.nn.MaxPool2d(3, 2, padding=1))
            h, w = t_size[2:]
            t_size = (1, 64, h//2 + (1 if h % 2 == 1 else 0),
                      w//2 + (1 if w % 2 == 1 else 0))
            print("MaxPool", t_size)
        else:
            # Addon -- To make it flexible for other tensor_size's
            print("DenseNet: MaxPool is ignored if min(h, w) <=  128")

        for i, n_blocks in enumerate(block_params):
            self.add_module("DenseBlock-"+str(i),
                            DenseBlock(t_size, 3, t_size[1]+n_blocks*k, 1,
                                       pre_nm=False if i == 0 else pre_nm,
                                       growth_rate=k, n_blocks=n_blocks,
                                       multiplier=4, **kwargs))
            t_size = getattr(self, "DenseBlock-"+str(i)).tensor_size
            print("DenseBlock-"+str(i), t_size)

            # Transition Block
            if i+1 == len(block_params):
                continue
            self.add_module("Transition-Shrink"+str(i),
                            Convolution(t_size, 1, t_size[1]//2, 1,
                                        pre_nm=False, **kwargs))
            self.add_module("Transition-Pool"+str(i),
                            torch.nn.AvgPool2d(2, 2))
            t_size = getattr(self, "Transition-Shrink"+str(i)).tensor_size
            t_size = (1, t_size[1], t_size[2]//2, t_size[3]//2)
            print("Transition-"+str(i), t_size)

        self.add_module("AveragePool", torch.nn.AvgPool2d(t_size[2:]))
        print("AveragePool", (1, t_size[1], 1, 1))
        self.tensor_size = (1, t_size[1])

        if n_embedding is not None and n_embedding > 0:
            self.add_module("Embedding", Linear(self.tensor_size, n_embedding,
                                                "", 0., False))
            self.tensor_size = (1, n_embedding)
            print("Linear", (1, n_embedding))

        if self.pretrained:
            self.load_pretrained()

    def load_pretrained(self):
        if self.in_tensor_size[1] == 1 or self.in_tensor_size[1] == 3:
            self.load_state_dict(map_pretrained(self.state_dict(), self.type))
        else:
            print(" ... pretrained not available")
            self.pretrained = False


# from core.NeuralLayers import Convolution, DenseBlock, Linear
# from core.utils import ImageNetNorm
# tensor_size = (1, 3, 224, 224)
# tensor = torch.rand(*tensor_size)
# test = DenseNet(tensor_size, "d121", pretrained=True)
# test(torch.rand(*tensor_size)).shape
# %timeit test(torch.rand(*tensor_size)).shape
# import torchvision.utils as tutils
# tutils.save_image(test.state_dict()['InitialConvolution.Convolution.weight'],
#                   "./models/ws.png")
