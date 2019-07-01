""" TensorMONK :: architectures """

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers import Convolution, Linear, MBBlock


class EfficientNet(torch.nn.Module):
    r"""EfficientNet - With the ability to adjust to a given size (provide
    tensor_size with architecture=None to find best possible architecture).
    To replicate the paper, use default parameters (and select architecture).
    Implemented from https://arxiv.org/pdf/1905.11946.pdf

    Args:
        tensor_size: shape of tensor in BCHW
            (None/any integer >0, channels, height, width)
        architecture (string): model type
            Available models
            ================
            efficientnet-b0
            efficientnet-b1
            efficientnet-b2
            efficientnet-b3
            efficientnet-b4
            efficientnet-b5
            efficientnet-b6
            efficientnet-b7

            base_block: nn.Module = None,
            activation: str = "swish",
            normalization: str = "batch",
            pre_nm: bool = False,
            weight_nm: bool = False,
            equalized: bool = False,
            shift: bool = False,
            seblock: bool = True,
            n_embedding: int = None
        base_block: None or any modules from layers should work. When None,
            uses the default MBBlock from the official implementation.
        activation: None/relu/relu6/lklu/elu/prelu/tanh/sigm/maxo/rmxo/swish
            default = swish
        normalization: None/batch/group/instance/layer/pixelwise default=batch
        pre_nm: if True, normalization -> activation -> convolution else
            convolution -> normalization -> activation
            default=False
        weight_nm: True/False, default = False
        equalized: True/False, default = False
        shift: True/False, default = False
            Shift replaces 3x3 convolution with pointwise convs after shifting.
            Requires tensor_size[1] >= 9 and filter_size = 3
        n_embedding: when not None and > 0, adds a linear layer to the network
            and returns a torch.Tensor of shape (None, n_embedding)

    Return:
        embedding/predictions, a torch.Tensor
    """

    from collections import namedtuple
    model_config = namedtuple(
        "architecture", ("name", "width", "depth", "tensor_size", "dropout"))
    layer_config = namedtuple("layer", ("filter_size", "out_channels",
                              "strides", "expansion", "repeat"))

    def __init__(self,
                 tensor_size: tuple = (None, 3, None, None),
                 architecture: str = "efficientnet-b0",
                 base_block: nn.Module = None,
                 activation: str = "swish",
                 normalization: str = "batch",
                 pre_nm: bool = False,
                 weight_nm: bool = False,
                 equalized: bool = False,
                 shift: bool = False,
                 seblock: bool = True,
                 n_embedding: int = None):

        super(EfficientNet, self).__init__()
        if not (isinstance(tensor_size, list) or
                isinstance(tensor_size, tuple) or tensor_size is None):
            raise TypeError("EfficientNet: tensor_size is not valid: "
                            "{}".format(type(tensor_size).__name__))
        if architecture is not None:
            if not isinstance(architecture, str):
                raise TypeError("EfficientNet: architecture must be string: "
                                "{}".format(type(architecture).__name__))
        self.model_config = self.get_model_config(tensor_size, architecture)
        self.layer_configs = self.get_layer_configs(self.model_config)
        self.in_size = tuple([1, ] + list(self.model_config.tensor_size[1:]))

        block_kwargs = {"dropconnect": True, "activation": activation,
                        "normalization": normalization, "pre_nm": pre_nm,
                        "weight_nm": weight_nm, "equalized": equalized,
                        "shift": shift, "seblock": seblock}
        if base_block is None:
            base_block = MBBlock
        self.network = nn.Sequential(
            *self.build_modules(self.model_config, self.layer_configs,
                                base_block, block_kwargs))
        self.tensor_size = (None, self.network[-1].tensor_size[1])

        if n_embedding is not None and n_embedding > 0:
            self.add_module("embedding", Linear(self.tensor_size, n_embedding,
                                                "", 0., False))
            self.tensor_size = (1, n_embedding)
            print("Linear", (1, n_embedding))

    def forward(self, tensor):
        tensor = self.network(tensor)
        tensor = F.adaptive_avg_pool2d(tensor, 1)
        if hasattr(self, "embedding"):
            return self.embedding(tensor)

        return tensor.view(tensor.shape[0], -1)

    @staticmethod
    def available(architecture: str = None):
        r""" Known architectures, and their parameters """

        if architecture is None:
            # return all avaiable architecture name's
            return ("efficientnet-b0", "efficientnet-b1",
                    "efficientnet-b2", "efficientnet-b3",
                    "efficientnet-b4", "efficientnet-b5",
                    "efficientnet-b6", "efficientnet-b7")
        else:
            # return parameters of available an architecture
            if architecture not in EfficientNet.available():
                raise ValueError("EfficientNet: architecture is not valid: "
                                 "{}".format(architecture))
            assert architecture in EfficientNet.available(), \
                "EfficientNet: Unknown architecture"
            # width, depth, tensor_size, dropout
            params = ((1.0, 1.0, 224, 0.2), (1.0, 1.1, 240, 0.2),
                      (1.1, 1.2, 260, 0.3), (1.2, 1.4, 300, 0.3),
                      (1.4, 1.8, 380, 0.4), (1.6, 2.2, 456, 0.4),
                      (1.8, 2.6, 528, 0.5), (2.0, 3.1, 600, 0.5))
            index = EfficientNet.available().index(architecture)
            # name, width, depth, tensor_size, dropout
            return EfficientNet.model_config(architecture, *params[index])

    @staticmethod
    def find_architecture_given_size(tensor_size: tuple):
        r""" Find a closest architecture given tensor_size """
        import numpy as np
        mean = (tensor_size[2] + tensor_size[3]) / 2

        track = []
        for arch in EfficientNet.available():
            track.append(mean - EfficientNet.available(arch).tensor_size)
        track = np.abs(np.array(track))
        best = np.argmin(track)
        architecture = EfficientNet.available()[best]
        width = EfficientNet.available(architecture).tensor_size
        assert (track[best] / width) < .25, "No close match"
        return architecture

    @staticmethod
    def get_model_config(tensor_size: tuple, architecture: str):
        r""" Base config given tensor_size and architecture """
        if tensor_size is None:
            tensor_size = (None, 3, None, None)
        if architecture is None:
            assert tensor_size[2] is not None and tensor_size[3] is not None, \
                "EfficientNet: Both architecture and tensor[2:3] are None!"
            architecture = EfficientNet.find_architecture_given_size(
                tensor_size)

        config = EfficientNet.available(architecture)
        if tensor_size[2] is None or tensor_size[3] is None:
            tensor_size = (None, tensor_size[1], config.tensor_size,
                           config.tensor_size)
        return EfficientNet.model_config(architecture, config.width,
                                         config.depth, tensor_size,
                                         config.dropout)

    @staticmethod
    def get_layer_configs(model_config: tuple):
        r""" Generates all the required layer configurations! """
        def base_layer_configs():
            # filter_size, out_channels, strides, expansion, repeat
            all_layer_config = (
                EfficientNet.layer_config(3,  32, 2, 1, -1),  # stem
                EfficientNet.layer_config(3,  16, 1, 1, 1),  # MBBlocks or any
                EfficientNet.layer_config(3,  24, 2, 6, 2),  # MBBlocks or any
                EfficientNet.layer_config(5,  40, 2, 6, 2),  # MBBlocks or any
                EfficientNet.layer_config(3,  80, 2, 6, 3),  # MBBlocks or any
                EfficientNet.layer_config(5, 112, 1, 6, 3),  # MBBlocks or any
                EfficientNet.layer_config(5, 192, 2, 6, 4),  # MBBlocks or any
                EfficientNet.layer_config(3, 320, 1, 6, 1))  # MBBlocks or any
            return all_layer_config

        layer_configs = base_layer_configs()
        new_layer_configs = []
        for layer in layer_configs:
            # filter_size, out_channels, strides, expansion, repeat
            new_layer_configs += [EfficientNet.layer_config(
                layer.filter_size,
                EfficientNet.update_channels(layer.out_channels, model_config),
                layer.strides,
                layer.expansion,
                EfficientNet.update_repeat(layer.repeat, model_config))]
        return new_layer_configs

    @staticmethod
    def update_channels(channels, config, depth_divisor: int = 8):
        channels *= config.width
        new_channels = int(channels + depth_divisor / 2) // \
            depth_divisor * depth_divisor
        if new_channels < 0.9 * channels:
            new_channels += depth_divisor
        return new_channels

    @staticmethod
    def update_repeat(repeat, config):
        if repeat == -1:
            return repeat
        return math.ceil(repeat * config.depth)

    @staticmethod
    def build_modules(model_config, layer_configs, base_block, block_kwargs):
        modules = []
        for i, layer_config in enumerate(layer_configs):
            if i == 0:
                modules.append(Convolution(model_config.tensor_size,
                                           layer_config.filter_size,
                                           layer_config.out_channels,
                                           layer_config.strides,
                                           True, **block_kwargs))
                continue
            modules.append(base_block(modules[-1].tensor_size,
                                      layer_config.filter_size,
                                      layer_config.out_channels,
                                      layer_config.strides,
                                      dropout=model_config.dropout,
                                      expansion=layer_config.expansion,
                                      r=modules[-1].tensor_size[1]//4,
                                      **block_kwargs))
            for j in range(layer_config.repeat - 1):
                modules.append(base_block(modules[-1].tensor_size,
                                          layer_config.filter_size,
                                          layer_config.out_channels,
                                          strides=1,
                                          dropout=model_config.dropout,
                                          expansion=layer_config.expansion,
                                          r=modules[-1].tensor_size[1]//4,
                                          **block_kwargs))
        modules.append(Convolution(modules[-1].tensor_size, 1,
                                   modules[-1].tensor_size[1] * 4,
                                   1, **block_kwargs))
        return modules


# from tensormonk.layers import Convolution, Linear, MBBlock
# test = EfficientNet(architecture="efficientnet-b0")
# %timeit test(torch.rand(*test.in_size)).shape
# test = EfficientNet(architecture="efficientnet-b2")
# %timeit test(torch.rand(*test.in_size)).shape
# test = EfficientNet(architecture="efficientnet-b4")
# %timeit test(torch.rand(*test.in_size)).shape
# test = EfficientNet(architecture="efficientnet-b6")
# %timeit test(torch.rand(*test.in_size)).shape
