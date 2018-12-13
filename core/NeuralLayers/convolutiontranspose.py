""" TensorMONK's :: NeuralLayers :: ConvolutionTranspose                     """

__all__ = ["ConvolutionTranspose", ]

import torch
import torch.nn as nn
from .activations import Activations
from .normalizations import Normalizations
# ============================================================================ #


class ConvolutionTranspose(nn.Module):
    """
        Parameters/Inputs
            tensor_size = (None/any integer >0, channels, height, width)
            filter_size = list(length=2)/tuple(length=2)/integer
            out_channels = return tensor.size(1)
            strides = list(length=2)/tuple(length=2)/integer
            pad = True/False
            activation = "relu"/"relu6"/"lklu"/"tanh"/"sigm"/"maxo"/"rmxo"/"swish"
            dropout = 0.-1.
            normalization = None/"batch"/"group"/"instance"/"layer"/"pixelwise"
            pre_nm = True/False
            groups = 1, ... out_channels
            weight_nm = True/False -- https://arxiv.org/pdf/1602.07868.pdf
            equalized = True/False -- https://arxiv.org/pdf/1710.10196.pdf
    """
    def __init__(self, tensor_size, filter_size, out_channels, strides=(1, 1),
                 pad=True, activation="relu", dropout=0., normalization=None,
                 pre_nm=False, groups=1, weight_nm=False, equalized=False, **kwargs):
        super(ConvolutionTranspose, self).__init__()
        # Checks
        assert len(tensor_size) == 4 and type(tensor_size) in [list, tuple], \
            "ConvolutionTranspose -- tensor_size must be of length 4 (tuple or list)"
        assert type(filter_size) in [int, list, tuple], \
            "Convolution -- filter_size must be int/tuple/list"
        if isinstance(filter_size, int):
            filter_size = (filter_size, filter_size)
        if isinstance(filter_size, list):
            filter_size = tuple(filter_size)
        assert len(filter_size) == 2, \
            "ConvolutionTranspose -- filter_size length must be 2"
        assert type(strides) in [int, list, tuple], \
            "ConvolutionTranspose -- strides must be int/tuple/list"
        if isinstance(strides, int):
            strides = (strides, strides)
        if isinstance(strides, list):
            strides = tuple(strides)
        assert len(strides) == 2, "ConvolutionTranspose -- strides length must be 2"
        assert isinstance(pad, bool), "ConvolutionTranspose -- pad must be boolean"
        assert isinstance(dropout, float), "ConvolutionTranspose -- dropout must be float"
        assert normalization in [None, "batch", "group", "instance", "layer", "pixelwise"], \
            "Convolution's normalization must be None/batch/group/instance/layer/pixelwise"
        assert isinstance(equalized, bool), "Convolution -- equalized must be boolean"
        self.equalized = equalized
        if activation is not None: activation = activation.lower()
        dilation = kwargs["dilation"] if "dilation" in kwargs.keys() else (1, 1)
        # Modules
        padding = (filter_size[0]//2, filter_size[1]//2) if pad else (0,0)
        if dropout > 0.:
            self.dropout = nn.Dropout2d(dropout)
        pre_expansion, pst_expansion = 1, 1
        if activation in ("maxo", "rmxo"):
            if pre_nm: pre_expansion = 2
            if not pre_nm: pst_expansion = 2
        if pre_nm:
            if normalization is not None:
                self.Normalization = Normalizations(tensor_size, normalization, **kwargs)
            if activation in ["relu", "relu6", "lklu", "tanh", "sigm", "maxo", "rmxo", "swish"]:
                self.Activation = Activations(activation)
        if weight_nm:
            self.ConvolutionTranspose = nn.utils.weight_norm(
                nn.ConvTranspose2d(tensor_size[1]//pre_expansion,
                out_channels*pst_expansion, filter_size, strides, padding,
                bias=False, dilation=dilation, groups=groups), name='weight')
        else:
            self.ConvolutionTranspose = nn.ConvTranspose2d(tensor_size[1]//pre_expansion,
                out_channels*pst_expansion, filter_size, strides, padding,
                bias=False, groups=groups)
            nn.init.kaiming_normal_(self.ConvolutionTranspose.weight,
                nn.init.calculate_gain("conv2d"))
            if equalized:
                import numpy as np
                gain = kwargs["gain"] if "gain" in kwargs.keys() else np.sqrt(2)
                fan_in = tensor_size[1] * out_channels * filter_size[0]
                self.scale = gain / np.sqrt(fan_in)
                self.ConvolutionTranspose.weight.data.mul_(self.scale)
            # nn.init.orthogonal_(self.ConvolutionTranspose.weight)
        self.oc = self.ConvolutionTranspose.weight.data.size(0)
        # out tensor size
        self.tensor_size = (tensor_size[0], out_channels,
            (tensor_size[2] - 1)*strides[0] - 2*padding[0] + filter_size[0],
            (tensor_size[3] - 1)*strides[1] - 2*padding[1] + filter_size[1],)
        if not pre_nm:
            if normalization is not None:
                self.Normalization = Normalizations((self.tensor_size[0],
                    out_channels*pst_expansion, self.tensor_size[2],
                    self.tensor_size[3]), normalization, **kwargs)
            if activation in ["relu", "relu6", "lklu", "tanh", "sigm", "maxo", "rmxo", "swish"]:
                self.Activation = Activations(activation)
        self.pre_nm = pre_nm

    def forward(self, tensor, output_size=None):
        if hasattr(self, "dropout"):
            tensor = self.dropout(tensor)
        if self.pre_nm:
            if hasattr(self, "Normalization"): tensor = self.Normalization(tensor)
            if hasattr(self, "Activation"): tensor = self.Activation(tensor)
            if output_size is None:
                output_size = self.tensor_size
            output_size = (tensor.size(0), self.oc, output_size[2], output_size[3])
            tensor = self.ConvolutionTranspose(tensor, output_size=output_size)
            if self.equalized: tensor = tensor.mul(self.scale)
        else:
            if output_size is None:
                output_size = self.tensor_size
            output_size = (tensor.size(0), self.oc, output_size[2], output_size[3])
            tensor = self.ConvolutionTranspose(tensor, output_size=output_size)
            if self.equalized: tensor = tensor.mul(self.scale)
            if hasattr(self, "Normalization"): tensor = self.Normalization(tensor)
            if hasattr(self, "Activation"): tensor = self.Activation(tensor)
        return tensor

# from core.NeuralLayers import Activations, Normalizations
# x = torch.rand(3,8,10,10)
# test = ConvolutionTranspose((1,8,10,10), (3,3), 16, (2,2), True, "rmxo", 0.5, "instance", False, equalized=True)
# test(x,).size()
# test(x, (1, 16, 20, 20)).size()
