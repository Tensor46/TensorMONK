""" TensorMONK's :: NeuralLayers :: Convolution                              """

__all__ = ["Convolution", ]

import torch
import torch.nn as nn
import torch.nn.functional as F
from .activations import Activations
from .normalizations import Normalizations
# ============================================================================ #


class Convolution(nn.Module):
    """
        Parameters/Inputs
            tensor_size = (None/any integer >0, channels, height, width)
            filter_size = list(length=2)/tuple(length=2)/integer
            out_channels = return tensor.size(1)
            strides = list(length=2)/tuple(length=2)/integer
            pad = True/False
            activation = relu/relu6/lklu/tanh/sigm/maxo/rmxo/swish
            dropout = 0.-1.
            normalization = None/"batch"/"group"/"instance"/"layer"/"pixelwise"
            pre_nm = True/False
            groups = 1, ... out_channels
            weight_nm = True/False -- https://arxiv.org/pdf/1602.07868.pdf
            equalized = True/False -- https://arxiv.org/pdf/1710.10196.pdf
            shift = True/False -- https://arxiv.org/pdf/1711.08141.pdf
                In this implementation, shift replaces 3x3 with 1x1 by shifting.
                Further, requires tensor_size[1] >= 9 -- Not required per paper.
                Only works for a 3x3 Kernel.
    """
    def __init__(self,
                 tensor_size,
                 filter_size,
                 out_channels,
                 strides        = (1, 1),
                 pad            = True,
                 activation     = "relu",
                 dropout        = 0.,
                 normalization  = None,
                 pre_nm         = False,
                 groups         = 1,
                 weight_nm      = False,
                 equalized      = False,
                 shift          = False,
                 **kwargs):
        super(Convolution, self).__init__()
        # Checks
        assert len(tensor_size) == 4 and type(tensor_size) in [list, tuple], \
            "Convolution -- tensor_size must be of length 4 (tuple or list)"
        assert type(filter_size) in [int, list, tuple], \
            "Convolution -- filter_size must be int/tuple/list"
        if isinstance(filter_size, int):
            filter_size = (filter_size, filter_size)
        if isinstance(filter_size, list):
            filter_size = tuple(filter_size)
        assert len(filter_size) == 2, \
            "Convolution -- filter_size length must be 2"
        assert type(strides) in [int, list, tuple], \
            "Convolution -- strides must be int/tuple/list"
        if isinstance(strides, int):
            strides = (strides, strides)
        if isinstance(strides, list):
            strides = tuple(strides)
        assert len(strides) == 2, "Convolution -- strides length must be 2"
        assert isinstance(pad, bool), "Convolution -- pad must be boolean"
        assert isinstance(dropout, float), "Convolution -- dropout must be float"
        assert normalization in [None, "batch", "group", "instance", "layer", "pixelwise"], \
            "Convolution's normalization must be None/batch/group/instance/layer/pixelwise"
        assert isinstance(equalized, bool), "Convolution -- equalized must be boolean"
        assert isinstance(shift, bool), "Convolution -- shift must be boolean"
        self.equalized = equalized

        if shift and not (filter_size[0] == 3 and filter_size[1] == 3 and
                tensor_size[1] >= 9):
            shift = False
        if shift: filter_size = (1, 1)
        self.shift = shift

        if activation is not None: activation = activation.lower()
        dilation = kwargs["dilation"] if "dilation" in kwargs.keys() else (1, 1)

        # Modules
        padding = (filter_size[0]//2, filter_size[1]//2) if pad else 0
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
            self.Convolution = nn.utils.weight_norm(
                nn.Conv2d(tensor_size[1]//pre_expansion, out_channels*pst_expansion,
                filter_size, strides, padding, bias=False, groups=groups,
                dilation=dilation), name='weight')
        else:
            self.Convolution = nn.Conv2d(tensor_size[1]//pre_expansion,
                out_channels*pst_expansion, filter_size, strides, padding,
                bias=False, groups=groups, dilation=dilation)
            nn.init.kaiming_normal_(self.Convolution.weight,
                nn.init.calculate_gain("conv2d"))
            if equalized:
                import numpy as np
                gain = kwargs["gain"] if "gain" in kwargs.keys() else np.sqrt(2)
                fan_in = tensor_size[1] * out_channels * filter_size[0]
                self.scale = gain / np.sqrt(fan_in)
                self.Convolution.weight.data.mul_(self.scale)
            # nn.init.orthogonal_(self.Convolution.weight)
        # out tensor size
        self.tensor_size = (tensor_size[0], out_channels,
            int(1+(tensor_size[2] + (filter_size[0]//2 * 2 if pad else 0)
                - filter_size[0])/strides[0]),
            int(1+(tensor_size[3] + (filter_size[1]//2 * 2 if pad else 0)
                - filter_size[1])/strides[1]))
        if not pre_nm:
            if normalization is not None:
                self.Normalization = Normalizations((self.tensor_size[0],
                    out_channels*pst_expansion, self.tensor_size[2],
                    self.tensor_size[3]), normalization, **kwargs)
            if activation in ["relu", "relu6", "lklu", "tanh", "sigm", "maxo", "rmxo", "swish"]:
                self.Activation = Activations(activation)
        self.pre_nm = pre_nm

    def forward(self, tensor):
        if hasattr(self, "dropout"):
            tensor = self.dropout(tensor)
        if self.shift:
            tensor = self.shift_pixels(tensor)
        if self.pre_nm:
            if hasattr(self, "Normalization"): tensor = self.Normalization(tensor)
            if hasattr(self, "Activation"): tensor = self.Activation(tensor)
            tensor = self.Convolution(tensor)
            if self.equalized: tensor = tensor.mul(self.scale)
        else:
            tensor = self.Convolution(tensor)
            if self.equalized: tensor = tensor.mul(self.scale)
            if hasattr(self, "Normalization"): tensor = self.Normalization(tensor)
            if hasattr(self, "Activation"): tensor = self.Activation(tensor)
        return tensor

    def shift_pixels(self, tensor):
        if tensor.size(1) >= 9: # only for 3x3
            padded = F.pad(tensor, [1]*4)
            tensor[:, 0::9, :, :] = padded[:, 0::9,  :-2, 1:-1]
            tensor[:, 1::9, :, :] = padded[:, 1::9, 2:  , 1:-1]
            tensor[:, 2::9, :, :] = padded[:, 2::9, 1:-1,  :-2]
            tensor[:, 3::9, :, :] = padded[:, 3::9, 1:-1, 2:  ]

            tensor[:, 5::9, :, :] = padded[:, 5::9,  :-2,  :-2]
            tensor[:, 6::9, :, :] = padded[:, 6::9, 2:  , 2:  ]
            tensor[:, 7::9, :, :] = padded[:, 7::9,  :-2, 2:  ]
            tensor[:, 8::9, :, :] = padded[:, 8::9, 2:  ,  :-2]
        return tensor


# from core.NeuralLayers import Activations, Normalizations
# x = torch.rand(3, 18, 10, 10)
# test = Convolution((1, 18, 10, 10), (3,3), 36, (1, 1), True, shift=True)
# test.Convolution.weight.shape
# test(x).size()
