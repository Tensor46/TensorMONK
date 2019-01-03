""" TensorMONK's :: NeuralLayers :: Convolution                             """

__all__ = ["Convolution", ]

import torch.nn as nn
import torch.nn.functional as F
from .activations import Activations
from .normalizations import Normalizations
import math


class Convolution(nn.Module):
    r"""2D convolutional layer with activations, normalizations and dropout
    included. Additionally, has weight normalization, equalized normalization,
    and shift. When transpose is True, nn.Conv2d is replaced by
    nn.ConvTranspose2d

    Args:
        tensor_size: shape of tensor in BCHW
            (None/any integer >0, channels, height, width)
        filter_size: size of kernel, integer or list/tuple of length 2.
            When shift = True and filter_size = 3, filter_size is changed to
            1x1 and shift operation is done  on 3x3 region.
        out_channels: output tensor.size(1)
        strides: integer or list/tuple of length 2, default = 1
        pad: True/False, default = True
        activation: None/relu/relu6/lklu/elu/prelu/tanh/sigm/maxo/rmxo/swish,
            default = relu
        dropout: 0. - 1., default = 0.
        normalization: None/batch/group/instance/layer/pixelwise, default= None
        pre_nm: if True, normalization -> activation -> convolution else
            convolution -> normalization -> activation
            default = False
        groups: grouped convolution, value must be divisble by tensor_size[1]
            and out_channels, default = 1
        weight_nm: True/False -- https://arxiv.org/pdf/1602.07868.pdf
            default = False
        equalized: True/False -- https://arxiv.org/pdf/1710.10196.pdf
            default = False
        shift: True/False -- https://arxiv.org/pdf/1711.08141.pdf
            Shift replaces 3x3 convolution with pointwise convs after shifting.
            Requires tensor_size[1] >= 9 and only works for a filter_size = 3
            default = False
        transpose: True/False, when True does nn.ConvTranspose2d. Transpose may
            have several possible outputs, you can readjust the built module
            tensor_size to achieve a specific shape. default = False
            Ex:
            test = Convolution((1, 18, 10, 10), 3, 36, 2, True, transpose=True)
            test.tensor_size = (3, 36, 20, 20)
            test(torch.rand((1, 18, 10, 10))).shape
        maintain_out_size: True/False, when True (and tranpose = True)
            output_tensor_size[2] = input_tensor_size[2]*strides[0]
            output_tensor_size[3] = input_tensor_size[3]*strides[1]
        bias: default=False

    Return:
        torch.Tensor of shape BCHW
    """
    def __init__(self,
                 tensor_size,
                 filter_size,
                 out_channels,
                 strides=1,
                 pad: bool = True,
                 activation: str = "relu",
                 dropout: float = 0.,
                 normalization: str = None,
                 pre_nm: bool = False,
                 groups: int = 1,
                 weight_nm: bool = False,
                 equalized: bool = False,
                 shift: bool = False,
                 transpose: bool = False,
                 maintain_out_size: bool = False,
                 bias: bool = False,
                 **kwargs):
        super(Convolution, self).__init__()
        self.t_size = tensor_size
        # Checks
        assert len(tensor_size) == 4 and type(tensor_size) in [list, tuple], \
            "Convolution: tensor_size must tuple/list of length 4"

        assert type(filter_size) in [int, list, tuple], \
            "Convolution: filter_size must be int/tuple/list"
        if isinstance(filter_size, int):
            filter_size = (filter_size, filter_size)
        if isinstance(filter_size, list):
            filter_size = tuple(filter_size)
        assert len(filter_size) == 2, \
            "Convolution: filter_size length must be 2"

        assert type(strides) in [int, list, tuple], \
            "Convolution: strides must be int/tuple/list"
        if isinstance(strides, int):
            strides = (strides, strides)
        if isinstance(strides, list):
            strides = tuple(strides)
        assert len(strides) == 2, "Convolution: strides length must be 2"

        assert isinstance(pad, bool), "Convolution: pad must be boolean"
        padding = (filter_size[0]//2, filter_size[1]//2) if pad else (0, 0)

        if isinstance(activation, str):
            activation = activation.lower()
        assert activation in [None, "", ] + Activations.available(),\
            "Linear: activation must be None/''/" + \
            "/".join(Activations.available())

        assert isinstance(dropout, float), "Convolution: dropout must be float"
        if dropout > 0.:
            self.dropout = nn.Dropout2d(dropout)

        assert normalization in [None] + Normalizations(available=True), \
            "Convolution's normalization must be None/" + \
            "/".join(Normalizations(available=True))
        assert isinstance(equalized, bool), \
            "Convolution: equalized must be boolean"
        self.equalized = False if weight_nm else equalized

        assert isinstance(shift, bool), "Convolution: shift must be boolean"
        if shift and not (filter_size[0] == 3 and filter_size[1] == 3 and
                          tensor_size[1] >= 9) and not transpose:
            shift = False
        if shift:
            filter_size, padding = (1, 1), (0, 0)
        self.shift = shift

        assert isinstance(transpose, bool), \
            "Convolution: transpose must be boolean"
        self.transpose = transpose

        pre_expansion = pst_expansion = 1
        if activation in ("maxo", "rmxo"):
            pre_expansion = 2 if pre_nm else 1
            pst_expansion = 1 if pre_nm else 2

        dilation = kwargs["dilation"] if "dilation" in kwargs.keys() and \
            not transpose else (1, 1)

        # out tensor size
        h, w = tensor_size[2:]
        if transpose:
            h = (h - 1) * strides[0] - 2*padding[0] + filter_size[0]
            w = (w - 1) * strides[1] - 2*padding[1] + filter_size[1]
        else:
            h = (h + 2*padding[0] - dilation[0]*(filter_size[0] - 1) - 1) / \
                strides[0] + 1
            w = (w + 2*padding[1] - dilation[1]*(filter_size[1] - 1) - 1) / \
                strides[1] + 1
        self.tensor_size = (tensor_size[0], out_channels,
                            math.floor(h), math.floor(w))

        # Modules
        if pre_nm and normalization is not None:
            self.Normalization = Normalizations(tensor_size, normalization,
                                                **kwargs)
        if pre_nm and activation in Activations.available():
            self.Activation = Activations(activation, tensor_size[1])
        self.conv_depth = out_channels*pst_expansion

        if transpose:
            out_pad = (0, 0)
            if maintain_out_size:
                out_pad = (tensor_size[2]*strides[0] - self.tensor_size[2],
                           tensor_size[3]*strides[1] - self.tensor_size[3])
                self.tensor_size = (tensor_size[0], out_channels,
                                    math.floor(h)+out_pad[0],
                                    math.floor(w)+out_pad[1])
            self.Convolution = \
                nn.ConvTranspose2d(tensor_size[1]//pre_expansion,
                                   out_channels*pst_expansion, filter_size,
                                   strides, padding, bias=bias, groups=groups,
                                   dilation=dilation, output_padding=out_pad)
        else:
            self.Convolution = nn.Conv2d(tensor_size[1]//pre_expansion,
                                         out_channels*pst_expansion,
                                         filter_size, strides, padding,
                                         bias=bias,
                                         groups=groups, dilation=dilation)

        nn.init.kaiming_normal_(self.Convolution.weight,
                                nn.init.calculate_gain("conv2d"))
        if weight_nm:
            self.Convolution = nn.utils.weight_norm(self.Convolution,
                                                    name="weight")
        if equalized and not weight_nm:
            import numpy as np
            gain = kwargs["gain"] if "gain" in kwargs.keys() else np.sqrt(2)
            fan_in = tensor_size[1] * out_channels * filter_size[0]
            self.scale = gain / np.sqrt(fan_in)
            self.Convolution.weight.data.mul_(self.scale)

        if (not pre_nm) and normalization is not None:
            t_size = (self.tensor_size[0], out_channels*pst_expansion,
                      self.tensor_size[2], self.tensor_size[3])
            self.Normalization = Normalizations(t_size, normalization,
                                                **kwargs)
        if (not pre_nm) and activation in Activations.available():
            self.Activation = Activations(activation,
                                          out_channels*pst_expansion)
        self.pre_nm = pre_nm
        self.activation = activation
        self.normalization = normalization

    def forward(self, tensor):
        if hasattr(self, "dropout"):
            tensor = self.dropout(tensor)
        if self.shift:
            tensor = self.shift_pixels(tensor)
        if self.pre_nm:  # normalization -> activation -> convolution
            if hasattr(self, "Normalization"):
                tensor = self.Normalization(tensor)
            if hasattr(self, "Activation"):
                tensor = self.Activation(tensor)
            tensor = self.Convolution(tensor)
            if self.equalized:
                tensor = tensor.mul(self.scale)
        else:  # convolution -> normalization -> activation
            tensor = self.Convolution(tensor)
            if self.equalized:
                tensor = tensor.mul(self.scale)
            if hasattr(self, "Normalization"):
                tensor = self.Normalization(tensor)
            if hasattr(self, "Activation"):
                tensor = self.Activation(tensor)
        return tensor

    def shift_pixels(self, tensor):
        if tensor.size(1) >= 9:  # only for 3x3
            padded = F.pad(tensor, [1]*4)
            tensor[:, 0::9, :, :] = padded[:, 0::9, :-2, 1:-1]
            tensor[:, 1::9, :, :] = padded[:, 1::9, 2:, 1:-1]
            tensor[:, 2::9, :, :] = padded[:, 2::9, 1:-1, :-2]
            tensor[:, 3::9, :, :] = padded[:, 3::9, 1:-1, 2:]

            tensor[:, 5::9, :, :] = padded[:, 5::9, :-2, :-2]
            tensor[:, 6::9, :, :] = padded[:, 6::9, 2:, 2:]
            tensor[:, 7::9, :, :] = padded[:, 7::9, :-2, 2:]
            tensor[:, 8::9, :, :] = padded[:, 8::9, 2:, :-2]
        return tensor

    def __repr__(self):
        ws = "x".join([str(x)for x in self.Convolution.weight.shape])
        cn = "{}({})".format("convT" if self.transpose else "conv", ws)
        nmac = "{}{}{}".format("" if self.normalization is None else
                               self.normalization, " -> " if self.normalization
                               else "", self.activation).rstrip(" -> ")
        osz = " -> " + "x".join(["_"]+[str(x)for x in self.tensor_size[1:]])
        isz = "x".join(["_"]+[str(x)for x in self.t_size[1:]]) + " -> "
        if self.pre_nm:
            nmac += (" -> " if len(nmac) > 1 else "")
        else:
            nmac = (" -> " if len(nmac) > 1 else "") + nmac
        if self.pre_nm:
            return "{}{}{}{}".format(isz, nmac, cn, osz)
        else:
            return "{}{}{}{}".format(isz, cn, nmac, osz)


# import torch
# from core.NeuralLayers import Activations, Normalizations
# x = torch.rand(3, 18, 10, 10)
# test = Convolution((1, 18, 10, 10), 3, 36, 2, True, "maxo", transpose=True,
#                    maintain_out_size=True)
# test.Convolution.weight.shape
# test(x).size()
# test.Convolution.weight.shape
