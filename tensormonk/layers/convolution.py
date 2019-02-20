""" TensorMONK :: layers :: Convolution """

__all__ = ["Convolution", ]

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..activations import Activations
from ..normalizations import Normalizations
from ..regularizations import DropOut
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
        bias: default=False
        dropblock: Uses dropblock instead of 2D dropout, default=False
            all the inputs for DropBlock (refer DropBlock) except tensor_size
            and p (dropout) are gathered from kwargs when available
            default = True
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
                 bias: bool = False,
                 dropblock: bool = True,
                 transpose: bool = False,
                 maintain_out_size: bool = False,
                 **kwargs):
        super(Convolution, self).__init__()
        show_msg = "x".join(["_"]+[str(x)for x in tensor_size[1:]]) + " -> "
        # Checks
        if not type(tensor_size) in [list, tuple]:
            raise TypeError("Convolution: tensor_size must be tuple/list: "
                            "{}".format(type(tensor_size).__name__))
        tensor_size = tuple(tensor_size)
        if not len(tensor_size) == 4:
            raise ValueError("Convolution: tensor_size must be of length 4: "
                             "{}".format(len(tensor_size)))

        if not type(filter_size) in [int, list, tuple]:
            raise TypeError("Convolution: filter_size must be int/tuple/list: "
                            "{}".format(type(filter_size).__name__))
        if isinstance(filter_size, int):
            filter_size = (filter_size, filter_size)
        filter_size = tuple(filter_size)
        if not len(filter_size) == 2:
            raise ValueError("Convolution: filter_size must be of length 2: "
                             "{}".format(len(filter_size)))

        if not type(strides) in [int, list, tuple]:
            raise TypeError("Convolution: strides must be int/tuple/list: "
                            "{}".format(type(strides).__name__))
        if isinstance(strides, int):
            strides = (strides, strides)
        strides = tuple(strides)
        if not len(strides) == 2:
            raise ValueError("Convolution: strides must be of length 2: "
                             "{}".format(len(strides)))

        # pad: designed for odd kernels to replicate h & w when strides = 1
        if not type(pad) == bool:
            raise TypeError("Convolution: pad must be boolean: "
                            "{}".format(type(pad).__name__))
        padding = (filter_size[0]//2, filter_size[1]//2) if pad else (0, 0)

        if not (activation is None or type(activation) == str):
            raise TypeError("Convolution: activation must be None/str: "
                            "{}".format(type(activation).__name__))
        if isinstance(activation, str):
            activation = activation.lower()
        if not (activation in [None, "", ] + Activations.available()):
            raise ValueError("Convolution: Invalid activation: "
                             "{}".format(activation))
        pre_expansion = pst_expansion = 1
        if activation in ("maxo", "rmxo"):  # compensate channels for maxout
            pre_expansion = 2 if pre_nm else 1
            pst_expansion = 1 if pre_nm else 2

        if not type(dropout) is float:
            raise TypeError("Convolution: dropout must be float: "
                            "{}".format(type(dropout).__name__))
        if not 0. <= dropout < 1:
            raise ValueError("Convolution: dropout must be >=0 and <1: "
                             "{}".format(dropout))
        self.dropout = DropOut(tensor_size, dropout, dropblock, **kwargs)
        if self.dropout is not None:
            show_msg += ("dropblock" if dropblock else "dropout2d") + " -> "

        if not (normalization is None or type(normalization) == str):
            raise TypeError("Convolution: normalization must be None/str: "
                            "{}".format(type(normalization).__name__))
        if isinstance(normalization, str):
            normalization = normalization.lower()
        if not (normalization in [None] + Normalizations(available=True)):
            raise ValueError("Convolution: Invalid normalization: "
                             "{}".format(normalization))

        if not type(pre_nm) == bool:
            raise TypeError("Convolution: pre_nm must be boolean: "
                            "{}".format(type(pre_nm).__name__))

        if not type(groups) == int:
            raise TypeError("Convolution: groups must be int: "
                            "{}".format(type(groups).__name__))

        if not type(weight_nm) == bool:
            raise TypeError("Convolution: weight_nm must be boolean: "
                            "{}".format(type(weight_nm).__name__))

        if not type(equalized) == bool:
            raise TypeError("Convolution: equalized must be boolean: "
                            "{}".format(type(equalized).__name__))
        equalized = False if weight_nm else equalized

        if not type(shift) == bool:
            raise TypeError("Convolution: shift must be boolean: "
                            "{}".format(type(shift).__name__))
        if shift and transpose:
            raise ValueError("Convolution: both shift and transpose are True")
        if shift:
            if not (filter_size[0] == 3 and filter_size[1] == 3 and
                    tensor_size[1] >= 9):
                raise ValueError("Convolution: if shift=True, filter_size "
                                 "must be 3x3: {}".format(filter_size))
        if shift:
            filter_size, padding = (1, 1), (0, 0)

        if not type(bias) == bool:
            raise TypeError("Convolution: bias must be boolean: "
                            "{}".format(type(bias).__name__))

        if not type(dropblock) == bool:
            raise TypeError("Convolution: dropblock must be boolean: "
                            "{}".format(type(dropblock).__name__))

        if not type(transpose) == bool:
            raise TypeError("Convolution: transpose must be boolean: "
                            "{}".format(type(transpose).__name__))

        if not type(maintain_out_size) == bool:
            raise TypeError("Convolution: maintain_out_size must be boolean: "
                            "{}".format(type(maintain_out_size).__name__))

        dilation = kwargs["dilation"] if "dilation" in kwargs.keys() and \
            not transpose else (1, 1)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        if isinstance(dilation, list):
            dilation = tuple(dilation)
        if dilation[0] > 1 or dilation[1] > 1:
            padding = (padding[0]*dilation[0], padding[1]*dilation[1])

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
            self.Normalization = Normalizations(tensor_size,
                                                normalization, **kwargs)
            show_msg += normalization + " -> "
        if pre_nm and activation in Activations.available():
            self.Activation = Activations(activation, tensor_size[1], **kwargs)
            show_msg += activation + " -> "

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
            show_msg += "convT({}) -> ".format(
                "x".join(map(str, self.Convolution.weight.shape)))
        else:
            if shift:
                # self.add_module("Shift", Shift3x3())
                show_msg += "shift -> "
            self.Convolution = \
                nn.Conv2d(tensor_size[1]//pre_expansion,
                          out_channels*pst_expansion, filter_size, strides,
                          padding, bias=bias, groups=groups, dilation=dilation)
            show_msg += "conv({}) -> ".format(
                "x".join(map(str, self.Convolution.weight.shape)))

        nn.init.kaiming_normal_(self.Convolution.weight,
                                nn.init.calculate_gain("conv2d"))
        if weight_nm:
            self.Convolution = nn.utils.weight_norm(self.Convolution,
                                                    name="weight")

        if equalized:
            import numpy as np
            fan_in = (tensor_size[1]//pre_expansion) * filter_size[0] * \
                filter_size[1]
            self.scale = math.sqrt(2 / np.sqrt(fan_in))
            self.Convolution.weight.data.normal_(0, 1).div_(self.scale)

        if (not pre_nm) and normalization is not None:
            t_size = (self.tensor_size[0], out_channels*pst_expansion,
                      self.tensor_size[2], self.tensor_size[3])
            self.Normalization = Normalizations(t_size, normalization,
                                                **kwargs)
            show_msg += normalization + " -> "
        if (not pre_nm) and activation in Activations.available():
            self.Activation = Activations(activation,
                                          out_channels*pst_expansion, **kwargs)
            show_msg += activation + " -> "

        show_msg += "x".join(["_"]+[str(x)for x in self.tensor_size[1:]])
        self.show_msg = show_msg

        self.pre_nm = pre_nm
        self.shift = shift
        self.equalized = equalized

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            tensor = self.dropout(tensor)
        if self.pre_nm:  # normalization -> activation -> convolution
            if hasattr(self, "Normalization"):
                tensor = self.Normalization(tensor)
            if hasattr(self, "Activation"):
                tensor = self.Activation(tensor)

        if self.shift:
            tensor = self.shift_pixels(tensor)
        tensor = self.Convolution(tensor)
        if self.equalized:
            tensor = tensor * self.scale

        if not self.pre_nm:  # convolution -> normalization -> activation
            if hasattr(self, "Normalization"):
                tensor = self.Normalization(tensor)
            if hasattr(self, "Activation"):
                tensor = self.Activation(tensor)
        return tensor

    def shift_pixels(self, tensor: torch.Tensor) -> torch.Tensor:
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
        return self.show_msg

    def equalize_w(self):
        w_bound = self.Convolution.weight.data.abs().max()
        self.Convolution.weight.data.mul_(self.scale).div_(w_bound)


# from tensormonk.activations import Activations
# from tensormonk.normalizations import Normalizations
# from tensormonk.regularizations import DropOut
# x = torch.rand(3, 18, 10, 10)
# test = Convolution((1, 18, 10, 10), 3, 36, 2, True, "relu", 0.1, "batch",
#                    False, equalized=True, shift=True)
# test.Convolution.weight.shape
# test(x).size()
# test
