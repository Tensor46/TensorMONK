""" TensorMONK :: layers :: CondConv2d """

__all__ = ["CondConv2d"]

import torch
import torch.nn as nn
import torch.nn.functional as F


class CondConv2d(torch.nn.Module):
    r"""Conditional Convolution (`"CondConv: Conditionally Parameterized
    Convolutions for Efficient Inference"
    <https://arxiv.org/pdf/1904.04971v2.pdf>`_).

    Args:
        tensor_size (tuple, required): Input tensor shape in BCHW
            (None/any integer >0, channels, height, width).
        n_kernels (int, required): number of kernels that are used for routing.
        filter_size (tuple/int, required): size of kernel, integer or tuple of
            length 2.
        out_channels (int, required): output tensor.size(1)
        strides (int/tuple, optional): integer or tuple of length 2,
            (default=:obj:`1`).
        pad (bool, optional): When True, pads to replicates input size for
            strides=1 (default=:obj:`True`).
        groups (int, optional): Enables grouped convolution (default=:obj:`1`).

    :rtype: :class:`torch.Tensor`

    # TODO: Include normalization and activation similar to Convolution?
    """
    def __init__(self,
                 tensor_size: tuple,
                 n_experts: int,
                 filter_size: int,
                 out_channels: int,
                 strides: int = 1,
                 pad: bool = True,
                 groups: int = 1):
        super(CondConv2d, self).__init__()

        if not isinstance(tensor_size, (list, tuple)):
            raise TypeError("CondConv2d: tensor_size must be tuple/list: "
                            "{}".format(type(tensor_size).__name__))
        tensor_size = tuple(tensor_size)
        if not len(tensor_size) == 4:
            raise ValueError("CondConv2d: tensor_size must be of length 4: "
                             "{}".format(len(tensor_size)))
        self.t_size = tensor_size

        if not isinstance(filter_size, (int, list, tuple)):
            raise TypeError("CondConv2d: filter_size must be int/tuple/list: "
                            "{}".format(type(filter_size).__name__))
        if isinstance(filter_size, int):
            filter_size = (filter_size, filter_size)
        filter_size = tuple(filter_size)
        if not len(filter_size) == 2:
            raise ValueError("CondConv2d: filter_size must be of length 2: "
                             "{}".format(len(filter_size)))

        if not isinstance(n_experts, int):
            raise TypeError("CondConv2d: n_experts must be int: "
                            "{}".format(type(n_experts).__name__))
        if not (n_experts > 1):
            raise ValueError("CondConv2d: n_experts must be >= 2: "
                             "{}".format(n_experts))

        if not type(out_channels) == int:
            raise TypeError("CondConv2d: out_channels must be int: "
                            "{}".format(type(out_channels).__name__))
        if not (out_channels >= 1):
            raise ValueError("CondConv2d: out_channels must be >= 1: "
                             "{}".format(groups))

        if not isinstance(strides, (int, list, tuple)):
            raise TypeError("CondConv2d: strides must be int/tuple/list: "
                            "{}".format(type(strides).__name__))
        if isinstance(strides, int):
            strides = (strides, strides)
        strides = tuple(strides)
        if not len(strides) == 2:
            raise ValueError("CondConv2d: strides must be of length 2: "
                             "{}".format(len(strides)))
        self.strides = strides

        if not type(groups) == int:
            raise TypeError("CondConv2d: groups must be int: "
                            "{}".format(type(groups).__name__))
        if tensor_size[1] % groups != 0:
            raise ValueError("CondConv2d: groups must be divisble by input "
                             "channels: {}".format(groups))

        c, (fh, fw) = tensor_size[1], filter_size
        # routing weights
        self.routing_ws = nn.Parameter(torch.randn(tensor_size[1], n_experts))
        nn.init.kaiming_normal_(self.routing_ws)
        self.routing_ws.data.mul_(0.1)
        # convolutional weights
        self.weight = nn.Parameter(
            torch.randn(n_experts, out_channels, c // groups, fh, fw))
        nn.init.kaiming_normal_(self.weight)
        self.weight.data.mul_(0.1)
        self.compute_osize(tensor_size, pad)

    def forward(self, tensor: torch.Tensor):
        n, c, h, w = tensor.shape
        n_experts, oc, ic, fh, fw = self.weight.shape
        # routing
        o = F.adaptive_avg_pool2d(tensor, 1).view(n, c).contiguous()
        routing = o @ self.routing_ws
        routing = routing.sigmoid()
        # replicate for all the channels
        routing = routing.repeat_interleave(oc, dim=1).contiguous()
        routing = routing.view(n, n_experts, oc, 1, 1, 1)
        # get convolution weights per sample -- dim-1 is n_experts
        ws = (routing * self.weight.unsqueeze(0)).sum(1)
        # convolution
        if self.pad is not None:
            tensor = F.pad(tensor, self.pad)
            n, c, h, w = tensor.shape
        o = F.conv2d(tensor.view(1, n*c, h, w),
                     ws.view(-1, ic, fh, fw),
                     stride=self.strides,
                     groups=n * (c // ic))
        return o.view(n, oc, o.size(-2), o.size(-1)).contiguous()

    def __repr__(self):
        isz = "Bx" + "x".join(map(str, self.t_size[1:]))
        osz = "Bx" + "x".join(map(str, self.tensor_size[1:]))
        return "CondConv2d: n_experts={}; {} -> {}".format(
            self.weight.shape[0], isz, osz)

    def compute_osize(self, tensor_size: tuple, pad: bool):
        if not pad:
            self.pad = None
            tensor = torch.rand(1, *tensor_size[1:])
            with torch.no_grad():
                t_size = F.conv2d(tensor, self.weight[0].data,
                                  stride=self.strides).shape
            self.tensor_size = (None, self.weight.shape[-4],
                                t_size[2], t_size[3])
        else:
            _, _, h, w = tensor_size
            sh, sw = self.strides
            fh, fw = self.weight.shape[-2], self.weight.shape[-1]
            nh = h if sh == 1 else (h // 2 + (h % 2 > 0))
            nw = w if sw == 1 else (w // 2 + (w % 2 > 0))
            ph = max((nh - 1) * sh + fh - h, 0)
            pw = max((nw - 1) * sw + fw - w, 0)
            self.pad = (pw - pw // 2, pw // 2, ph - ph // 2, ph // 2)
            self.tensor_size = (None, self.weight.shape[-4], nh, nw)
