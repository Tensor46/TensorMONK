""" TensorMONK :: layers :: attention's """

__all__ = ["SelfAttention", "LocalAttention"]

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
from .convolution import Convolution
from .utils import compute_flops


class SelfAttention(nn.Module):
    r"""Self-Attention (`"Self-Attention Generative Adversarial Networks"
    <https://arxiv.org/pdf/1805.08318.pdf>`_).

    Args:
        tensor_size (tuple, required): Input tensor shape in BCHW
            (None/any integer >0, channels, height, width).
        shrink (int, optional): Used to compute output channels of key and
            query, i.e, int(tensor_size[1] / shrink), (default = :obj:`8`).
        scale_factor (float, optional): Scale at which attention is computed.
            (use scale_factor <1 for speed). When scale_factor != 1, input is
            scaled using nearest neighbor interpolation (default = :obj:`1`).
        return_attention (bool, optional): When True, returns a tuple
            (output, attention) (default = :obj:`False`).

    :rtype: :class:`torch.Tensor`
    """

    def __init__(self,
                 tensor_size: tuple,
                 shrink: int = 8,
                 scale_factor: float = 1.,
                 return_attention: bool = False,
                 **kwargs):
        super(SelfAttention, self).__init__()

        if not isinstance(tensor_size, (list, tuple)):
            raise TypeError("SelfAttention: tensor_size must be tuple/list: "
                            "{}".format(type(tensor_size).__name__))
        tensor_size = tuple(tensor_size)
        if not len(tensor_size) == 4:
            raise ValueError("SelfAttention: tensor_size must be of length 4"
                             ": {}".format(len(tensor_size)))

        if not isinstance(shrink, int):
            raise TypeError("SelfAttention: shrink must be int: "
                            "{}".format(type(shrink).__name__))
        if not (tensor_size[1] >= shrink >= 1):
            raise TypeError("SelfAttention: shrink must be tensor_size[1] >= "
                            "shrink > 0: {}".format(shrink))
        self.shrink = shrink

        if not isinstance(scale_factor, float):
            raise TypeError("SelfAttention: scale_factor must be float: "
                            "{}".format(type(scale_factor).__name__))
        self.scale_factor = scale_factor

        if not isinstance(return_attention, bool):
            raise TypeError("SelfAttention: return_attention must be bool: "
                            "{}".format(type(return_attention).__name__))
        self.return_attention = return_attention

        self.oc = int(tensor_size[1] / shrink)
        self.key = Convolution(tensor_size, 1, self.oc, 1, True, None)
        self.query = Convolution(tensor_size, 1, self.oc, 1, True, None)
        self.value = Convolution(tensor_size, 1, tensor_size[1], 1, True, None)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.tensor_size = tensor_size

    def forward(self, tensor: torch.Tensor):
        if self.scale_factor != 1:
            o = F.interpolate(tensor, scale_factor=self.scale_factor)
            _tensor = tensor.clone()
            tensor = F.interpolate(tensor, scale_factor=self.scale_factor)
        n, c, h, w = tensor.shape

        key = self.key(tensor).view(n, -1, h*w)
        query = self.query(tensor).view(n, -1, h*w)
        value = self.value(tensor).view(n, -1, h*w)

        attention = F.softmax(torch.bmm(query.permute(0, 2, 1), key), dim=2)
        o = torch.bmm(value, attention.permute(0, 2, 1)).view(n, c, h, w)

        if self.scale_factor != 1:
            o = F.interpolate(o, size=_tensor.shape[2:])
            tensor = _tensor
        if self.return_attention:
            return tensor + o * self.gamma, attention
        return tensor + o * self.gamma

    def flops(self):
        flops = 0
        c, h, w = self.tensor_size[1:]
        if self.scale_factor != 1:
            # assuming nearest
            nh, nw = int(h*self.scale_factor), int(w*self.scale_factor)
            flops += (c*h*w + c*nh*nw) * 2
        # attention - bmm
        flops += ((2 * self.oc * self.oc) - 1) * ((h * w)**2)
        # attention - softmax
        flops += (h * w) * (h * w * 3)
        # o - bmm
        flops += c * ((2 * h * w) - 1) * h * w
        # tensor + o*gamma
        flops += c * h * w * 2
        return compute_flops(self) + flops


class LocalAttention(nn.Module):
    r"""LocalAttention (`"Stand-Alone Self-Attention in Vision Models"
    <https://arxiv.org/pdf/1906.05909.pdf>`_).

    Args:
        tensor_size (tuple, required): Input tensor shape in BCHW
            (None/any integer >0, channels, height, width).
        filter_size (int/tuple, required): size of kernel, integer or
            list/tuple of length 2.
        out_channels (int, required): output tensor.size(1)
        strides (int/tuple, optional): convolution stride (default = :obj:`1`).
        groups (int, optional): enables grouped convolution
            (default = :obj:`4`).
        bias (bool): When True, key, query & value 1x1 convolutions have bias
            (default = :obj:`False`).
        replicate_paper (bool, optional): When False, relative attention logic
            is different from that of paper (default = :obj:`True`).
        normalize_offset (bool, optional): When True (and replicate_paper =
            :obj:`False`), normalizes the row and column offsets
            (default = :obj:`False`).

    :rtype: :class:`torch.Tensor`
    """
    def __init__(self,
                 tensor_size: tuple,
                 filter_size: Union[int, tuple],
                 out_channels: int,
                 strides: int = 1,
                 groups: int = 4,
                 bias: bool = False,
                 replicate_paper: bool = True,
                 normalize_offset: bool = False,
                 **kwargs):
        super(LocalAttention, self).__init__()

        if not isinstance(tensor_size, (list, tuple)):
            raise TypeError("LocalAttention: tensor_size must be tuple/list: "
                            "{}".format(type(tensor_size).__name__))
        tensor_size = tuple(tensor_size)
        if not len(tensor_size) == 4:
            raise ValueError("LocalAttention: tensor_size must be of length 4"
                             ": {}".format(len(tensor_size)))
        if not isinstance(filter_size, (int, list, tuple)):
            raise TypeError("LocalAttention: filter_size must be int/tuple/"
                            "list: {}".format(type(filter_size).__name__))
        if isinstance(filter_size, int):
            filter_size = (filter_size, filter_size)
        filter_size = tuple(filter_size)
        if not len(filter_size) == 2:
            raise ValueError("LocalAttention: filter_size must be of length 2"
                             ": {}".format(len(filter_size)))
        if not isinstance(out_channels, int):
            raise TypeError("LocalAttention: out_channels must be int: "
                            "{}".format(type(out_channels).__name__))
        if not out_channels >= 1:
            raise ValueError("LocalAttention: out_channels must be >= 1: "
                             "{}".format(len(out_channels)))
        if not isinstance(strides, (int, list, tuple)):
            raise TypeError("LocalAttention: strides must be int/tuple/list: "
                            "{}".format(type(strides).__name__))
        if isinstance(strides, int):
            strides = (strides, strides)
        strides = tuple(strides)
        if not len(strides) == 2:
            raise ValueError("LocalAttention: strides must be of length 2: "
                             "{}".format(len(strides)))
        if not isinstance(groups, int):
            raise TypeError("LocalAttention: groups must be int: "
                            "{}".format(type(groups).__name__))
        if out_channels % groups or groups < 1:
            raise ValueError("LocalAttention: groups must be divisible by "
                             "out_channels and >=1:  {}".format(groups))
        if not isinstance(bias, bool):
            raise TypeError("LocalAttention: bias must be bool: "
                            "{}".format(type(bias).__name__))
        if not isinstance(replicate_paper, bool):
            raise TypeError("LocalAttention: replicate_paper must be bool: "
                            "{}".format(type(replicate_paper).__name__))
        if not isinstance(normalize_offset, bool):
            raise TypeError("LocalAttention: normalize_offset must be bool: "
                            "{}".format(type(normalize_offset).__name__))

        self.fs = filter_size
        self.st = strides
        self.gs = groups
        self.replicate_paper = replicate_paper
        ic = tensor_size[1]

        # 1x1 convolutions for spatial-relative attention
        self.query = nn.Conv2d(ic, out_channels, 1, self.st, bias=bias,
                               groups=groups)
        self.key = nn.Conv2d(ic, out_channels, 1, bias=bias, groups=groups)
        self.value = nn.Conv2d(ic, out_channels, 1, bias=bias, groups=groups)
        torch.nn.init.kaiming_normal_(self.query.weight)
        torch.nn.init.kaiming_normal_(self.key.weight)
        torch.nn.init.kaiming_normal_(self.value.weight)

        fh, fw = self.fs
        self.pad = (fw // 2 - int(fw % 2 == 0), fw // 2,
                    fh // 2 - int(fh % 2 == 0), fh // 2)

        # relative attention
        offset = torch.arange(fh).view(fh, 1).repeat(1, fw) - self.pad[2]
        self.register_buffer("row_offset", offset.view(-1).float())
        offset = torch.arange(fw).view(1, fw).repeat(fh, 1) - self.pad[0]
        self.register_buffer("col_offset", offset.view(-1).float())
        if normalize_offset and not replicate_paper:
            self.row_offset.data.div_(self.row_offset.abs().max())
            self.col_offset.data.div_(self.col_offset.abs().max())
        if replicate_paper:
            # as per paper
            self.row_w = nn.Parameter(torch.rand(fh*fw, out_channels//2))
            self.col_w = nn.Parameter(torch.rand(fh*fw,
                                      out_channels - out_channels//2))
        else:
            # made more logical sense
            self.row_w = nn.Parameter(torch.rand(fh*fw, out_channels))
            self.col_w = nn.Parameter(torch.rand(fh*fw, out_channels))
        torch.nn.init.normal_(self.row_w, 0, 1)
        torch.nn.init.normal_(self.col_w, 0, 1)
        self.in_size = tensor_size
        self.tensor_size = (
            None, out_channels,
            (tensor_size[2] + self.pad[2] + self.pad[3]) / self.st[0],
            (tensor_size[3] + self.pad[0] + self.pad[1]) / self.st[1])

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        n, c, h, w = tensor.shape
        fh, fw = self.fs

        # key, query and value
        k, q, v = self.key(tensor), self.query(tensor), self.value(tensor)
        oc, nh, nw = q.shape[1:]
        q = F.unfold(q, 1, padding=0, stride=1)
        q = q.view(n, oc, 1, nh, nw).contiguous()
        k = F.unfold(F.pad(k, self.pad), self.fs, stride=self.st)
        k = k.view(n, oc, fh * fw, nh, nw).contiguous()
        v = F.unfold(F.pad(v, self.pad), self.fs, stride=self.st)
        v = v.view(n, oc, fh * fw, nh, nw).contiguous()

        # encoding offsets
        if self.replicate_paper:  # as per paper
            r_ai_bi = torch.cat((self.row_offset @ self.row_w,
                                 self.col_offset @ self.col_w))
        else:  # made more logical sense
            r_ai_bi = ((self.row_offset @ self.row_w) +
                       (self.col_offset @ self.col_w))
        r_ai_bi = r_ai_bi.view(1, oc, 1, 1, 1).contiguous()

        # equation 3 - spatial-relative attention
        attention = (F.softmax(q * k + q * r_ai_bi, dim=2) * v).sum(dim=2)
        return attention

    def flops(self):
        flops = 0
        # key and value
        c, h, w = self.in_size[1:]
        nc, nh, nw = self.tensor_size[1:]
        flops += nc * c / self.gs * h * w
        flops += nc * c / self.gs * h * w
        # query
        flops += nc * c / self.gs * nh * nw
        # encoding
        flops += (self.row_offset.numel() * 2) * self.row_w.shape[-1]
        flops += (self.row_offset.numel() * 2) * self.row_w.shape[-1]
        # attention
        flops += (c * h * w * self.fs[0] * self.fs[1]) * 6
        return int(flops)


# from tensormonk.layers import Convolution
# from tensormonk.layers.utils import compute_flops
# tensor_size = (3, 16, 60, 60)
# x = torch.rand(*tensor_size)
# test = SelfAttention(tensor_size, 8, 1.)
# test(x)[1].shape
# %timeit test(x)[1].shape
# test = SelfAttention(tensor_size, 8, 0.25)
# test(x)[1].shape
# %timeit test(x)[1].shape
