""" TensorMONK :: layers :: attention's """

__all__ = ["SelfAttention"]

import torch
import torch.nn as nn
import torch.nn.functional as F
from .convolution import Convolution
from .utils import compute_flops


class SelfAttention(nn.Module):
    r""" Self-Attention from Self-Attention Generative Adversarial Networks

    Args:
        tensor_size: shape of tensor in BCHW
            (None/any integer >0, channels, height, width)
        shrink (int, optional): used to compute output channels of key and
            query, i.e, tensor_size[1] / shrink, default = 8
        scale_factor (float, optional): Used to speedup the module by
            computing the attention at a lower scale (after interpolation).
    """

    def __init__(self, tensor_size, shrink=8, scale_factor=1.,
                 return_attention=False, **kwargs):
        super(SelfAttention, self).__init__()

        self.shrink = shrink
        self.scale_factor = scale_factor
        self.oc = int(tensor_size[1] / shrink)
        self.return_attention = return_attention

        self.key = Convolution(tensor_size, 1, self.oc, 1, True, None)
        self.query = Convolution(tensor_size, 1, self.oc, 1, True, None)
        self.value = Convolution(tensor_size, 1, tensor_size[1], 1, True, None)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.tensor_size = tensor_size

    def forward(self, tensor):
        if self.scale_factor != 1:
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
