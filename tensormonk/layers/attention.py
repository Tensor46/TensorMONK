""" TensorMONK :: layers :: attention's """

__all__ = ["SelfAttention"]

import torch
import torch.nn as nn
import torch.nn.functional as F
from .convolution import Convolution


class SelfAttention(nn.Module):
    r""" Self-Attention from Self-Attention Generative Adversarial Networks

    Args:
        shrink (int, optional): used to compute output channels of key and
            query, i.e, tensor_size[1] / shrink, default = 8
        scale_factor (float, optional): Used to speedup the module by
            computing the attention at a lower scale (after interpolation).
    """

    def __init__(self, tensor_size, shrink=8, scale_factor=1., **kwargs):
        super(SelfAttention, self).__init__()

        self.scale_factor = scale_factor
        oc = int(tensor_size[1] / shrink)

        self.key = Convolution(tensor_size, 1, oc, 1, True, None)
        self.query = Convolution(tensor_size, 1, oc, 1, True, None)
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
            return _tensor + o, attention
        return tensor + o, attention


# from tensormonk.layers import Convolution
# tensor_size = (3, 16, 60, 60)
# x = torch.rand(*tensor_size)
# test = SelfAttention(tensor_size, 8, 1.)
# test(x)[1].shape
# %timeit test(x)[1].shape
# test = SelfAttention(tensor_size, 8, 0.25)
# test(x)[1].shape
# %timeit test(x)[1].shape
