""" tensorMONK's :: neuralLayers :: PrimaryCapsule                           """

import torch.nn as nn
from .Convolution import Convolution
# ============================================================================ #


class PrimaryCapsule(nn.Module):
    """ https://arxiv.org/pdf/1710.09829.pdf """
    def __init__(self, tensor_size, filter_size, out_channels, strides=(1, 1), pad=True,
                 activation="relu", dropout=0., batch_nm=False, pre_nm=False,
                 growth_rate=32, block=Convolution, n_capsules=8, capsule_length=32, *args, **kwargs):
        super(PrimaryCapsule, self).__init__()
        assert out_channels == n_capsules*capsule_length, "PrimaryCapsule -- out_channels!=n_capsules*capsule_length"
        self.primaryCapsules = block(tensor_size, filter_size, out_channels, strides, pad,
                                     activation, dropout, batch_nm, pre_nm, growth_rate =growth_rate)
        self.tensor_size = (6, capsule_length) + self.primaryCapsules.tensor_size[2:] + (n_capsules,)

    def forward(self, tensor):
        tensor = self.primaryCapsules(tensor)
        tensor = tensor.view(-1, self.tensor_size[1], self.tensor_size[4], self.tensor_size[2], self.tensor_size[3])
        return tensor.permute(0, 1, 3, 4, 2).contiguous()

# import torch
# x = torch.rand(3,3,10,10)
# test = PrimaryCapsule((1,3,10,10), (3,3), 256, (2,2), True, "relu", 0., True, False,
#                       block=Convolution, n_capsules=8, capsule_length=32)
# test(x).size()
