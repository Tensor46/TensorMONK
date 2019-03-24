""" TensorMONK :: layers :: RoutingCapsule """

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..activations import Activations


class RoutingCapsule(nn.Module):
    r""" Routing capsule from Dynamic Routing Between Capsules.
    Implemented -- https://arxiv.org/pdf/1710.09829.pdf

    Args:
        tensor_size: 5D shape of tensor from PrimaryCapsule
            (None/any integer >0, capsule_length, height, width, n_capsules)
        n_capsules (int, required): number of capsules, usually, number of
            labels per paper
        capsule_length (int, required): length of capsules
        iterations (int, required): routing iterations, default = 3

    Return:
        3D torch.Tensor of shape
            (None/any integer >0, n_capsules, capsule_length)
    """
    def __init__(self,
                 tensor_size,
                 n_capsules: int = 10,
                 capsule_length: int = 32,
                 iterations: int = 3,
                 *args, **kwargs):
        super(RoutingCapsule, self).__init__()
        self.iterations = iterations
        # Ex from paper
        #   For tensor_size=(1,32,6,6,8), n_capsules=10 and capsule_length=16
        #   weight_size = (tensor_size[1]*tensor_size[2]*tensor_size[3], \
        #                  tensor_size[4], n_capsules*capsule_length)
        #               = (32*6*6, 8 , 10*16)
        weight_size = (int(np.prod(tensor_size[1:-1])), tensor_size[-1],
                       n_capsules*capsule_length)
        self.weight = nn.Parameter(torch.randn(*weight_size).normal_(0., 0.1))
        self.activation = Activations((None, int(np.prod(tensor_size[1:-1])),
                                       tensor_size[-1]), "squash")
        self.tensor_size = (6, n_capsules, capsule_length)

    def forward(self, tensor):
        batch_size, primary_capsule_length, h, w, n_primary_capsules = \
            tensor.size()
        # Initial squash
        tensor = tensor.view(batch_size, -1, n_primary_capsules)
        tensor = self.activation(tensor)

        # from the given example:
        #   tensor is of size _ x 32 x 6 x 6 x 8
        #   after matrix mulitplication the size of u is _x32x6x6x10x16
        #   essentially, each of the pixel from 8 primary capsules is project
        #   to a dimension of n_capsules x capsule_length
        u = tensor.view(batch_size, -1, 1,
                        n_primary_capsules).matmul(self.weight)
        u = u.view(*((batch_size, primary_capsule_length, h, w) +
                     self.tensor_size[1:]))

        bias = torch.zeros(batch_size, primary_capsule_length, h, w,
                           self.tensor_size[1])
        if tensor.is_cuda:
            bias = bias.to(tensor.device)

        # routing
        for i in range(self.iterations):
            # softmax
            #   initial softmax gives equal probabilities (since bias is
            #   initialized with zeros), eventually, bias updates will change
            #   the probabilities
            c = F.softmax(bias, 4)  # size = _ x 32 x 6 x 6 x 10
            # could be done with a single sum after reorganizing the tensor's,
            #   however, retaining dimensions can explain better
            # s size without sum's = _ x 32 x 6 x 6 x 10 x 16
            # s size = _ x 10 x 16
            s = (c.unsqueeze(5)*u).sum(3).sum(2).sum(1)
            # squash -- v size = _ x 10 x 16
            v = self.activation(s)
            # bias update -- size = _ x 32 x 6 x 6 x 10
            if i < self.iterations-1:
                bias = bias + (u * v.view(batch_size, 1, 1, 1,
                                          self.tensor_size[1],
                                          self.tensor_size[2])).sum(5)
        return v

    def flops(self):
        # activations
        flops = self.activation.flops() * (1 + self.iterations)
        # matmul
        flops += np.prod(self.weight.shape) * self.weight.shape[1]
        # softmax
        flops += (self.weight.shape[0] * self.tensor_size[1] * 3) * \
            self.iterations
        # s computation
        flops += (self.weight.shape[0] * (self.weight.shape[2] + 1)) * \
            self.iterations
        # bias update _x32x6x6x10x16
        flops += self.weight.shape[0] * (self.weight.shape[2] + 2)
        return flops

# from tensormonk.activations import Activations
# x = torch.rand(3, 32, 10, 10, 8)
# test = RoutingCapsule((3, 32, 10, 10, 8), 10, 16, 3,)
# test(x).size()
# test.flops()
