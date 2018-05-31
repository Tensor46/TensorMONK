""" TensorMONK's :: NeuralLayers :: RoutingCapsule                           """

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
#==============================================================================#


class RoutingCapsule(nn.Module):
    """ https://arxiv.org/pdf/1710.09829.pdf """
    def __init__(self, tensor_size, n_capsules=8, capsule_length=32, iterations=3, *args, **kwargs):
        super(RoutingCapsule, self).__init__()
        import numpy as np
        self.iterations = iterations
        # Ex: For tensor_size=(1,32,6,6,8), n_capsules=8 and capsule_length=16
        # weight_size = (tensor_size[1]*tensor_size[2]*tensor_size[3], tensor_size[4], n_capsules*capsule_length)
        #          = (32*6*6, 8 , 8*16)
        weight_size = (np.prod(tensor_size[1:-1]), tensor_size[-1], n_capsules*capsule_length)
        self.weight = nn.Parameter(torch.Tensor(*weight_size))
        nn.init.orthogonal_(self.weight, gain=1./np.sqrt(tensor_size[-1]))
        self.tensor_size = (6, n_capsules, capsule_length)

    def forward(self, tensor):
        batch_size, primary_capsule_length, h, w, n_primary_capsules = tensor.size()
        u = tensor.view(batch_size, -1, 1, n_primary_capsules).matmul(self.weight)
        u = u.view(*(tensor.size()[:4] + self.tensor_size[1:]))
        bias = Variable(torch.zeros(batch_size, primary_capsule_length, h, w, self.tensor_size[1]))
        if tensor.is_cuda:
            bias = bias.cuda()
        for i in range(self.iterations):
            c = F.softmax(bias, 4)
            s = (c.unsqueeze(5)*u).sum(3).sum(2).sum(1)
            sum_squares = (s**2).sum(2).unsqueeze(2)
            v = (sum_squares/(1+sum_squares)) * s / (sum_squares**0.5)
            bias = bias + (u * v.view(batch_size, 1, 1, 1, self.tensor_size[1], self.tensor_size[2])).sum(5)
        return v

# x = torch.rand(3,32,10,10,8)
# test = RoutingCapsule((3,32,10,10,8), 10, 16, 3,)
# test(x).size()
