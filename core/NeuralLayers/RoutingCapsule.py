""" TensorMONK's :: NeuralLayers :: RoutingCapsule                           """

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
#==============================================================================#


class RoutingCapsule(nn.Module):
    """ https://arxiv.org/pdf/1710.09829.pdf """
    def __init__(self, tensor_size, n_capsules=10, capsule_length=32, iterations=3, *args, **kwargs):
        super(RoutingCapsule, self).__init__()
        import numpy as np
        self.iterations = iterations
        # Ex from paper
        #   For tensor_size=(1,32,6,6,8), n_capsules=10 and capsule_length=16
        #   weight_size = (tensor_size[1]*tensor_size[2]*tensor_size[3], \
        #                  tensor_size[4], n_capsules*capsule_length)
        #               = (32*6*6, 8 , 10*16)
        weight_size = (int(np.prod(tensor_size[1:-1])), tensor_size[-1], n_capsules*capsule_length)
        self.weight = nn.Parameter(torch.Tensor(*weight_size))
        nn.init.xavier_normal_(self.weight, gain=0.01)
        # nn.init.orthogonal_(self.weight, gain=1./np.sqrt(tensor_size[-1]))
        self.tensor_size = (6, n_capsules, capsule_length)

    def forward(self, tensor):
        batch_size, primary_capsule_length, h, w, n_primary_capsules = tensor.size()
        # Initial squash
        tensor = tensor.view(batch_size, -1, n_primary_capsules)
        sum_squares = (tensor.view(batch_size, -1, n_primary_capsules)**2).sum(2).unsqueeze(2)
        tensor = (sum_squares/(1+sum_squares)) * tensor / (sum_squares**0.5)

        # from the given example:
        #   tensor is of size _ x 32 x 6 x 6 x 8
        #   after matrix mulitplication the size of u is _ x 32 x 6 x 6 x 10 x 16
        #   essentially, each of the pixel from 8 primary capsules is project to
        #   a dimension of n_capsules x capsule_length
        u = tensor.view(batch_size, -1, 1, n_primary_capsules).matmul(self.weight)
        u = u.view(*((batch_size, primary_capsule_length, h, w) + self.tensor_size[1:]))

        bias = Variable(torch.zeros(batch_size, primary_capsule_length, h, w, self.tensor_size[1]))
        if tensor.is_cuda:
            bias = bias.to(tensor.device) if torch.__version__.startswith("0.4") else bias.cuda()

        # routing
        for i in range(self.iterations):
            # softmax
            #   initial softmax gives equal probabilities (since bias is
            #   initialized with zeros), eventually, bias updates will change
            #   the probabilities
            c = F.softmax(bias, 4) # size = _ x 32 x 6 x 6 x 10
            # could be done with a single sum after reorganizing the tensor's,
            #   however, retaining dimensions can explain better
            # s size without sum's = _ x 32 x 6 x 6 x 10 x 16
            # s size = _ x 10 x 16
            s = (c.unsqueeze(5)*u.detach()).sum(3).sum(2).sum(1)
            # squash -- v size = _ x 10 x 16
            sum_squares = (s**2).sum(2).unsqueeze(2)
            v = (sum_squares/(1+sum_squares)) * s / (sum_squares**0.5)
            # bias update -- size = _ x 32 x 6 x 6 x 10
            if i < self.iterations-1:
                bias = bias + (u * v.view(batch_size, 1, 1, 1, self.tensor_size[1], self.tensor_size[2])).sum(5)

        return v


# x = torch.rand(3,32,10,10,8)
# test = RoutingCapsule((3,32,10,10,8), 10, 16, 3,)
# test(x).size()
