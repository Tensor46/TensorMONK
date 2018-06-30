""" TensorMONK's :: NeuralEssentials                                         """

import torch
import torch.nn as nn
#==============================================================================#


class CudaModel(nn.Module):
    """ Works on both CPU & GPU """
    def __init__(self, is_cuda, gpus, net, net_kwargs):
        super(CudaModel, self).__init__()
        self.gpus = gpus
        self.is_cuda = is_cuda
        self.NET46 = net( **net_kwargs )
        self.tensor_size = self.NET46.tensor_size

    def forward(self, inputs):
        if type(inputs) in [list,tuple]:
            if self.is_cuda:
                inputs = [x.cuda() for x in inputs]
            return self.NET46(*inputs)
        else:
            if self.is_cuda:
                inputs = inputs.cuda()
            if self.is_cuda and self.gpus>1:
                return nn.parallel.data_parallel(self.NET46, inputs, range(self.gpus))
            else:
                return self.NET46(inputs)
