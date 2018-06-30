""" TensorMONK's :: NeuralLayers :: Linear                                   """

__all__ = ["ConvolutionalAE", ]


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as neuralOptimizer
from .Convolution import Convolution
from .ConvolutionTranspose import ConvolutionTranspose
import numpy as np
#==============================================================================#


class ConvolutionalAE(nn.Module):
    """ Auto-encoder for StackedConvolutionalAE """
    def __init__(self, tensor_size, filter_size, out_channels, strides=(1, 1),
                 pad=True, activation="relu", dropout=0., batch_nm=False,
                 pre_nm=False, groups=1, weight_norm=False, learningRate=0.1,
                 block=Convolution, *args, **kwargs):
        super(ConvolutionalAE, self).__init__()

        self.encoder = block(tensor_size, filter_size, out_channels, strides, pad, activation, dropout,
                                   batch_nm, pre_nm, groups, weight_norm)
        self.decoder = ConvolutionTranspose(self.encoder.tensor_size, filter_size, tensor_size[1], strides, pad,
                                            activation, 0., batch_nm, pre_nm, groups, weight_norm)
        self.decoder.tensor_size = tensor_size

        self.Optimizer = torch.optim.SGD(self.parameters(), lr=learningRate)
        self.tensor_size = self.encoder.tensor_size

    def forward(self, tensor):
        features = self.encoder(tensor.detach())

        if self.training:
            self.Optimizer.zero_grad()
            loss = F.mse_loss(self.decoder(features), tensor.detach())
            loss.backward()
            self.Optimizer.step()

        return features.detach()

# from core.NeuralLayers import *
# tensor_size = (1, 3, 10, 10)
# tensor = torch.rand(*tensor_size)
# test = ConvolutionalAE(tensor_size, (3,3), 16, (2,2), True, "relu", 0.1, True, False)
# test.train()
# test(tensor).size()
