""" TensorMONK :: layers :: ConvolutionalSAE """

__all__ = ["ConvolutionalSAE", ]


import torch
import torch.nn as nn
import torch.nn.functional as F
from .convolution import Convolution


class ConvolutionalSAE(nn.Module):
    r"""Base block for Stacked Convolutional Auto-encoder """
    def __init__(self, tensor_size, filter_size, out_channels, strides=(1, 1),
                 pad=True, activation="relu", dropout=0., normalization=None,
                 pre_nm=False, groups=1, weight_norm=False, equalized=False,
                 shift=False, bias=False, dropblock=True, learningRate=0.1,
                 **kwargs):

        super(ConvolutionalSAE, self).__init__()
        self.encoder = Convolution(tensor_size, filter_size, out_channels,
                                   strides, pad, activation, dropout,
                                   normalization, pre_nm, groups, weight_norm,
                                   equalized, shift, bias, dropblock, **kwargs)
        self.decoder = Convolution(self.encoder.tensor_size, filter_size,
                                   out_channels, strides, pad, activation,
                                   dropout, normalization, pre_nm, groups,
                                   weight_norm, equalized, shift, bias,
                                   dropblock, transpose=True,
                                   maintain_out_size=True, **kwargs)
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
        return features


# from tensormonk.layers import Convolution
# tensor_size = (1, 3, 10, 10)
# tensor = torch.rand(*tensor_size)
# test = ConvolutionalAE(tensor_size, (3, 3), 16, (2, 2), True,
#                        "relu", 0.1, True, False)
# test.train()
# test(tensor).size()
