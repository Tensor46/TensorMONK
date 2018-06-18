""" TensorMONK's :: NeuralArchitectures                                      """

__all__ = ["ConvolutionalVAE", ]


import torch
import torch.nn as nn
import torch.nn.functional as F
from ..NeuralLayers import *
import numpy as np
#==============================================================================#


class ReShape(nn.Module):
    def __init__(self, tensor_size):
        super(ReShape, self).__init__()
        self.tensor_size = tensor_size

    def forward(self, tensor):
        return tensor.view(tensor.size(0), *self.tensor_size[1:])


class ConvolutionalVAE(nn.Module):
    """
        Convolutional Variational Auto Encoder

        Parameters
            tensor_size :: expected size of input tensor
            embedding_layers :: a list of (filter_size, out_channels, strides) in each
                                intermediate layer of the encoder.
                                A flip is used for decoder
            n_latent :: length of latent vecotr Z
            decoder_final_activation :: tanh/sigm

            activation, batch_nm, pre_nm, weight_nm, bias :: refer to core.NeuralLayers.Layers

    """
    def __init__(self, tensor_size=(6, 1, 28, 28), embedding_layers=[(3, 32, 2), (3, 64, 2),], n_latent=128,
                 decoder_final_activation="tanh", pad=True, activation="relu", batch_nm=False, pre_nm=False,
                 groups=1, weight_nm=False, bias=False, *args, **kwargs):
        super(ConvolutionalVAE, self).__init__()

        assert type(tensor_size) in [list, tuple], "ConvolutionalVAE -- tensor_size must be tuple or list"
        assert len(tensor_size) == 4, "ConvolutionalVAE -- len(tensor_size) != 4"

        # encoder with Convolution layers
        encoder = []
        _tensor_size = tensor_size
        for f, c, s in embedding_layers:
            encoder.append(Convolution(_tensor_size, f, c, s, pad,  activation, 0., batch_nm, pre_nm, groups, weight_nm))
            _tensor_size = encoder[-1].tensor_size
        self.encoder = nn.Sequential(*encoder)
        # mu and log_var to synthesize Z
        self.mu = Linear(_tensor_size, n_latent, "", 0., batch_nm, pre_nm, weight_nm, bias)
        self.log_var = Linear(_tensor_size, n_latent, "", 0., batch_nm, pre_nm, weight_nm, bias)

        # decoder - (Linear layer + ReShape) to generate encoder last output shape,
        #           followed by inverse of encoder
        decoder = []
        decoder.append(Linear(self.mu.tensor_size, int(np.prod(_tensor_size[1:])),
                              activation, 0., batch_nm, pre_nm, weight_nm, bias))
        decoder.append(ReShape(_tensor_size))

        decoder_layers = []
        for i,x in enumerate(embedding_layers[::-1]):
            if i+1 == len(embedding_layers):
                decoder_layers += [(x[0], tensor_size[1],x[2], tensor_size) ]
            else:
                decoder_layers += [(x[0], embedding_layers[::-1][i+1][1], x[2], encoder[-(i+2)].tensor_size)]

        for i, (f, c, s, o) in enumerate(decoder_layers):
            decoder.append(ConvolutionTranspose(_tensor_size, f, c, s, pad,  activation, 0., batch_nm, pre_nm, groups, weight_nm))
            decoder[-1].tensor_size = o # adjusting the output tensor size
            _tensor_size = decoder[-1].tensor_size
        # Final normalization
        if decoder_final_activation == "tanh":
            decoder.append(nn.Tanh())
        else:
            decoder.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder)

        self.tensor_size = (6, n_latent)

    def forward(self, tensor):

        encoded = self.encoder(tensor)
        mu, log_var = self.mu(encoded), self.log_var(encoded)

        std = log_var.mul(0.5).exp_()
        if torch.__version__.startswith("0.3"):
            _eps = torch.FloatTensor(std.size()).normal_()
            if tensor.is_cuda:
                _eps = _eps.cuda
        else:
            _eps = torch.FloatTensor(std.size()).normal_().to(tensor.device)
        # mutlivariate latent
        latent = _eps.mul(std).add_(mu)
        kld = torch.mean(mu.pow(2).add_(log_var.exp()).mul_(-1).add_(1).add_(log_var)).mul_(-0.5)
        decoded = self.decoder(latent)
        mse = F.mse_loss(decoded, tensor)

        return encoded, mu, log_var, latent, decoded, kld, mse

# from core.NeuralLayers import *
# tensor_size = (1, 1, 28, 28)
# tensor = torch.rand(*tensor_size)
# test = ConvolutionalVAE(tensor_size)
# test(tensor)[3].shape
# test(tensor)[4].shape
