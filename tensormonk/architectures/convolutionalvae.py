""" TensorMONK :: architectures """

__all__ = ["ConvolutionalVAE", ]


import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers import Convolution, Linear
import numpy as np


class ConvolutionalVAE(nn.Module):
    r""" Example Convolutional Variational Auto Encoder

    Args:
        tensor_size: shape of tensor in BCHW
            (None/any integer >0, channels, height, width)
        embedding_layers: a list/tuple of (filter_size, out_channels, strides)
            in each intermediate layer of the encoder. A flip is used for
            decoder.
        n_latent: length of latent vecotr Z
        decoder_final_activation: tanh/sigm
        activation, dropout, normalization, pre_nm, weight_nm, equalized, bias:
            refer to tensormonk.layers.Convolution

    Return:
        encoded, mu, log_var, latent, decoded, kld, mse
    """
    def __init__(self,
                 tensor_size: tuple = (6, 1, 28, 28),
                 embedding_layers: list = [(3, 32, 2), (3, 64, 2)],
                 n_latent: int = 128,
                 decoder_final_activation: str = "tanh",
                 pad: bool = True,
                 activation: str = "relu",
                 dropout: float = 0.,
                 normalization: str = None,
                 pre_nm: bool = False,
                 groups: int = 1,
                 weight_nm: bool = False,
                 equalized: bool = False,
                 bias: bool = False,
                 *args, **kwargs):
        super(ConvolutionalVAE, self).__init__()

        assert type(tensor_size) in [list, tuple],\
            "ConvolutionalVAE -- tensor_size must be tuple or list"
        assert len(tensor_size) == 4,\
            "ConvolutionalVAE -- len(tensor_size) != 4"

        kwargs["pad"] = pad
        kwargs["activation"] = activation
        kwargs["dropout"] = dropout
        kwargs["normalization"] = normalization
        kwargs["pre_nm"] = pre_nm
        kwargs["groups"] = groups
        kwargs["weight_nm"] = weight_nm
        kwargs["equalized"] = equalized
        # encoder with Convolution layers
        encoder = []
        t_size = tensor_size
        for f, c, s in embedding_layers:
            encoder.append(Convolution(t_size, f, c, s, **kwargs))
            t_size = encoder[-1].tensor_size
        self.encoder = nn.Sequential(*encoder)

        # mu and log_var to synthesize Z
        self.mu = Linear(t_size, n_latent, "", dropout, bias=bias)
        self.log_var = Linear(t_size, n_latent, "", dropout, bias=bias)

        # decoder - (Linear layer + ReShape) to generate encoder last output
        # shape, followed by inverse of encoder
        decoder = []
        decoder.append(Linear(self.mu.tensor_size,
                              int(np.prod(t_size[1:])), activation, dropout,
                              bias=bias, out_shape=t_size[1:]))

        decoder_layers = []
        for i, x in enumerate(embedding_layers[::-1]):
            if i+1 == len(embedding_layers):
                decoder_layers += [(x[0], tensor_size[1], x[2], tensor_size)]
            else:
                decoder_layers += [(x[0], embedding_layers[::-1][i+1][1], x[2],
                                   encoder[-(i+2)].tensor_size)]

        for i, (f, c, s, o) in enumerate(decoder_layers):
            if i == len(decoder_layers)-1:
                kwargs["activation"] = None
            decoder.append(Convolution(t_size, f, c, s, transpose=True,
                                       maintain_out_size=False if i == 0 and
                                       tensor_size[2] == 28 else True,
                                       **kwargs))
            t_size = decoder[-1].tensor_size
        self.decoder = nn.Sequential(*decoder)

        # Final normalization
        self.activation = decoder_final_activation

        self.tensor_size = (6, n_latent)

    def forward(self, tensor, noisy_tensor=None):

        encoded = self.encoder(tensor if noisy_tensor is None else
                               noisy_tensor)
        mu, log_var = self.mu(encoded), self.log_var(encoded)

        std = log_var.mul(0.5).exp_()
        _eps = torch.FloatTensor(std.size()).normal_().to(tensor.device)

        # mutlivariate latent
        latent = _eps.mul(std).add_(mu)
        kld = torch.mean(1 + log_var - (mu.pow(2) + log_var.exp())).mul(-0.5)
        decoded = self.decoder(latent)
        decoded = torch.tanh(decoded) if self.activation == "tanh" else \
            torch.sigmoid(decoded)
        mse = F.mse_loss(decoded, tensor)
        return encoded, mu, log_var, latent, decoded, kld, mse


# from tensormonk.layers import Convolution, Linear
# tensor_size = (1, 1, 28, 28)
# tensor = torch.rand(*tensor_size)
# test = ConvolutionalVAE(tensor_size)
# test(tensor)[3].shape
# test(tensor)[4].shape
# test(tensor)[5]
