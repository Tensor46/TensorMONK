""" TensorMONK's :: NeuralArchitectures                                     """

__all__ = ["LinearVAE", ]


import torch
import torch.nn as nn
import torch.nn.functional as F
from ..NeuralLayers import Linear
import numpy as np


class LinearVAE(nn.Module):
    """
        Variational Auto Encoder
        Implemented https://arxiv.org/pdf/1312.6114v10.pdf

        Parameters
            tensor_size :: expected size of input tensor
            embedding_layers :: a list of neurons in each intermediate layer of
                                the encoder. A flip is used for decoder
            n_latent :: length of latent vecotr Z
            decoder_final_activation :: tanh/sigm

            activation, normalization, pre_nm, weight_nm, equalized, bias ::
                refer to core.NeuralLayers

    """
    def __init__(self,
                 tensor_size=(6, 784),
                 embedding_layers=[1024, 512],
                 n_latent=128,
                 decoder_final_activation="tanh",
                 activation="relu",
                 bias=False,
                 *args, **kwargs):
        super(LinearVAE, self).__init__()

        assert type(tensor_size) in [list, tuple], \
            "LinearVAE -- tensor_size must be tuple or list"
        if len(tensor_size) > 2:  # In case, tensor_size is in BCHW
            tensor_size = (tensor_size[0], int(np.prod(tensor_size[1:])))
        decoder_final_activation = decoder_final_activation.lower()
        assert decoder_final_activation in ("tanh", "sigm"), \
            "LinearVAE -- decoder_final_activation must be sigm/tanh"

        kwargs["dropout"] = 0.
        kwargs["bias"] = bias

        # encoder with Linear layers
        encoder = []
        _tensor_size = tensor_size
        for x in embedding_layers:
            encoder.append(Linear(_tensor_size, x, activation, **kwargs))
            _tensor_size = encoder[-1].tensor_size

        # One more linear layer to get to n_latent length vector
        encoder.append(Linear(_tensor_size, n_latent, activation, **kwargs))
        _tensor_size = encoder[-1].tensor_size
        self.encoder = nn.Sequential(*encoder)
        # mu and log_var to synthesize Z
        self.mu = Linear(_tensor_size, n_latent, "", **kwargs)
        self.log_var = Linear(_tensor_size, n_latent, "", **kwargs)
        # decoder - inverse of encoder
        decoder = []
        for x in embedding_layers[::-1]:
            decoder.append(Linear(_tensor_size, x, activation, **kwargs))
            _tensor_size = decoder[-1].tensor_size
        decoder.append(Linear(_tensor_size, tensor_size[1], "", **kwargs))
        self.activation = decoder_final_activation
        self.decoder = nn.Sequential(*decoder)

        self.tensor_size = (6, n_latent)

    def forward(self, tensor):
        if tensor.dim() != 2:
            tensor = tensor.view(tensor.size(0), -1)

        encoded = self.encoder(tensor)
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

# from core.NeuralLayers import Linear
# tensor_size = (1, 1, 28, 28)
# tensor = torch.rand(*tensor_size)
# test = LinearVAE(tensor_size, [1024, 512], 64)
# test(tensor)[0].shape
# test(tensor)[-3].shape
