""" TensorMONK :: architectures """

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers import Convolution, Linear, SelfAttention
from ..normalizations.categoricalbatch import CategoricalBNorm
import numpy as np


class GBlock(nn.Module):
    def __init__(self, tensor_size, out_channels, n_labels, n_latent=None):
        super(GBlock, self).__init__()
        from torch.nn.utils import spectral_norm
        # residual connection
        n, c, h, w = tensor_size
        self.residual = Convolution((n, c, h*2, w*2), 1, out_channels,
                                    strides=1, pad=True, activation=None,
                                    normalization=None)
        self.residual.Convolution = spectral_norm(self.residual.Convolution)

        # categorical batch norm + relu + upscale + convolution
        self.bnrm1 = CategoricalBNorm(tensor_size, n_labels, n_latent)
        self.conv1 = Convolution((n, c, h*2, w*2), 3, out_channels,
                                 strides=1, pad=True, activation=None)
        self.conv1.Convolution = spectral_norm(self.conv1.Convolution)
        # categorical batch norm + relu + convolution
        self.conv2 = Convolution((n, out_channels, h*2, w*2), 3, out_channels,
                                 strides=1, pad=True, activation="relu",
                                 normalization="cbatch", pre_nm=True,
                                 n_labels=n_labels, n_latent=n_latent)
        self.conv2.Convolution = spectral_norm(self.conv2.Convolution)

        self.tensor_size = (None, out_channels, h*2, w*2)
        self.name = "gblock"

    def forward(self, tensor, labels):
        residue = self.residual(
            F.interpolate(tensor, size=self.tensor_size[2:]))

        o = F.interpolate(F.relu(self.bnrm1(tensor, labels)),
                          size=self.tensor_size[2:])
        o = self.conv2(self.conv1(o), labels)
        return residue + o


class DBlock(nn.Module):
    def __init__(self, tensor_size, out_channels):
        super(DBlock, self).__init__()
        from torch.nn.utils import spectral_norm
        n, c, h, w = tensor_size
        # avg_pool outsize
        t_size = F.avg_pool2d(torch.randn(1, out_channels, h, w), 2).shape

        # residual connection
        if c != out_channels:
            self.residual = Convolution((n, c, t_size[2], t_size[3]), 1,
                                        out_channels,
                                        activation=None, normalization=None)
            self.residual.Convolution = spectral_norm(
                self.residual.Convolution)

        # convolution + relu + convolution
        self.conv1 = Convolution(tensor_size, 3, out_channels,
                                 strides=1, pad=True,
                                 activation="relu" if c > 3 else None,
                                 pre_nm=True)
        self.conv1.Convolution = spectral_norm(self.conv1.Convolution)
        self.conv2 = Convolution(self.conv1.tensor_size, 3, out_channels,
                                 strides=1, pad=True, activation="relu",
                                 pre_nm=True)
        self.conv2.Convolution = spectral_norm(self.conv2.Convolution)

        self.tensor_size = (None, out_channels, t_size[2], t_size[3])

    def forward(self, tensor):
        residue = F.avg_pool2d(tensor, 2)
        if hasattr(self, "residual"):
            residue = self.residual(residue)
        o = F.avg_pool2d(self.conv2(self.conv1(tensor)), 2)
        return residue + o


class SAGAN(nn.Module):
    r"""Self-Attention Generative Adversarial Networks -
    https://arxiv.org/pdf/1805.08318.pdf
    There are obviously some difference!

    Args:
        is_generator (bool): When True, SAGAN is generatoe else discriminator
        n_latent (int): length of latent vector
        n_labels (int): number of labels
        levels (int): levels to grow, default = 4
            Ex: if l1_size = (4, 4) and levels = 4, final size or
            max_tensor_size = 32x32
        l1_size (tuple): inital size, default = (4, 4)
        growth_rate (int): final level has growth_rate channels, default=32
            a layer below will have growth_rate*(2**1) channels
            2 layers below will have growth_rate*(2**2) channels
        pow_gr (bool): when True the growth of each layer is by a power of 2
            else a multiple of layers, default=True
            Ex: levels=4, l1_size=(4, 4) gr=32
                when pow_gr=True
                    channels in level1 = 32*(2**3) = 256
                    channels in level2 = 32*(2**2) = 128
                    channels in level3 = 32*(2**1) = 64
                    channels in level4 = 32*(2**0) = 32
                when pow_gr=False (fewer parameters when you have more levels)
                    channels in level1 = 32*4 = 128
                    channels in level2 = 32*3 = 96
                    channels in level3 = 32*2 = 64
                    channels in level4 = 32*1 = 32
        rgb (bool): True/False, default=True
        max_channels (int): maximum convolutional channels any layer can have
            regardless the growth_rate and pwr_gr
        initial_linear (bool): When True, generator initial layer is a Linear
            layer else it is a ConvTranspose2d


    Ex:
        generator = SAGAN(is_generator=True, n_latent=128, n_labels=10,
                          levels=4, l1_size=(4, 4), growth_rate=64)
        discriminator = SAGAN(is_generator=False, n_latent=128, n_labels=10,
                              levels=4, l1_size=(4, 4), growth_rate=64)

    """
    def __init__(self,
                 is_generator: bool,
                 n_latent: int,
                 n_labels: int,
                 levels: int = 4,
                 l1_size: tuple = (4, 4),
                 growth_rate: int = 64,
                 pow_gr: bool = True,
                 rgb: bool = True,
                 max_channels: int = 256,
                 initial_linear: bool = True,
                 **kwargs):
        super(SAGAN, self).__init__()

        if not isinstance(is_generator, bool):
            raise TypeError("SAGAN: is_generator must be bool")
        if not isinstance(n_latent, int):
            raise TypeError("SAGAN: n_latent must be int")
        if not isinstance(n_labels, int):
            raise TypeError("SAGAN: n_labels must be int")
        if not isinstance(levels, int):
            raise TypeError("SAGAN: levels must be int")
        if not isinstance(l1_size, tuple):
            raise TypeError("SAGAN: l1_size must be tuple")
        if not isinstance(growth_rate, int):
            raise TypeError("SAGAN: growth_rate must be int")
        if not isinstance(pow_gr, bool):
            raise TypeError("SAGAN: pow_gr must be bool")
        if not isinstance(rgb, bool):
            raise TypeError("SAGAN: rgb must be bool")
        if not isinstance(max_channels, int):
            raise TypeError("SAGAN: max_channels must be int")
        if not isinstance(initial_linear, bool):
            raise TypeError("SAGAN: initial_linear must be bool")

        c = 3 if rgb else 1
        self.c = c
        self.n_latent = n_latent
        self.n_labels = n_labels
        self.levels = levels
        self.l1_size = l1_size
        self.growth_rate = growth_rate
        self.pow_gr = pow_gr
        self.max_channels = max_channels
        self.initial_linear = initial_linear

        self.is_generator = is_generator
        if self.is_generator:
            # creating generator modules
            self.g_modules = nn.ModuleList(self.build_generator())
            self.tensor_size = self.g_modules[-1].tensor_size
        else:
            # creating discriminator modules
            self.d_modules = nn.ModuleList(self.build_discriminator())
            self.tensor_size = 1,

    def forward(self, tensor: torch.Tensor, labels: torch.Tensor):
        if self.is_generator:
            tensor = tensor.view(tensor.size(0), -1, 1, 1)
            for module in self.g_modules:
                if hasattr(module, "name"):
                    if module.name == "gblock":
                        tensor = module(tensor, labels)
                        continue
                tensor = module(tensor)
            return torch.tanh(tensor)

        for _, module in zip(range(len(self.d_modules) - 2), self.d_modules):
            tensor = module(tensor)
        tensor = F.avg_pool2d(tensor, tensor.shape[2:])
        tensor = tensor.view(tensor.size(0), -1)
        return torch.sigmoid((self.d_modules[-1](labels) * tensor).sum(1) +
                             self.d_modules[-2](tensor).view(-1))

    def compute_sizes(self):
        t_sizes = []
        for i in range(self.levels):
            channels = self.growth_rate*2**i if self.pow_gr else \
                self.growth_rate*(i+1)
            channels = min(channels, self.max_channels)
            mul = 2**(self.levels-i-1)
            t_sizes += [(1, channels, self.l1_size[0]*mul,
                         self.l1_size[1]*mul)]
        return [(1, self.c, t_sizes[0][2]*2, t_sizes[0][3]*2)] + t_sizes

    def build_generator(self):
        from torch.nn.utils import spectral_norm
        t_sizes = self.compute_sizes()
        modules = []
        if self.initial_linear:
            # initial layer is linear layer
            modules.append(Linear(tensor_size=self.n_latent,
                                  out_features=int(np.prod(t_sizes[-1])),
                                  out_shape=t_sizes[-1][1:]))
            modules[-1] = spectral_norm(modules[-1])
        else:
            # initial layer is convolutional transpose
            modules.append(Convolution(tensor_size=(1, self.n_latent, 1, 1),
                                       filter_size=self.l1_size,
                                       out_channels=t_sizes[-1][1],
                                       strides=1, pad=False, activation=None,
                                       transpose=True))
            modules[-1].Convolution = spectral_norm(
                modules[-1].Convolution)

        for i in range(1, self.levels+1):
            modules.append(GBlock(modules[-1].tensor_size, t_sizes[-i][1],
                                  self.n_labels))
            if i == (self.levels//2 + 1):
                modules.append(SelfAttention(modules[-1].tensor_size,
                                             shrink=8, scale_factor=1.))
        modules.append(Convolution(modules[-1].tensor_size, 3, self.c,
                                   strides=1, pad=True, activation="relu",
                                   normalization="batch", pre_nm=True))
        return modules

    def build_discriminator(self):
        from torch.nn.utils import spectral_norm
        t_sizes = self.compute_sizes()
        modules = []
        for i in range(self.levels):
            modules.append(DBlock(modules[-1].tensor_size if i > 0 else
                                  t_sizes[i], t_sizes[i+1][1]))
        modules.append(spectral_norm(Linear(modules[-1].tensor_size[1], 1)))
        modules.append(spectral_norm(nn.Embedding(num_embeddings=self.n_labels,
                       embedding_dim=modules[-2].tensor_size[1])))
        return modules

    def noisy_latent(self, n):
        r""" generates a random tensor for generator.
        """
        return torch.randn(n, self.n_latent, requires_grad=True)

    def test_image(self, n):
        r""" generates a random tensor for discriminator
        """
        t_sizes = self.compute_sizes()
        return torch.randn(n, *t_sizes[0][1:], requires_grad=True)


# from tensormonk.layers import Convolution, Linear, SelfAttention
# from tensormonk.normalizations.categoricalbatch import CategoricalBNorm
# test = SAGAN(False, 1024, 10, 6, (3, 4), max_channels=1024,
#              initial_linear=True)
# test(test.test_image(3), torch.Tensor([0, 1, 9]).long())
# test = SAGAN(True, 1024, 10, 6, (3, 4), max_channels=1024,
#              initial_linear=True)
# test(test.noisy_latent(3), torch.Tensor([0, 1, 9]).long()).shape
