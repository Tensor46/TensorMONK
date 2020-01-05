""" TensorMONK's :: architectures :: ESRGAN """

__all__ = ["Generator", "Discriminator", "VGG19"]

import torch
import torch.nn as nn
import torchvision
from ..layers import Convolution


class DenseBlock(nn.Module):
    r"""From DenseNet - https://arxiv.org/pdf/1608.06993.pdf."""
    def __init__(self, tensor_size: tuple, filter_size: int = 3,
                 activation: str = "lklu", normalization: str = None,
                 n_blocks: int = 5, beta: float = 0.2, **kwargs):

        super(DenseBlock, self).__init__()
        n, c, h, w = tensor_size
        cnns = []
        for i in range(n_blocks):
            cnns.append(Convolution(
                (1, c*(i+1), h, w), filter_size, out_channels=c, strides=1,
                activation=None if (i + 1) == n_blocks else activation,
                normalization=normalization, lklu_negslope=0.1))
        self.cnns = nn.ModuleList(cnns)
        self.tensor_size = tensor_size

        # As defined in https://arxiv.org/pdf/1602.07261.pdf
        self.beta = beta

    def forward(self, tensor: torch.Tensor):
        r"""Residual dense block with scaling."""
        x, o = None, None
        for i, cnn in enumerate(self.cnns):
            x = tensor if i == 0 else torch.cat((x, o), 1)
            o = cnn(x)
        return tensor + (o * self.beta)


class RRDB(nn.Module):
    r"""Residual-in-Residual Dense Block."""
    def __init__(self, tensor_size: tuple, filter_size: int = 3,
                 activation: str = "lklu", normalization: str = None,
                 n_dense: int = 3, n_blocks: int = 5, beta: float = 0.2,
                 **kwargs):

        super(RRDB, self).__init__()
        cnns = []
        for i in range(n_dense):
            cnns.append(DenseBlock(
                tensor_size, filter_size, activation, normalization,
                n_blocks, beta, **kwargs))
        self.cnn = nn.Sequential(*cnns)
        self.tensor_size = tensor_size

        # As defined in https://arxiv.org/pdf/1602.07261.pdf
        self.beta = beta

    def forward(self, tensor: torch.Tensor):
        r"""Residual-in-Residual Dense Block with scaling (beta)."""
        return tensor + self.cnn(tensor) * self.beta


class Generator(nn.Module):
    r"""ESRGAN generator network using Residual-in-Residual Dense Blocks.

    Paper: ESRGAN
    URL:   https://arxiv.org/pdf/1809.00219.pdf

    Args:
        tensor_size (tuple, required): Shape of tensor in
            (None/any integer >0, channels, height, width).

        n_filters (int): The number of filters used through out the network,
            however, DenseBlock will have multiples of n_filters.
            default = 64

        n_rrdb (int): The number of Residual-in-Residual Dense Block (RRDB).
            default = 16

        n_dense (int): The number of dense blocks in RRDB.
            default = 3

        n_blocks (int): The number of convolutions in dense blocks.
            default = 5

        n_upscale (int): Number of upscale done on input shape using
            pixel-shuffle.
            default = 2

        beta (float): The scale factor of output before adding to any residue.
            default = 0.2
    """

    def __init__(self,
                 tensor_size: tuple = (1, 3, 32, 32),
                 n_filters: int = 64,
                 n_rrdb: int = 16,
                 n_dense: int = 3,
                 n_blocks: int = 5,
                 n_upscale: int = 2,
                 beta: float = 0.2,
                 **kwargs):

        super(Generator, self).__init__()
        self.initial = Convolution(
            tensor_size, 3, n_filters, 1, activation=None)
        modules = []
        t_size = self.initial.tensor_size
        for _ in range(n_rrdb):
            modules.append(RRDB(t_size, 3, "lklu", n_dense=n_dense,
                                n_blocks=n_blocks, beta=beta))
        modules.append(Convolution(t_size, 3, n_filters, 1, activation=None))
        self.rrdbs = nn.Sequential(*modules)
        modules = []
        for _ in range(n_upscale):
            modules.append(
                Convolution(t_size, 3, n_filters * 4, 1, activation="lklu"))
            modules.append(nn.PixelShuffle(upscale_factor=2))
            t_size = (t_size[0], t_size[1], t_size[2]*2, t_size[3]*2)
        modules.append(Convolution(t_size, 3, n_filters, 1, activation="lklu"))
        modules.append(Convolution(t_size, 3, tensor_size[1], activation=None))
        self.upscale = nn.Sequential(*modules)
        self.tensor_size = tensor_size
        self.initialize()

    def forward(self, tensor: torch.Tensor):
        r"""Expects normalized tensor (mean = 0.5 and std = 0.25)."""
        o = self.initial(tensor)
        o = o + self.rrdbs(o)
        o = self.upscale(o)
        return o

    def enhance(self, tensor: torch.Tensor):
        with torch.no_grad():
            return self(tensor).mul_(0.25).add_(0.5).clamp_(0, 1)

    def initialize(self):
        r"""As defined in https://arxiv.org/pdf/1809.00219.pdf."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data.mul_(0.1)


class Discriminator(nn.Module):
    r"""ESRGAN discriminator network.

    Paper: ESRGAN
    URL:   https://arxiv.org/pdf/1809.00219.pdf

    Args:
        tensor_size (tuple, required): Shape of tensor in
            (None/any integer >0, channels, height, width).
    """

    def __init__(self, tensor_size: tuple = (1, 3, 128, 128), **kwargs):
        super(Discriminator, self).__init__()
        self.t_size = tensor_size
        self.tensor_size = None, 1

        modules = []
        t_size = self.t_size
        for oc in (64, 128, 256, 512):
            modules.append(Convolution(
                t_size, 3, oc, 1, normalization=None if oc == 64 else "batch",
                activation="lklu", lklu_negslope=0.2))
            t_size = modules[-1].tensor_size
            modules.append(Convolution(
                t_size, 3, oc, 2, normalization="batch",
                activation="lklu", lklu_negslope=0.2))
            t_size = modules[-1].tensor_size

        self.discriminator = nn.Sequential(*modules)

    def forward(self, tensor: torch.Tensor):
        r"""Expects normalized tensor (mean = 0.5 and std = 0.25)."""
        return self.discriminator(tensor)

    def initialize(self):
        r"""As defined in https://arxiv.org/pdf/1809.00219.pdf."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data.mul_(0.1)


class VGG19(nn.Module):
    r"""Pretrained VGG19 model from torchvision."""
    def __init__(self, **kwargs):
        super(VGG19, self).__init__()
        self.vgg19 = torchvision.models.vgg19(pretrained=True).features[:35]

    def forward(self, tensor: torch.Tensor):
        r"""Expects normalized tensor (mean = 0.5 and std = 0.25)."""
        return self.vgg19(tensor)


class ESRGAN:
    Generator = Generator
    Discriminator = Discriminator
    VGG19 = VGG19
