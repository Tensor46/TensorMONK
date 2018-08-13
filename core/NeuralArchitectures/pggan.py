""" TensorMONK's :: NeuralArchitectures                                      """


import torch
import torch.nn as nn
import torch.nn.functional as F
from ..NeuralLayers import *
from ..NeuralLayers.normalizations import PixelWise
import numpy as np
# ============================================================================ #


class ConvConvT(nn.Module):
    """
        (Convolution/ConvolutionTranspose) + LeakyReLU + pixelwise normalization
        * weight-norm is used instead of equalized Convolution/ConvolutionTranspose
        layer -- "conv"/"trans"
    """
    def __init__(self,
                 layer,
                 tensor_size,
                 filter_size,
                 out_channels,
                 strides       = 1,
                 pad           = True,
                 activation    = "lklu",
                 dropout       = 0.,
                 normalization = "pixelwise",
                 pre_nm        = False,
                 groups        = 1,
                 weight_nm     = False,
                 equalized     = True):

        super(ConvConvT, self).__init__()

        assert layer in ["conv", "trans"], "layer must be conv/trans --> {}".format(layer)
        if layer == "conv": # convolution
            self.network = Convolution(tensor_size, filter_size, out_channels, strides,
                                       pad, activation, dropout, normalization, pre_nm,
                                       groups, weight_nm, equalized, gain=np.sqrt(2)/4)
        else: # convolution transpose
            if tensor_size[2] == 1 or tensor_size[3] == 1: pad = False
            self.network = ConvolutionTranspose(tensor_size, filter_size, out_channels, strides,
                                                pad, activation, dropout, normalization, pre_nm,
                                                groups, weight_nm, equalized, gain=np.sqrt(2)/4)
            if tensor_size[2] > 1 and tensor_size[3] > 1: # safe check
                self.network.tensor_size = (1, out_channels, tensor_size[2]*strides, tensor_size[3]*strides)
        self.activation = activation
        self.tensor_size = self.network.tensor_size

    def forward(self, tensor):
        return self.network(tensor)


# from core.NeuralLayers import *
# tensor_size = (1, 6, 10, 10)
# tensor = torch.rand(*tensor_size)
# test = ConvConvT("conv", tensor_size, 3, 6, 2)
# test(tensor).shape
# ============================================================================ #


class ReShape(nn.Module):
    def __init__(self, tensor_size):
        super(ReShape, self).__init__()
        self.tensor_size = tensor_size

    def forward(self, tensor):
        return tensor.view(tensor.size(0), *self.tensor_size[1:])
# ============================================================================ #


class PGGAN(nn.Module):
    """
        Progressing growth of GANs
        Implemented https://arxiv.org/pdf/1710.10196.pdf
        There are obviously some difference!

        Parameters
            n_embedding :: length of latent vector
            levels :: levels to grow
                    Ex: if l1_size = (4, 4) and levels = 4
                    final size or max_tensor_size = 32x32
            l1_size :: inital size
            l1_iterations :: number of iterations in level1. Also, used to grow
                    number of iterations in consecutive levels
                    Ex: If l1_iterations = 10 and levels = 10
                    l1 will have 10 iterations
                    l2 with transit will have 15
                    l2 without transit will have 20
                    l3 with transit will have 25
                    l3 without transit will have 30
                    ...
            gr :: growth rate final level has gr channels
                    a layer below will have gr*(2**1) channels
                    2 layers below will have gr*(2**2) channels
            pow_gr :: when True the growth of each layer is by a power of 2 else
                    a multiple of layers
                    Ex: levels=4, l1_size=(4, 4) growth_rate=32
                    when pow_gr=True
                        channels in level1 ConvConvT = 32*(2**3) = 256
                        channels in level2 ConvConvT = 32*(2**2) = 128
                        channels in level3 ConvConvT = 32*(2**1) = 64
                        channels in level4 ConvConvT = 32*(2**0) = 32
                    when pow_gr=False (fewer parameters when you have more levels)
                        channels in level1 ConvConvT = 32*4 = 128
                        channels in level2 ConvConvT = 32*3 = 96
                        channels in level3 ConvConvT = 32*2 = 64
                        channels in level4 ConvConvT = 32*1 = 32

        ** Known issue - retains unused parameters
    """

    def __init__(self,
                 n_embedding   = 600,
                 levels        = 4,
                 l1_size       = (4, 4),
                 l1_iterations = 100000,
                 growth_rate   = 32,
                 pow_gr        = True,
                 *args, **kwargs):
        super(PGGAN, self).__init__()

        assert isinstance(n_embedding, int), "n_embedding must be integer -> {}".format(type(n_embedding))
        self.alpha = 0.5
        self.n_embedding = n_embedding
        self.levels = levels
        self.l1_iterations = l1_iterations
        # compute iterations required per level and transition stages using
        # l1_iterations
        iterations_per_level = [l1_iterations] + [l1_iterations//2 * (x+3)
                                                   for x in range((levels-1)*2)]
        self.cumsum_iterations = list(np.cumsum(iterations_per_level))
        self.max_iterations = np.sum(iterations_per_level)
        self.iterations_per_level = iterations_per_level
        # all levels ("current_level", "trasition")
        # for n levels -- you will have n*2-1 stages
        all_levels = [1,] + list(np.arange(2, levels+1).repeat(2))
        self.all_updates = [{"current_level": ls, "transition": i % 2 == 1} for i, ls in
                            enumerate(all_levels)]

        # gather all generator modules
        self.g_modules = nn.ModuleDict()
        tensor_size = (1, n_embedding, 1, 1)
        self.g_list = []
        for i in range(1, levels+1):
            # every level has one ConvolutionTranspose and to-RGB converter
            nc = (growth_rate*(2**(levels-i))) if pow_gr else (growth_rate*(levels-i+1))
            if i == 1:
                self.g_modules.update({"level"+str(i):
                    nn.Sequential(Linear((1, n_embedding), int(nc*np.prod(l1_size)), "lklu", bias=True),
                                  ReShape((1, nc, l1_size[0], l1_size[1])), PixelWise(),
                                  ConvConvT("trans", (1, nc, l1_size[0], l1_size[1]), 3, nc, 1))})
                tensor_size = self.g_modules["level"+str(i)][-1].tensor_size
            else:
                self.g_modules.update({"level"+str(i): ConvConvT("trans", tensor_size, 3, nc, 2)})
                tensor_size = self.g_modules["level"+str(i)].tensor_size
            self.g_modules.update({"level"+str(i)+"_rgb":
                ConvConvT("trans", tensor_size, 3, 3, 1, activation="sigm")})
            self.g_list.append("level"+str(i))
            self.g_list.append("level"+str(i)+"_rgb")

        # gather all discriminator modules
        self.d_modules = nn.ModuleDict()
        tensor_size = (1, 3, l1_size[0]*(2**(levels-1)), l1_size[1]*(2**(levels-1)))
        self.max_tensor_size = tensor_size
        self.d_list = []
        for i in range(levels, 0, -1):
            # every level has one Convolution and from-RGB converter
            nc = (growth_rate*(2**(levels-i))) if pow_gr else (growth_rate*(levels-i+1))
            _tensor_size = (tensor_size[0], 3, tensor_size[2], tensor_size[3])
            self.d_modules.update({"level"+str(i)+"_rgb": ConvConvT("conv", _tensor_size, 3, nc)})
            tensor_size = self.d_modules["level"+str(i)+"_rgb"].tensor_size
            self.d_modules.update({"level"+str(i): ConvConvT("conv", tensor_size, 3,
                                    nc*2 if pow_gr else (growth_rate*(levels-i+1+1)), 1 if i == 1 else 2)})
            # if pow_gr else (growth_rate*(levels-i+1))
            tensor_size = self.d_modules["level"+str(i)].tensor_size
            self.d_list.append("level"+str(i)+"_rgb")
            self.d_list.append("level"+str(i))

        # Average pool and linear layer answer fake or real
        self.d_modules.update({"decide":
            nn.Sequential(nn.AvgPool2d(l1_size, (1, 1)),
                          Linear((1, tensor_size[1], 1, 1), 1, "sigm", .2, bias=False))})
        self.d_list.append("decide")

        # level 1 requirements
        self.current_level = 0
        self.updates(0)
        self.scale = lambda x, y: F.interpolate(x, scale_factor=y)

    def updates(self, iteration, force_update=False):
        """
            Using iteration find level, transition and alpha.
                Update g_base, g_rgb, g_transit, d_base, d_rgb, d_transit &
                alpha when required!
        """
        self.alpha = 0.5
        # find level
        for i, x in enumerate(self.cumsum_iterations):
            if x > iteration:
                break
        level = self.all_updates[i]["current_level"]
        transition = self.all_updates[i]["transition"]
        if transition:
            ittr = self.cumsum_iterations[i] - self.cumsum_iterations[i-1]
            self.alpha = float(iteration - self.cumsum_iterations[i-1]) / ittr

        if level != self.current_level or transition != self.transition or force_update:
            if level == 1: transition = False
            self.transition = transition
            if level > self.levels: level = self.levels

            if level == 1:
                self.g_base = ["level1"]
                self.g_rgb = ["level1_rgb"]
                self.g_transit = None

                self.d_rgb = ["level1_rgb"]
                self.d_base = ["level1", "decide"]
                self.d_transit = None
            else:
                modules = ["level"+str(i) for i in range(1, level+1)] + ["level"+str(level)+"_rgb"]
                self.g_base = modules[:-2]
                self.g_rgb = modules[-2:]
                self.g_transit = ["level"+str(level-1)+"_rgb"] if transition else None

                modules = ["level"+str(level)+"_rgb"] + ["level"+str(i) for i in range(level, 0, -1)]\
                          + ["decide"]
                self.d_rgb = modules[:2]
                self.d_base = modules[2:]
                self.d_transit = ["level"+str(level-1)+"_rgb"] if transition else None

            self.tensor_size = self.g_modules[self.g_rgb[-1]].tensor_size
            self.current_level = level

    def forward(self, tensor):
        if tensor.dim() == 2: # generate
            tensor = tensor.view(tensor.size(0), tensor.size(1), 1, 1)
            if self.g_transit is None:
                for i in self.g_base + self.g_rgb:
                    tensor = self.g_modules[i](tensor)
                return tensor

            for i in self.g_base:
                tensor = self.g_modules[i](tensor)
            last_rgb = self.scale(self.g_modules[self.g_transit[0]](tensor), 2)
            this_rgb = tensor
            for i in self.g_rgb:
                this_rgb = self.g_modules[i](this_rgb)

            return this_rgb.mul(self.alpha) + last_rgb.mul(1 - self.alpha)
        else: # discriminate
            tensor = F.interpolate(tensor, size=self.tensor_size[2:])
            if self.d_transit is None:
                for i in self.d_rgb + self.d_base:
                    tensor = self.d_modules[i](tensor)
                return tensor
            last_2base = self.d_modules[self.d_transit[0]](self.scale(tensor, 0.5))
            for i in self.d_rgb:
                tensor = self.d_modules[i](tensor)
            tensor = tensor.mul(self.alpha) + last_2base.mul(1 - self.alpha)
            for i in self.d_base:
                tensor = self.d_modules[i](tensor)
            return tensor


# from core.NeuralLayers import *
# from core.NeuralLayers.normalizations import PixelWise
# test = PGGAN(growth_rate=128, pow_gr = True, l1_iterations = 100)
# test.cumsum_iterations
# test.updates(99)
# test.alpha
# test(torch.rand(6, 600)).shape
# test(torch.rand(*test.tensor_size))
# test.updates(101)
# test.alpha
# test(torch.rand(6, 600)).shape
# test(torch.rand(*test.tensor_size))
# test.updates(251)
# test.alpha
# test(torch.rand(6, 600)).shape
# test(torch.rand(*test.tensor_size))
# test.updates(451)
# test.alpha
# test(torch.rand(6, 600)).shape
# test(torch.rand(*test.tensor_size))
# test.updates(701)
# test.alpha
# test(torch.rand(6, 600)).shape
# test(torch.rand(*test.tensor_size))
# test.updates(1001)
# test.alpha
# test(torch.rand(6, 600)).shape
# test(torch.rand(*test.tensor_size))
# test.updates(1351)
# test.alpha
# test(torch.rand(6, 600)).shape
# test(torch.rand(*test.tensor_size))
