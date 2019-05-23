""" TensorMONK :: architectures """

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers import Convolution, Linear
from ..normalizations.pixelwise import PixelWise
import numpy as np
import math


class PGGAN(nn.Module):
    r"""Progressing growth of GANs - https://arxiv.org/pdf/1710.10196.pdf
    There are obviously some difference!

    Args:
        levels (int): levels to grow, default = 4
            Ex: if l1_size = (4, 4) and levels = 4, final size or
            max_tensor_size = 32x32
        l1_size (tuple): inital size, default = (4, 4)
        l1_ittr (tuple): number of iterations in level1. Also, used to grow
            number of iterations in consecutive levels, default = 100000
            Ex: If l1_ittr = 10 and levels = 10
            l1 will have 10 iterations
            l2 with transit will have 15
            l2 without transit will have 20
            l3 with transit will have 25
            l3 without transit will have 30
            ...
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

    ** n_embedding is set to channels in level1
    ** weight normalization is used instead of equalized normalization
    ** Known issue - retains unused parameters
    """
    def __init__(self,
                 levels: int = 4,
                 l1_size: tuple = (4, 4),
                 l1_ittr: int = 100000,
                 growth_rate: int = 64,
                 pow_gr: bool = True,
                 rgb: bool = True,
                 max_channels: int = 256,
                 **kwargs):
        super(PGGAN, self).__init__()

        if not isinstance(levels, int):
            raise TypeError("PGGAN: levels must be int")
        if not isinstance(l1_size, tuple):
            raise TypeError("PGGAN: l1_size must be tuple")
        if not isinstance(l1_ittr, int):
            raise TypeError("PGGAN: l1_ittr must be int")
        if not isinstance(growth_rate, int):
            raise TypeError("PGGAN: growth_rate must be int")
        if not isinstance(pow_gr, bool):
            raise TypeError("PGGAN: pow_gr must be bool")

        c = 3 if rgb else 1
        self.c = c

        self.levels = levels
        self.l1_ittr = l1_ittr

        # compute ittr required per each stage using l1_ittr
        self.ittr_per_level = [x * l1_ittr//2 for x in
                               range(2, levels*2+1)]
        self.total_ittr = np.sum(self.ittr_per_level)
        self.max_iterations = self.total_ittr  # duplicate -- remove

        # compute possible stages. Given n levels, there are n*2-1 stages
        stages = [int(math.ceil(x / l1_ittr)) for x in self.ittr_per_level]
        self.stages = [{"current_level": ls, "transition": i % 2 == 1}
                       for i, ls in enumerate(stages)]

        # compute tensor_size at different levels
        t_sizes = []
        for i in range(levels):
            channels = growth_rate*2**i if pow_gr else growth_rate*(i+1)
            channels = min(channels, max_channels)
            mul = 2**(levels-i-1)
            t_sizes += [(1, channels, l1_size[0]*mul, l1_size[1]*mul)]
        self.t_sizes = t_sizes
        self.n_embedding = self.t_sizes[-1][1]

        # creating generator modules
        kwgs = {"normalization": None, "pre_nm": False,
                "weight_nm": True, "lklu_negslope": 0.2}
        # creating generator modules
        for i in range(1, levels+1):
            modules = [Convolution(t_sizes[-i][:2]+(1, 1) if i == 1 else
                                   t_sizes[-i+1][:2] + t_sizes[-i][2:],
                                   4 if i == 1 else 3, t_sizes[-i][1], 1,
                                   False if i == 1 else True, "lklu",
                                   transpose=True, **kwgs),
                       PixelWise()]
            # level i - features
            setattr(self, "glevel"+str(i), nn.Sequential(*modules))
            # level i - rgb/grey output
            setattr(self, "glevel"+str(i)+"out",
                    Convolution(t_sizes[-i], 3, c, 1, True, "sigm",
                                transpose=True, **kwgs))

        # creating discriminator modules
        kwgs["activation"] = "lklu"
        kwgs["lklu_negslope"] = 0.02
        for i in range(1, levels+1):
            modules = [Convolution(t_sizes[-i], 3, t_sizes[-i][1] if i == 1
                                   else t_sizes[-i+1][1],
                                   1, True, **kwgs), PixelWise()]
            if i == 1:
                modules += [nn.AvgPool2d(t_sizes[-i][2:], 1),
                            Linear(t_sizes[-i][:2], 1, "sigm", .2, bias=False)]
            else:
                modules += [nn.AvgPool2d(3, 2, 1)]
            # level i - features
            setattr(self, "dlevel"+str(i), nn.Sequential(*modules))
            # level i - rgb/grey output
            setattr(self, "dlevel"+str(i)+"in",
                    Convolution((1, c) + t_sizes[-i][2:], 3, t_sizes[-i][1],
                                1, True, **kwgs))
        self.register_buffer("iterations", torch.Tensor([0]).sum().long())
        self.update()

    def forward(self, tensor):
        if tensor.dim() == 2:
            tensor = tensor.view(tensor.size(0), -1, 1, 1)
            for i in range(1, self.current_level + 1):
                tensor = getattr(self, "glevel" + str(i))(tensor)
                if self.transition and (self.current_level - 1) == i:
                    # compute transition
                    transition = getattr(self, "glevel"+str(i)+"out")(tensor)
                if i != self.current_level and self.current_level > 1:
                    sz = (2*tensor.shape[2], 2*tensor.shape[3])
                    tensor = self.interpolate(tensor, sz)
            tensor = getattr(self, "glevel" + str(i) + "out")(tensor)
            if "transition" in locals():
                # if transition exists add tensor * alpha to
                # transition * (1 - alpha)
                tensor = self.interpolate(transition, tensor.shape[2:]) * \
                    (1 - self.alpha) + tensor * self.alpha
            # tensor = torch.sigmoid(tensor)
            tensor = tensor.clamp(0, 1)

        elif tensor.dim() == 4 and tensor.size(1) == self.c:
            if tensor.shape[2] != self.required_size[0] or \
               tensor.shape[3] != self.required_size[1]:
                # auto adjust size for discriminator
                tensor = self.interpolate(tensor, self.required_size)

            if self.transition and self.current_level > 1:
                # compute transition
                h, w = tensor.shape[2:]
                h, w = h//2, w//2
                transition = getattr(self, "dlevel"+str(self.current_level-1) +
                                     "in")(self.interpolate(tensor, (h, w)))
            tensor = getattr(self, "dlevel" + str(self.current_level) +
                             "in")(tensor)
            for i in range(self.current_level, 0, -1):
                tensor = getattr(self, "dlevel" + str(i))(tensor)
                if i == (self.current_level - 1) and "transition" in locals():
                    # if transition exists add tensor * alpha to
                    # transition * (1 - alpha)
                    tensor = getattr(self, "dlevel" + str(i))(transition) * \
                        (1 - self.alpha) + tensor * self.alpha
        return tensor

    def interpolate(self, tensor, size):
        return F.interpolate(tensor, size=size,
                             mode="bilinear", align_corners=True)

    def update(self, lr: float = 0.0001):
        r"""Checks level using iterations and ittr_per_level (precomputed
        during initialization), to update alpha and optimizer.
        Using current_level, generator and discriminator optimizer are
        dynamically adjusted with higher learning rate for current_level when
        compared to previous levels (if any).
        Further, generator learning rate is greater than discriminator lr.

        One can always use a different optimizer, use g_params and d_params to
        get a list of generator and discriminator parameters.
        """
        for i, x in enumerate(np.cumsum(self.ittr_per_level)):
            if x > self.iterations:
                break
        if hasattr(self, "current_level"):
            if (self.current_level == self.stages[i]["current_level"] and
               self.transition == self.stages[i]["transition"]):
                if self.transition:
                    n = np.cumsum(self.ittr_per_level)[i-1]
                    self.alpha = float(self.iterations - n) / \
                        self.ittr_per_level[i]
                # no change
                return None
        # updates
        self.current_level = self.stages[i]["current_level"]
        self.transition = self.stages[i]["transition"]
        self.alpha = 0.5
        if self.transition:
            n = np.cumsum(self.ittr_per_level)[i-1]
            self.alpha = float(self.iterations - n) / self.ittr_per_level[i]
        self.required_size = self.t_sizes[self.levels - self.current_level][2:]
        self.tensor_size = self.t_sizes[- self.current_level]
        # generator optimizer
        g_lr, d_lr, low_lr = lr*2, lr, lr/2
        params = []
        for i in range(1, self.current_level + 1):
            lr = g_lr if self.current_level == i else low_lr
            for p in getattr(self, "glevel" + str(i)).parameters():
                params.append({"params": p, "lr": lr})
            if self.transition and (self.current_level - 1) == i:
                # transition to output parameters
                for p in getattr(self, "glevel"+str(i)+"out").parameters():
                    params.append({"params": p, "lr": low_lr})
        # to output parameters
        for p in getattr(self, "glevel"+str(i)+"out").parameters():
            params.append({"params": p, "lr": g_lr})
        self.g_optimizer = torch.optim.Adam(params, weight_decay=0.00005,
                                            amsgrad=True)
        # discriminator optimizer
        params = []
        if self.transition and self.current_level > 1:
            # input to transition parameters
            for p in getattr(self, "dlevel" + str(self.current_level-1) +
                             "in").parameters():
                params.append({"params": p, "lr": low_lr})
        # input to parameters
        for p in getattr(self, "dlevel" + str(self.current_level) +
                         "in").parameters():
            params.append({"params": p, "lr": d_lr})
        for i in range(self.current_level, 0, -1):
            lr = d_lr if self.current_level == i else low_lr
            for p in getattr(self, "dlevel" + str(i)).parameters():
                params.append({"params": p, "lr": lr})
        self.d_optimizer = torch.optim.Adam(params, weight_decay=0.00005,
                                            amsgrad=True)

    def g_params(self):
        r""" generates a list of all generator parameters.
        """
        gparams = []
        for i in range(1, self.levels + 1):
            gparams += list(getattr(self, "glevel"+str(i)+"out").parameters())
            gparams += list(getattr(self, "glevel"+str(i)).parameters())
        return gparams

    def d_params(self):
        r""" generates a list of all discriminator parameters.
        """
        dparams = []
        for i in range(1, self.levels + 1):
            dparams += list(getattr(self, "dlevel"+str(i)+"in").parameters())
            dparams += list(getattr(self, "dlevel"+str(i)).parameters())
        return dparams

    def noisy_latent(self, n):
        r""" generates the noisy tensor for generator.
        """
        return torch.randn(n, self.n_embedding, requires_grad=True)


# from tensormonk.layers import Convolution, Linear
# from tensormonk.normalizations.pixelwise import PixelWise
# test = PGGAN(4, (4, 4), 100000, 128, False, True)
# test.t_sizes
# np.sum([np.prod(p.shape) for p in test.parameters()]) * 4 / 1024 / 1024
# for x in range(int(np.sum(test.ittr_per_level) / 50000)):
#     test.iterations.add_(50000)
#     test.update()
#     print(test.iterations, test.current_level, test.transition, test.alpha,
#           test(torch.rand(1, 256)).shape)
#     print(test.iterations, test.current_level, test.transition, test.alpha,
#           test(torch.rand(1, 3, *test.required_size)).shape)
