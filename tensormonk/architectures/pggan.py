""" TensorMONK :: architectures """

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers import Convolution, Linear
import numpy as np
import math


class PGGAN(nn.Module):
    r"""Progressing growth of GANs - https://arxiv.org/pdf/1710.10196.pdf
    There are obviously some difference!

    Args:
        n_embedding (int): length of latent vector, default = 600
        levels (int): levels to grow, default = 4
            Ex: if l1_size = (4, 4) and levels = 4, final size or
            max_tensor_size = 32x32
        l1_size (tuple): inital size, default = (4, 4)
        l1_ittr: number of iterations in level1. Also, used to grow
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
        pow_gr :: when True the growth of each layer is by a power of 2 else
            a multiple of layers, default=True
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
        rgb :: True/False, default=True

    ** Known issue - retains unused parameters
    """
    def __init__(self,
                 n_embedding: int = 600,
                 levels: int = 4,
                 l1_size: tuple = (4, 4),
                 l1_ittr: int = 100000,
                 growth_rate: int = 32,
                 pow_gr: bool = True,
                 rgb: bool = True,
                 **kwargs):
        super(PGGAN, self).__init__()

        if not isinstance(n_embedding, int):
            raise TypeError("PGGAN: n_embedding must be int")
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
        self.n_embedding = n_embedding
        self.levels = levels
        self.l1_ittr = l1_ittr

        # compute ittr required per each stage using l1_ittr
        self.ittr_per_level = [x * l1_ittr//2 for x in
                               range(2, levels*2+1)]
        self.total_ittr = np.sum(self.ittr_per_level)
        # compute possible stages. Given n levels, there are n*2-1 stages
        stages = [int(math.ceil(x / l1_ittr)) for x in self.ittr_per_level]
        self.stages = [{"current_level": ls, "transition": i % 2 == 1}
                       for i, ls in enumerate(stages)]
        # compute tensor_size at different levels
        t_sizes = []
        for i in range(levels):
            channels = growth_rate*2**i if pow_gr else growth_rate*(i+1)
            mul = 2**(levels-i-1)
            t_sizes += [(1, channels, l1_size[0]*mul, l1_size[1]*mul)]
        self.t_sizes = t_sizes
        # creating genrator modules
        kwgs = {"normalization": "pixelwise", "pre_nm": True,
                "equalized": True}
        # creating genrator modules
        for i in range(1, levels+1):
            if i == 1:  # level i - features
                modules = [Linear((1, n_embedding), int(np.prod(t_sizes[-i])),
                                  "", out_shape=t_sizes[-i][1:]),
                           Convolution(t_sizes[-i], 3, t_sizes[-i][1], 1,
                                       True, "lklu", transpose=True, **kwgs)]
                setattr(self, "glevel1", nn.Sequential(*modules))
            else:
                setattr(self, "glevel"+str(i),
                        Convolution(t_sizes[-i+1], 3, t_sizes[-i][1], 2, True,
                                    "lklu", transpose=True,
                                    maintain_out_size=True, **kwgs))
            # level i - rgb/grey output
            setattr(self, "glevel"+str(i)+"out",
                    Convolution(t_sizes[-i], 3, c, 1,
                                True, "sigm", transpose=True, **kwgs))

        # creating discriminator modules
        kwgs["pre_nm"] = False
        kwgs["activation"] = "lklu"
        for i in range(1, levels+1):
            if i == 1:  # level i - features
                modules = [Convolution(t_sizes[-i], 4, t_sizes[-i][1], 1,
                                       False, **kwgs),
                           Linear((1, t_sizes[-i][1]), 1, "sigm")]
                setattr(self, "dlevel1", nn.Sequential(*modules))
            else:
                setattr(self, "dlevel"+str(i),
                        Convolution(t_sizes[-i][:2]+t_sizes[-i][2:], 3,
                                    t_sizes[-i+1][1], 2, True, **kwgs))
            # level i - rgb/grey output
            setattr(self, "dlevel"+str(i)+"in",
                    Convolution((1, c) + t_sizes[-i][2:], 3, t_sizes[-i][1],
                                1, True, **kwgs))
        self.register_buffer("iterations", torch.Tensor([0]).sum().long())
        self.update()

    def forward(self, tensor):
        if tensor.dim() == 2:
            self.iterations.add_(1)
            self.update()

            for i in range(1, self.current_level + 1):
                tensor = getattr(self, "glevel" + str(i))(tensor)
                if self.transition and self.current_level > 1:
                    # compute transition
                    if (self.current_level - 1) == i:
                        transition = getattr(self,
                                             "glevel"+str(i)+"out")(tensor)
            tensor = getattr(self, "glevel" + str(i) + "out")(tensor)
            if "transition" in locals():
                # if transition exists add tensor * alpha to
                # transition * (1 - alpha)
                tensor = F.interpolate(transition, size=tensor.shape[2:]) * \
                    (1 - self.alpha) + tensor * self.alpha

        elif tensor.dim() == 4:
            if tensor.shape[2] != self.required_size[0] or \
               tensor.shape[3] != self.required_size[1]:
                # auto adjust size for discriminator
                tensor = F.interpolate(tensor, size=self.required_size)

            if self.transition and self.current_level > 1:
                # compute transition
                h, w = tensor.shape[2:]
                h, w = h//2, w//2
                transition = getattr(self, "dlevel"+str(self.current_level-1) +
                                     "in")(F.interpolate(tensor, size=(h, w)))
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

    def update(self):
        for i, x in enumerate(np.cumsum(self.ittr_per_level)):
            if x > self.iterations:
                break
        self.current_level = self.stages[i]["current_level"]
        self.transition = self.stages[i]["transition"]
        self.alpha = 0.5
        if self.transition:
            n = np.cumsum(self.ittr_per_level)[i-1]
            self.alpha = float(self.iterations - n) / self.ittr_per_level[i]
        self.required_size = self.t_sizes[self.levels - self.current_level][2:]
        self.tensor_size = self.t_sizes[- self.current_level - 1]

    def trainable_parameters_per_level(self):
        g_params, d_params = [], []
        # collect generator parameters
        for i in range(1, self.current_level + 1):
            g_params += list(getattr(self, "glevel" + str(i)).parameters())
            if self.transition and self.current_level > 1:
                if (self.current_level - 1) == i:
                    g_params += list(getattr(self, "glevel"+str(i) +
                                             "out").parameters())
        g_params += list(getattr(self, "glevel" + str(i) + "out").parameters())
        # collect discriminator parameters
        if self.transition and self.current_level > 1:
            d_params += list(getattr(self, "dlevel"+str(self.current_level-1) +
                                     "in").parameters())
        d_params += list(getattr(self, "dlevel" + str(self.current_level) +
                                 "in").parameters())
        for i in range(self.current_level, 0, -1):
            d_params += list(getattr(self, "dlevel" + str(i)).parameters())
            if i == (self.current_level - 1) and self.transition:
                d_params += list(getattr(self, "dlevel" + str(i)).parameters())
        return g_params, d_params

    def trainable_parameters(self):
        g_params, d_params = [], []
        # collect generator parameters
        for i in range(1, self.levels + 1):
            g_params += list(getattr(self, "glevel"+str(i)).parameters())
            g_params += list(getattr(self, "glevel"+str(i)+"out").parameters())
        # collect discriminator parameters
        for i in range(self.levels, 0, -1):
            d_params += list(getattr(self, "dlevel"+str(i)).parameters())
            d_params += list(getattr(self, "dlevel"+str(i)+"in").parameters())
        return g_params, d_params


# from tensormonk.layers import Convolution, Linear
# test = PGGAN(600, 6, (4, 4), 100000, 64, False, True)
# np.sum([np.prod(p.shape) for p in test.parameters()])
# for x in range(int(np.sum(test.ittr_per_level) / 5000)):
#     test.iterations.add_(5000)
#     test(torch.rand(1, 600)).shape
#     test(torch.rand(1, 3, *test.required_size)).shape
#     print(test.iterations, test.current_level, test.transition, test.alpha)
