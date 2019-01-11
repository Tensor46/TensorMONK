""" TensorMONK :: layers :: Detail-Preserving Pooling """

__all__ = ["DetailPooling", ]

import torch
import torch.nn as nn
import torch.nn.functional as F


class DetailPooling(nn.Module):
    """ Implemented - https://arxiv.org/pdf/1804.04076.pdf """
    def __init__(self, tensor_size, asymmetric=False, lite=True,
                 *args, **kwargs):
        super(DetailPooling, self).__init__()

        # Notes: A user study in [31] showed that on average people preferred
        # DPID (0.5 <= lambda <= 1) over all considered downscaling techniques.
        self._lambda = nn.Parameter(torch.Tensor(1))
        self._lambda.data.mul_(0).add_(.6)

        self._alpha = nn.Parameter(torch.Tensor(1))
        self._alpha.data.mul_(0).add_(.1)

        self.asymmetric = asymmetric
        self.lite = lite

        if self.lite:
            # non trainable
            self.weight = torch.FloatTensor([[[[1, 2, 1]]]])
            self.weight = self.weight.expand((tensor_size[1], 1, 1, 3))
        else:
            # trainable
            self.weight = nn.Parameter(torch.rand(*(tensor_size[1], 1, 3, 3)))
            self.weight = nn.init.xavier_normal_(self.weight, gain=0.01)

        # Computing tensor_size
        self.tensor_size = tensor_size[:2] + \
            F.avg_pool2d(torch.rand(1, 1, tensor_size[2],
                                    tensor_size[3]), (2, 2)).size()[2:]

    def forward(self, tensor):

        # non-negative alpha and lambda - requires update
        self._alpha.data.pow_(2).pow_(.5)
        self._lambda.data.pow_(2).pow_(.5)

        # equation 2 - linearly downscaled image
        padded_tensor = F.pad(tensor, (1, 1, 1, 1), mode="replicate")
        if self.lite:
            if tensor.is_cuda and not self.weight.is_cuda:
                self.weight = self.weight.cuda()
            equation2 = F.conv2d(F.conv2d(padded_tensor, self.weight,
                                          groups=tensor.size(1)),
                                 self.weight.transpose(2, 3),
                                 groups=tensor.size(1)).div(16)
        else:
            equation2 = F.conv2d(padded_tensor, self.weight,
                                 groups=tensor.size(1))

        eps = 1e-6
        if self.asymmetric:
            # equation 6 -  asymmetric variant
            equation56 = equation2.mul(-1).add(tensor).clamp(0).pow(2)
            equation56 = equation56.add(eps**2).pow(2).pow(self._lambda)
        else:
            # equation 5 -  charbonnier penalty
            equation56 = equation2.mul(-1).add(tensor).pow(2).add(eps**2)
            equation56 = equation56.pow(2).pow(self._lambda)

        # equation 4 - adding alpha - trainable parameter
        equation4 = equation56.add(self._alpha)
        # equation 7 - normalizing
        equation7 = equation4.div(F.avg_pool2d(F.pad(equation4, (0, 1, 0, 1),
                                                     mode="replicate"),
                                               (2, 2), (1, 1)).add(1e-8))
        # equation 8 - final DPP
        equation8 = F.avg_pool2d(tensor.mul(equation7), (2, 2))
        return equation8


# tensor_size = (3,3,10,10)
# x = torch.rand(*tensor_size)
# test = DetailPooling(tensor_size)
# test(x).size()
# import numpy as np
# from PIL import Image as ImPIL
# image = ImPIL.open("../data/test.jpeg")
# tensor = np.array(image).astype(np.float32).transpose(2, 0, 1)[np.newaxis]
# tensor = torch.from_numpy(tensor / 255.)
# test = DetailPooling((1, 3, 256, 256), True, False)
# ImPIL.fromarray((test(tensor)).clamp(0, 1).mul(255.)[0,].data.numpy()
#                 .transpose(1, 2, 0).astype(np.uint8))
