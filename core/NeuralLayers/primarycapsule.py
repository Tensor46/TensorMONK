""" TensorMONK's :: NeuralLayers :: PrimaryCapsule                          """

import torch
from .convolution import Convolution


class PrimaryCapsule(torch.nn.Module):
    r""" Primary capsule from Dynamic Routing Between Capsules. A single
    convolution is used to generate all the capsules.
    Implemented -- https://arxiv.org/pdf/1710.09829.pdf

    Args:
        All args are similar to Convolution

        growth_rate (int, optional): used from custom blocks, default = 32
        block (nn.Module, optional): Any convolutional block can be used,
            default = Convolution
        n_capsules (int, required): number of capsules
        capsule_length (int, required): length of each capsule
        * out_channels must be equal to n_capsules*capsule_length

    Return:
        5D torch.Tensor of shape
            (None/any integer >0, capsule_length, height, width, n_capsules)
    """
    def __init__(self,
                 tensor_size,
                 filter_size,
                 out_channels: int = None,
                 strides: int = 1,
                 pad: bool = True,
                 activation: str = "relu",
                 dropout: float = 0.,
                 normalization: str = None,
                 pre_nm: bool = False,
                 groups: int = 1,
                 weight_nm: bool = False,
                 equalized: bool = False,
                 shift: bool = False,
                 growth_rate: int = 32,
                 block: torch.nn.Module = Convolution,
                 n_capsules: int = 8,
                 capsule_length: int = 32,
                 *args, **kwargs):
        super(PrimaryCapsule, self).__init__()
        # TODO: depreciate out_channels?
        if out_channels is None:
            out_channels = n_capsules*capsule_length
        assert out_channels == n_capsules*capsule_length, \
            "PrimaryCapsule -- out_channels!=n_capsules*capsule_length"

        self.primaryCapsules = block(tensor_size, filter_size, out_channels,
                                     strides, pad, activation, dropout,
                                     normalization, pre_nm, groups,
                                     weight_nm, equalized, shift,
                                     growth_rate=growth_rate, **kwargs)
        self.tensor_size = (6, capsule_length) + \
            self.primaryCapsules.tensor_size[2:] + (n_capsules,)

    def forward(self, tensor):
        tensor = self.primaryCapsules(tensor)
        tensor = tensor.view(-1, self.tensor_size[1], self.tensor_size[4],
                             self.tensor_size[2], self.tensor_size[3])
        return tensor.permute(0, 1, 3, 4, 2).contiguous()


# x = torch.rand(3,3,10,10)
# test = PrimaryCapsule((1, 3, 10, 10), (3, 3), 256, (2, 2), True, "relu", 0.,
#                       True, False, block=Convolution, n_capsules=8,
#                       capsule_length=32)
# test(x).size()
