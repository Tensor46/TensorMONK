""" TensorMONK's :: NeuralLayers :: Normalizations                          """

__all__ = ["Normalizations", ]

import torch


class PixelWise(torch.nn.Module):
    r""" Implemented - https://arxiv.org/pdf/1710.10196.pdf """
    def __init__(self, eps=1e-8):
        super(PixelWise, self).__init__()
        self.eps = eps

    def forward(self, tensor):
        return tensor.div(tensor.pow(2).mean(1, True).add(self.eps).pow(.5))


def Normalizations(tensor_size=None, normalization=None, available=False,
                   **kwargs):
    r"""Does normalization on 4D tensor.

    Args:
        tensor_size: shape of tensor in BCHW
            (None/any integer >0, channels, height, width)
        normalization: None/batch/group/instance/layer/pixelwise
        available: if True, returns all available normalization methods
        groups: for group (GroupNorm), when not provided groups is the center
            value of all possible - ex: for a tensor_size[1] = 128, groups is
            set to 16 as the possible groups are [1, 2, 4, 8, 16, 32, 64, 128]
        affine: for group and instance normalization, default False
        elementwise_affine: for layer normalization. default True
    """
    list_available = ["batch", "group", "instance", "layer", "pixelwise"]
    if available:
        return list_available

    normalization = normalization.lower()
    assert normalization in list_available, \
        "Normalization must be None/" + "/".join(list_available)

    if normalization == "batch":
        return torch.nn.BatchNorm2d(tensor_size[1])

    elif normalization == "group":
        affine = kwargs["affine"] if "affine" in \
            kwargs.keys() else False

        if "groups" in kwargs.keys():
            return torch.nn.GroupNorm(kwargs["groups"], tensor_size[1],
                                      affine=affine)
        else:
            possible = [tensor_size[1]//i for i in range(tensor_size[1], 0, -1)
                        if tensor_size[1] % i == 0]
            groups = possible[len(possible)//2]
            return torch.nn.GroupNorm(groups, tensor_size[1], affine=affine)

    elif normalization == "instance":
        affine = kwargs["affine"] if "affine" in \
            kwargs.keys() else False
        return torch.nn.InstanceNorm2d(tensor_size[1], affine=affine)

    elif normalization == "layer":
        elementwise_affine = kwargs["elementwise_affine"] if \
            "elementwise_affine" in kwargs.keys() else True
        return torch.nn.LayerNorm(tensor_size[1:],
                                  elementwise_affine=elementwise_affine)

    elif normalization == "pixelwise":
        return PixelWise()
