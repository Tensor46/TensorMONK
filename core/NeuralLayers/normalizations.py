""" TensorMONK's :: NeuralLayers :: Normalizations                           """

__all__ = ["Normalizations", ]

import torch
# ============================================================================ #


class PixelWise(torch.nn.Module):
    """ Implemented - https://arxiv.org/pdf/1710.10196.pdf """
    def __init__(self, eps=1e-8):
        super(PixelWise, self).__init__()
        self.eps = eps

    def forward(self, tensor):
        return tensor.div(tensor.pow(2).mean(1, True).add(self.eps).pow(.5))
# ============================================================================ #


def Normalizations(tensor_size, normalization, **kwargs):
    normalization = normalization.lower()
    assert normalization in ["batch", "group", "instance", "layer", "pixelwise"], \
        "Convolution's normalization must be None/batch/group/instance/layer/pixelwise"
    if normalization == "batch":
        return torch.nn.BatchNorm2d(tensor_size[1])
    elif normalization == "group":
        affine = kwargs["normalization_affine"] if "normalization_affine" in \
            kwargs.keys() else False
        if "normalization_groups" in kwargs.keys():
            return torch.nn.GroupNorm(kwargs["normalization_groups"], tensor_size[1], affine=affine)
        else:
            if tensor_size[1] % 4 == 0:
                return torch.nn.GroupNorm(4, tensor_size[1], affine=affine)
            elif tensor_size[1] % 4 != 0 and tensor_size[1] % 3 == 0:
                return torch.nn.GroupNorm(3, tensor_size[1], affine=affine)
            elif tensor_size[1] % 4 != 0 and tensor_size[1] % 3 != 0 and tensor_size[1] % 2 == 0:
                return torch.nn.GroupNorm(2, tensor_size[1], affine=affine)
            else:
                return torch.nn.LayerNorm(tensor_size[1:])
    elif normalization == "instance":
        affine = kwargs["normalization_affine"] if "normalization_affine" in \
            kwargs.keys() else False
        return torch.nn.InstanceNorm2d(tensor_size[1], affine=affine)
    elif normalization == "layer":
        elementwise_affine = kwargs["normalization_elementwise_affine"] if \
            "normalization_elementwise_affine" in kwargs.keys() else True
        return torch.nn.LayerNorm(tensor_size[1:], elementwise_affine=elementwise_affine)
    elif normalization == "pixelwise":
        return PixelWise()
