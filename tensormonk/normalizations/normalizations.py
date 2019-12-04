""" TensorMONK :: layers :: Normalizations """

__all__ = ["Normalizations", "FrozenBNorm2d"]

import torch
import numpy as np
from .pixelwise import PixelWise
from .categoricalbatch import CategoricalBNorm


class FrozenBNorm2d(torch.nn.Module):
    def __init__(self, num_features: int, **kwargs):
        super(FrozenBNorm2d, self).__init__()
        self.register_buffer("weight", torch.zeros(num_features))
        self.register_buffer("bias", torch.ones(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.register_buffer("num_batches_tracked", torch.tensor(0))

    def forward(self, tensor: torch.Tensor):
        return torch.nn.functional.batch_norm(
            tensor, self.running_mean, self.running_var,
            self.weight, self.bias, False)


def Normalizations(tensor_size=None, normalization=None, available=False,
                   just_flops=False, **kwargs):
    r"""Does normalization on 4D tensor.

    Args:
        tensor_size: shape of tensor in BCHW
            (None/any integer >0, channels, height, width)
        normalization: None/batch/group/instance/layer/pixelwise/cbatch/
            frozenbnorm
        available: if True, returns all available normalization methods
        groups: for group (GroupNorm), when not provided groups is the center
            value of all possible - ex: for a tensor_size[1] = 128, groups is
            set to 16 as the possible groups are [1, 2, 4, 8, 16, 32, 64, 128]
        affine: for group and instance normalization, default False
        elementwise_affine: for layer normalization. default True
    """
    list_available = ["batch", "group", "instance", "layer", "pixelwise",
                      "cbatch", "frozenbnorm"]
    if available:
        return list_available

    normalization = normalization.lower()
    assert normalization in list_available, \
        "Normalization must be None/" + "/".join(list_available)

    if normalization == "frozenbnorm":
        if just_flops:
            # inference -> (x - mean) / (std + eps) * gamma + beta
            _eps_adds = tensor_size[1]
            _element_muls_adds = 4
            return _element_muls_adds * np.prod(tensor_size[1:]) + _eps_adds
        return FrozenBNorm2d(tensor_size[1])
    elif normalization == "batch":
        if just_flops:
            # inference -> (x - mean) / (std + eps) * gamma + beta
            _eps_adds = tensor_size[1]
            _element_muls_adds = 4
            return _element_muls_adds * np.prod(tensor_size[1:]) + _eps_adds
        return torch.nn.BatchNorm2d(tensor_size[1])

    elif normalization == "group":
        affine = kwargs["affine"] if "affine" in \
            kwargs.keys() else False

        if just_flops:
            # inference -> (x - mean) / (std + eps) * gamma + beta
            _eps_adds = tensor_size[1]
            _element_muls_adds = (4 if affine else 2)
            return _element_muls_adds * np.prod(tensor_size[1:]) + _eps_adds
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
        if just_flops:
            # inference -> (x - mean) / (std + eps)
            _eps_adds = tensor_size[1]
            _element_muls_adds = (4 if affine else 2)
            flops = _element_muls_adds * np.prod(tensor_size[1:]) + _eps_adds
            # mean computation on the fly as track_running_stats=False
            flops += np.prod(tensor_size[1:])
            # std computation on the fly as track_running_stats=False
            flops += np.prod(tensor_size[1:])*3 + np.prod(tensor_size[2:]) + \
                + tensor_size[1]
            # inference -> (x - mean) / (std + eps) * gamma + beta
            return flops
        return torch.nn.InstanceNorm2d(tensor_size[1], affine=affine)

    elif normalization == "layer":
        elementwise_affine = kwargs["elementwise_affine"] if \
            "elementwise_affine" in kwargs.keys() else True
        if just_flops:
            # inference -> (x - mean) / (std + eps) * gamma + beta
            _eps_adds = tensor_size[1]
            _element_muls_adds = 4 if elementwise_affine else 2
            return _element_muls_adds * np.prod(tensor_size[1:]) + _eps_adds
        return torch.nn.LayerNorm(tensor_size[1:],
                                  elementwise_affine=elementwise_affine)

    elif normalization == "pixelwise":
        if just_flops:
            # inference -> x / x.pow(2).sum(1).pow(0.5).add(eps)
            return np.prod(tensor_size[1:])*3 + np.prod(tensor_size[2:])*2
        return PixelWise()

    elif normalization == "cbatch":
        if just_flops:
            # inference -> (x - mean) / (std + eps) * gamma
            # TODO: update for n_latent
            _eps_adds = tensor_size[1]
            _element_muls_adds = 3
            return _element_muls_adds * np.prod(tensor_size[1:]) + _eps_adds
        return CategoricalBNorm(tensor_size, **kwargs)
