""" TensorMONK :: data :: FolderITTR """

import os
import errno
import torch
import torch.nn.functional as F
from PIL import Image as ImPIL
from torchvision import transforms
_totensor = transforms.ToTensor()


def totensor(input, t_size: tuple = None):
    r"""Converts image_file or PIL image to torch tensor.

    Args:
        input (str/pil-image): full path of image or pil-image
        t_size (list, optional): tensor_size in BCHW, used to resize the input
    """
    if isinstance(input, torch.Tensor):
        if t_size is not None:
            if len(t_size) == input.dim() == 4:
                if t_size[2] != input.size(2) or t_size[3] != input.size(3):
                    input = F.interpolate(input, size=t_size[2:])
        return input

    if isinstance(input, str):
        if not os.path.isfile(input):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                    input)
        input = ImPIL.open(input).convert("RGB")

    if ImPIL.isImageType(input):
        if t_size is not None:
            if t_size[1] == 1:
                input = input.convert("L")
            if t_size[2] != input.size[1] or t_size[3] != input.size[0]:
                input = input.resize((t_size[3], t_size[2]), ImPIL.BILINEAR)
    else:
        raise TypeError("totensor: input must be str/pil-imgage: "
                        "{}".format(type(input).__name__))
    tensor = _totensor(input)
    if tensor.dim() == 2:
        tensor.unsqueeze_(0)
    return tensor
