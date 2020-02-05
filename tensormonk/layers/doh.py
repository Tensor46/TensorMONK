""" TensorMONK :: layers :: HessianBlob """

__all__ = ["DoH", "HessianBlob"]


import torch
import torch.nn as nn
import torch.nn.functional as F
from .dog import GaussianBlur


def DoH(tensor: torch.Tensor, width: int = 3):
    r""" Computes determinant of Hessian of BCHW torch.Tensor using the cornor
    pixels of widthxwidth patch.

    Args:
        tensor: 4D BCHW torch.Tensor
        width: width of kernel, default = 3

    Return:
        4D BCHW torch.Tensor with size same as input
    """
    pad = width // 2
    padded = F.pad(tensor, [pad]*4)
    dx = padded[:, :, pad:-pad, width-1:] - padded[:, :, pad:-pad, :-width+1]
    dy = padded[:, :, width-1:, pad:-pad] - padded[:, :, :-width+1, pad:-pad]
    dx = F.pad(dx, [pad]*4)
    dy = F.pad(dy, [pad]*4)
    dxx = dx[:, :, pad:-pad, width-1:] - dx[:, :, pad:-pad, :-width+1]
    dyy = dy[:, :, width-1:, pad:-pad] - dy[:, :, :-width+1, pad:-pad]
    dxy = dx[:, :, width-1:, pad:-pad] - dx[:, :, :-width+1, pad:-pad]
    return (dxx*dyy - dxy**2)


class HessianBlob(nn.Module):
    r""" Aggregates determinant of Hessian with width ranging from min_width to
    max_width (skips every other).

    Args:
        min_width: minimum width of kernel, default = 3
        max_width: maximum width of kernel, default = 15
        blur_w: computes determinant of Hessian on a blurred image when
            blur_w > 3

    Return:
        4D BCHW torch.Tensor with size same as input tensor
    """
    def __init__(self,
                 min_width: int = 3,
                 max_width: int = 15,
                 blur_w: int = 0):

        super(HessianBlob, self).__init__()
        if min_width % 2 == 0:
            min_width += 1
        self.min_width = min_width

        if max_width % 2 == 0:
            max_width += 1
        self.max_width = max_width

        if blur_w >= 3:
            self.blur = GaussianBlur(0., blur_w)

    def forward(self, tensor):
        t_size = tensor.shape

        if hasattr(self, "blur"):
            blur = self.blur(tensor)
        blob_tensor = torch.zeros(*t_size).to(tensor.device)
        for width in range(self.min_width, self.max_width, 2):
            blob_tensor = blob_tensor + DoH(blur if width > 3 and
                                            hasattr(self, "blur") else tensor,
                                            width)
        return blob_tensor
