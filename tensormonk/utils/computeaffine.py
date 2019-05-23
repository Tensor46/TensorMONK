""" TensorMONK :: utils """

import cv2
import math
import torch
import numpy as np


def compute_affine(center: tuple = (0, 0),
                   angle: float = 0.,
                   translate: tuple = (0, 0),
                   scale: float = 1.,
                   shear: float = 0.,
                   inverted: bool = False,
                   to_theta: bool = False):
    r"""Computes affine matrix for torch.nn.functional.affine_grid and
    PIL image.

    Args:
        center (tuple): image center (x, y), default=(0, 0)
        angle (int/float): rotation angle in degrees (-180 to 180), default=0
        translate (tuple): translations in (x, y), default=(0, 0)
        scale (float): image scale factor, default=1.
        shear (float): shear angle in degrees (-180 to 180), default=0
        inverted (bool): when True, delivers the inverted affine, default=False
        to_theta (bool): when True, delivers the affine matrix compatible with
            torch.nn.functional.affine_grid, default=False

    Returns:
        np.array / torch.Tensor of shape 2x3

    **Built on torchvision.transform._get_inverse_affine_matrix
    """
    cx, cy = center
    tx, ty = translate
    if to_theta:
        # convert to theta for torch.nn.functional.affine_grid
        tx, ty = tx/cx, ty/cy
        cx, cy = 0, 0

    rads, shear = math.radians(angle), math.radians(shear)
    scale = 1.0 / scale
    tm = np.zeros((2, 3)).astype(np.float32)
    tm[0, 0] = math.cos(rads + shear)
    tm[0, 1] = math.sin(rads + shear)
    tm[1, 0] = - math.sin(rads)
    tm[1, 1] = math.cos(rads)

    d = math.cos(rads + shear) * math.cos(rads) + \
        math.sin(rads + shear) * math.sin(rads)
    tm = scale / d * tm

    tm[0, 2] += tm[0, 0] * (-cx - tx) + tm[0, 1] * (-cy - ty)
    tm[1, 2] += tm[1, 0] * (-cx - tx) + tm[1, 1] * (-cy - ty)

    tm[0, 2] += cx
    tm[1, 2] += cy

    if inverted:
        tm = cv2.invertAffineTransform(tm)
    return torch.from_numpy(tm).float() if to_theta else tm
