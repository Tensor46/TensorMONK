""" TensorMONK's :: utils                                                    """

__all__ = ["utils"]

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def DoH(tensor:torch.Tensor, width:int=3):
    """ Determinant of Hessian """
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
    """
        Hessian Blob!
    """
    def __init__(self, min_width:int=3, max_width:int=15, blur_w:int=0):
        super(HessianBlob, self).__init__()
        if min_width%2 == 0:
            min_width += 1
        self.min_width = min_width

        if max_width%2 == 0:
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
            blob_tensor = blob_tensor + DoH(blur if width > 3 and \
                hasattr(self, "blur") else tensor, width)
        return blob_tensor


def GaussianKernel(sigma:float=1., width:int=0):
    """ Gaussian Kernel with given sigma and width, when one is 0 or None a
        rough estimate is done using n_stds = 3. """

    assert not ((width is None or width == 0) and (sigma is None or sigma == 0)), \
        "GaussianKernel :: both sigma ({}) & width ({}) are not valid".format(
        sigma, width)

    if width is None or width == 0:
        width = int(2.0 * 3.0 * sigma + 1.0)
    if width%2 == 0:
        width += 1

    if sigma is None or sigma == 0:
        sigma = (width - 1)/6.

    x,y = np.meshgrid(np.linspace(-(width//2), width//2, width),
                      np.linspace(-(width//2), width//2, width), indexing='xy')
    w = np.exp(- (x**2 + y**2) / (2.*(sigma**2)) )
    w /= np.sum(w)
    return torch.from_numpy(w.astype(np.float32)).view(1, 1, width, width)


class GaussianBlur(nn.Module):
    """
        Gaussian Blur!
            when sigma/width is 0/None a rough estimate is done using n_stds=3
    """
    def __init__(self, sigma:float=1., width:int=0):
        super(GaussianBlur, self).__init__()

        self.register_buffer("gaussian", GaussianKernel(sigma, width))
        self.pad = self.gaussian.shape[2] // 2

    def forward(self, tensor):
        c = tensor.size(1)
        w = self.gaussian.repeat(c, 1, 1, 1).to(tensor.device)
        return F.conv2d(tensor, w, padding=(self.pad, self.pad), groups=c)


class DoG(nn.Module):
    """
        Difference of Gaussians!
    """
    def __init__(self, sigma1:float=0., sigma2:float=0.,
                 width1:int=5, width2:int=9):
        super(DoG, self).__init__()

        self.gaussian1 = GaussianBlur(sigma1, width1)
        self.gaussian2 = GaussianBlur(sigma2, width2)

    def forward(self, tensor):
        return self.gaussian1(tensor) - self.gaussian2(tensor)


class DoGBlob(nn.Module):
    """
        Accumulates DoGs at different scales!
    """
    def __init__(self, dog_params:list=[0., 0., 5, 9],
                 scales:list=[0.75, 1, 1.25]):
        super(DoGBlob, self).__init__()

        self.dog = DoG(*dog_params)
        self.scales = scales

    def forward(self, tensor):
        t_size = tensor.shape

        blob_tensor = torch.zeros(*t_size).to(tensor.device)
        for x in self.scales:
            if x == 1.:
                blob_tensor = blob_tensor + self.dog(tensor)
            else:
                blob_tensor = blob_tensor + \
                    F.interpolate(self.dog(F.interpolate(tensor, scale_factor=x,
                    mode="bilinear", align_corners=True)), size=t_size[2:],
                    mode="bilinear", align_corners=True)
        return blob_tensor


def corr_1d(tensor_a:torch.Tensor, tensor_b:torch.Tensor):
    assert tensor_a.dim() == 2 and tensor_b.dim() == 2, \
        "correlation_1d :: tensor_a and tensor_b must be 2D"

    return (tensor_a.mul(tensor_b).mean(1) - tensor_a.mean(1)*tensor_b.mean(1))/\
        ((tensor_a.pow(2).mean(1) - tensor_a.mean(1).pow(2)).pow(0.5) *
         (tensor_b.pow(2).mean(1) - tensor_b.mean(1).pow(2)).pow(0.5))


def xcorr_1d(tensor:torch.Tensor):
    assert tensor.dim() == 2, "xcorr_1d :: tensor must be 2D"
    n = tensor.size(0)
    return (tensor.view(n, 1, -1).mul(tensor.view(1, n, -1)).mean(2)
        - tensor.view(n, 1, -1).mean(2).mul(tensor.view(1, n, -1).mean(2))) / \
        ((tensor.view(n, 1, -1).pow(2).mean(2) -
          tensor.view(n, 1, -1).mean(2).pow(2)).pow(0.5) *
         (tensor.view(1, n, -1).pow(2).mean(2) -
          tensor.view(1, n, -1).mean(2).pow(2)).pow(0.5))


class utils:
    corr_1d = corr_1d
    xcorr_1d = xcorr_1d
    DoH = DoH
    HessianBlob = HessianBlob
    GaussianKernel = GaussianKernel
    DoG = DoG
    DoGBlob = DoGBlob
