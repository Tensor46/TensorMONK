""" TensorMONK :: layers :: DoGBlob """

__all__ = ["DoG", "DoGBlob", "GaussianBlur"]


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def GaussianKernel(sigma: float = 1., width: int = 0):
    r""" Creates Gaussian kernel given sigma and width. n_stds is fixed to 3.

    Args:
        sigma: spread of gaussian. If 0. or None, sigma is calculated using
            width and n_stds = 3. default is 1.
        width: width of kernel. If 0. or None, width is calculated using sigma
            and n_stds = 3. width is odd number. default is 0.

    Return:
        4D torch.Tensor of shape (1, 1, width, with)
    """
    assert not ((width is None or width == 0) and
                (sigma is None or sigma == 0)), \
        "GaussianKernel :: both sigma ({}) & width ({}) are not valid".format(
        sigma, width)

    if width is None or width == 0:
        width = int(2.0 * 3.0 * sigma + 1.0)
    if width % 2 == 0:
        width += 1

    if sigma is None or sigma == 0:
        sigma = (width - 1)/6.
    half = width//2
    x, y = np.meshgrid(np.linspace(-half, half, width),
                       np.linspace(-half, half, width), indexing='xy')
    w = np.exp(- (x**2 + y**2) / (2.*(sigma**2)))
    w /= np.sum(w)
    return torch.from_numpy(w.astype(np.float32)).view(1, 1, width, width)


class GaussianBlur(nn.Module):
    r""" Blurs each channel of the input tensor with a Gaussian kernel of given
    sigma and width. Refer to GaussianKernel for details on kernel computation.

    Args:
        sigma: spread of gaussian. If 0. or None, sigma is calculated using
            width and n_stds = 3. default is 1.
        width: width of kernel. If 0. or None, width is calculated using sigma
            and n_stds = 3. default is 0.

    Return:
        Blurred 4D BCHW torch.Tensor with size same as input tensor
    """
    def __init__(self, sigma: float = 1., width: int = 0):
        super(GaussianBlur, self).__init__()

        self.register_buffer("gaussian", GaussianKernel(sigma, width))
        self.pad = self.gaussian.shape[2] // 2

    def forward(self, tensor):
        c = tensor.size(1)
        w = self.gaussian.repeat(c, 1, 1, 1).to(tensor.device)
        return F.conv2d(tensor, w, padding=(self.pad, self.pad), groups=c)


class DoG(nn.Module):
    r""" Computes difference of two blurred tensors with different gaussian
    kernels.

    Args:
        sigma1: spread of first gaussian. If 0. or None, sigma1 is calculated
            using width1 and n_stds = 3. default is 1.
        sigma2: spread of second gaussian. If 0. or None, sigma2 is calculated
            using width2 and n_stds = 3. default is 1.
        width1: width of first kernel. If 0. or None, width1 is calculated
            using sigma1 and n_stds = 3. default is 0.
        width2: width of second kernel. If 0. or None, width2 is calculated
            using sigma2 and n_stds = 3. default is 0.

    Return:
        4D BCHW torch.Tensor with size same as input torch.Tensor
    """
    def __init__(self, sigma1: float = 0., sigma2: float = 0.,
                 width1: int = 5, width2: int = 9):
        super(DoG, self).__init__()

        self.gaussian1 = GaussianBlur(sigma1, width1)
        self.gaussian2 = GaussianBlur(sigma2, width2)

    def forward(self, tensor):
        return self.gaussian1(tensor) - self.gaussian2(tensor)


class DoGBlob(nn.Module):
    r""" Accumulates DoG's at different scales.

    Args:
        scales: a list of various scales DoG is computed
        sigma1: spread of first gaussian. If 0. or None, sigma1 is calculated
            using width1 and n_stds = 3. default is 1.
        sigma2: spread of second gaussian. If 0. or None, sigma2 is calculated
            using width2 and n_stds = 3. default is 1.
        width1: width of first kernel. If 0. or None, width1 is calculated
            using sigma1 and n_stds = 3. default is 0.
        width2: width of second kernel. If 0. or None, width2 is calculated
            using sigma2 and n_stds = 3. default is 0.

    Return:
        4D BCHW torch.Tensor with size same as input torch.Tensor
    """
    def __init__(self,
                 scales: list = [0.75, 1, 1.25],
                 sigma1: float = 0., sigma2: float = 0.,
                 width1: int = 5, width2: int = 9):
        super(DoGBlob, self).__init__()

        self.dog = DoG(sigma1, sigma2, width1, width2)
        self.scales = scales

    def forward(self, tensor):
        t_size = tensor.shape

        blob_tensor = torch.zeros(*t_size).to(tensor.device)
        for x in self.scales:
            if x == 1.:
                blob_tensor = blob_tensor + self.dog(tensor)
            else:
                resize = F.interpolate(tensor, scale_factor=x,
                                       mode="bilinear", align_corners=True)
                resize = F.interpolate(self.dog(resize), size=t_size[2:],
                                       mode="bilinear", align_corners=True)
                blob_tensor = blob_tensor + resize
        return blob_tensor
