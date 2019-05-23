""" TensorMONK :: layers :: SSIM """

__all__ = ["SSIM"]


import torch
import torch.nn as nn
from .dog import GaussianBlur


class SSIM(nn.Module):
    r""" Computes Structural Similarity Index (SSIM) between two tensors using
    a Gaussian kernel of given sigma and width. Refer to GaussianKernel for
    details on kernel computation. Works for inputs with range 0-1 or -1-1.

    Args:
        sigma: spread of gaussian. If 0. or None, sigma is calculated using
            width and n_stds = 3. default is 1.5
        width: width of kernel. If 0. or None, width is calculated using sigma
            and n_stds = 3. default is 11

    Return:
        torch.Tensor with SSIM scores for each sample
    """
    def __init__(self, sigma: float = 1.5, width: int = 11):
        super(SSIM, self).__init__()
        self.gblur = GaussianBlur(sigma, width)

    def forward(self, tensor_a, tensor_b):
        # average's
        mu_a, mu_b = self.gblur(torch.cat((tensor_a, tensor_b))).chunk(2, 0)
        mu_a2, mu_b2, mu_ab = mu_a.pow(2), mu_b.pow(2), mu_a*mu_b

        # variance's
        vr_a2, vr_b2, vr_ab = self.gblur(torch.cat(
            (tensor_a.pow(2), tensor_b.pow(2), tensor_a*tensor_b))).chunk(3, 0)
        vr_a2, vr_b2, vr_ab = vr_a2 - mu_a2, vr_b2 - mu_b2, vr_ab - mu_ab

        k1, k2, L = 0.01, 0.03, 1.
        if tensor_a.min() < -0.5 or tensor_b.min() < -0.5:  # range -1 to 1
            L = 2
        c1, c2 = (k1 + L)**2, (k2 + L)**2

        ssim = (mu_ab.mul(2).add(c1).mul(vr_ab.mul(2).add(c2)).
                div((mu_a2 + mu_b2 + c1).mul(vr_a2 + vr_b2 + c2)))
        return ssim.view(ssim.size(0), -1).mean(1)
