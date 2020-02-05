""" TensorMONK :: layers :: Lucas-Kanade """

__all__ = ["LucasKanade", ]

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LucasKanade(nn.Module):
    r"""Lucas-Kanade tracking (based on `"Supervision-by-Registration: An
    Unsupervised Approach to Improve the Precision of Facial Landmark
    Detectors" <https://arxiv.org/pdf/1807.00966>`_).

    A cleaner version based on original repo with some corrections
    (yx must be xy) and speed improvements.

    Args:
        n_steps (int, optional): n correction steps (default: :obj:`64`).
        width (int, optional): Width of patches (default: :obj:`15`).
        sigma (float, optional): Sigma for gaussian kernel
            (default: :obj:`None`).

    :rtype: :class:`torch.Tensor`
    """

    def __init__(self, n_steps: int = 64, width: int = 15, sigma: int = None):
        super(LucasKanade, self).__init__()
        self.n_steps = n_steps
        self.gaussian(width, sigma)
        self.register_buffer(
            "sobel_x", torch.Tensor([[-1./8, 0, 1./8], [-2./8, 0, 2./8],
                                     [-1./8, 0, 1./8]]).view(1, 1, 3, 3))
        self.register_buffer(
            "sobel_y", torch.Tensor([[-1./8, -2./8, -1./8], [0, 0, 0],
                                     [1./8, 2./8, 1./8]]).view(1, 1, 3, 3))

    def gaussian(self, width: int, sigma: float = None):
        if width % 2 == 0:
            width += 1
        self.w = width
        if sigma is None or sigma == 0:
            sigma = width / 2.
        half = width // 2
        x, y = np.meshgrid(np.linspace(-half, half, width),
                           np.linspace(-half, half, width), indexing='xy')
        w = np.exp(- (x**2 + y**2) / (2.*(sigma**2)))
        w[0, :] = w[:, 0] = w[-1, :] = w[:, -1] = 0
        w = torch.from_numpy(w.astype(np.float32))
        self.register_buffer("gkernel", w.view(1, 1, width, width).clone())

    def forward(self, frame_t0: torch.Tensor, frame_t1: torch.Tensor,
                points_xy: torch.Tensor):
        r"""Tracks points_xy on frame_t0 to frame_t1.

        Args:
            frame_t0 (torch.Tensor): 4D tensor of shape BCHW.
            frame_t1 (torch.Tensor): 4D tensor of shape BCHW.
            points_xy (torch.Tensor): 3D tensor of shape B x n_points x 2.

        :rtype: :class:`torch.Tensor`
        """
        assert frame_t0.ndim == 4 and frame_t1.ndim == 4
        assert frame_t0.shape == frame_t1.shape
        assert points_xy.ndim == 3
        assert frame_t0.size(0) == frame_t1.size(0) == points_xy.size(0) == 1
        n, c, h, w = frame_t0.shape
        n_pts = points_xy.shape[1]
        # extract patches
        patches_t0 = self.extract_patches(frame_t0, points_xy)
        # extract gradients
        gx = F.conv2d(patches_t0, self.sobel_x.expand(c, 1, 3, 3), None,
                      padding=1, groups=c)
        gy = F.conv2d(patches_t0, self.sobel_y.expand(c, 1, 3, 3), None,
                      padding=1, groups=c)
        J = torch.stack((gx, gy), 1)
        weightedJ = J * self.gkernel.unsqueeze(0)
        # Hessian matrix
        H = torch.bmm(weightedJ.view(n_pts, 2, -1),
                      J.view(n_pts, 2, -1).transpose(2, 1))
        a, b, c, d = H[..., 0, 0], H[..., 0, 1], H[..., 1, 0], H[..., 1, 1]
        eps = np.finfo(float).eps
        a = a + eps
        d = d + eps
        inverseH = (torch.stack((d, -b, -c, a), 1) /
                    (a * d - b * c + eps).unsqueeze(1)).view(-1, 2, 2)
        # recurssive correction
        for _ in range(self.n_steps):
            # extract patches
            patches_t1 = self.extract_patches(frame_t1, points_xy)
            r = patches_t1 - patches_t0
            sigma = torch.bmm(weightedJ.view(n_pts, 2, -1),
                              r.view(n_pts, -1, 1))
            deltap = torch.bmm(inverseH, sigma).squeeze(-1)
            points_xy = points_xy - deltap
            if deltap.data.abs().lt(1e-4).all():
                # early stopping when delta's are minimal
                break
        return points_xy

    def extract_patches(self, tensor: torch.Tensor, points_xy: torch.Tensor):
        n, c, h, w = tensor.shape
        n_pts = points_xy.shape[1]
        half = self.w // 2

        bbox = torch.cat([points_xy - half, points_xy + half], -1)
        bbox[..., 0::2] = -1. + 2. * bbox[..., 0::2] / (w - 1)
        bbox[..., 1::2] = -1. + 2. * bbox[..., 1::2] / (h - 1)

        theta = torch.stack([(bbox[..., 2] - bbox[..., 0]) / 2,
                             bbox[..., 0] * 0,
                             (bbox[..., 2] + bbox[..., 0]) / 2,
                             bbox[..., 0] * 0,
                             (bbox[..., 3] - bbox[..., 1]) / 2,
                             (bbox[..., 3] + bbox[..., 1]) / 2], -1)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, torch.Size((n_pts, 1, self.w, self.w)),
                             align_corners=True)
        patches = F.grid_sample(tensor.expand(n_pts, c, h, w), grid,
                                align_corners=True)
        return patches
