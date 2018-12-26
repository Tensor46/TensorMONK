""" TensorMONK's :: utils                                                   """

__all__ = ["utils"]

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.interpolate as interp
import matplotlib.pyplot as plt


def roc(genuine_or_scorematrix, impostor_or_labels, filename=None,
        print_show=False, semilog=True, lower_triangle=True):
    r"""Computes receiver under operating curve for a given combination of
    (genuine and impostor) or (score matrix and labels).

    Args:
        genuine_or_scorematrix: genuine scores or all scores (square matrix) in
            list/tuple/numpy.ndarray/torch.Tensor
        impostor_or_labels: impostor scores or labels in
            list/tuple/numpy.ndarray/torch.Tensor
            list/tuple of strings for labels is accepted
        filename: fullpath of image to save
        print_show: True = prints gars at fars and shows the roc
        semilog: True = plots the roc on semilog
        lower_triangle: True = avoids duplicates in score matrix

    Return:
        A dictionary with gar and their corresponding far, auc, and
        gar_samples.
            gar - genuine accept rates with a range 0 to 1
            far - false accept rates with a range 0 to 1
            auc - area under curve
            gar_samples - gar's at far = 0.00001, 0.0001, 0.001, 0.01, 0.01, 1.
    """
    # convert to numpy
    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy()
        elif isinstance(x, np.ndarray):
            return x
        elif isinstance(x, list) or isinstance(x, tuple):
            assert type(x[0]) in (int, float, str), \
                ("list/tuple of int/float/str are accepted," +
                 " given {}").format(type(x[0]))
            if isinstance(x[0], str):
                classes = sorted(list(set(x)))
                x = [classes.index(y) for y in x]
            return np.array(x)
        else:
            raise NotImplementedError
    gs = to_numpy(genuine_or_scorematrix)
    il = to_numpy(impostor_or_labels)

    # get genuine and impostor scores if score matrix and labels are provided
    if gs.ndim == 2:
        if gs.shape[0] == gs.shape[1] and gs.shape[0] == il.size:
            # genuine_or_scorematrix is a score matrix
            if lower_triangle:
                indices = il.reshape((-1, 1))
                indices = np.concatenate([indices]*indices.shape[0], 1)
                indices = (indices == indices.T).astype(np.int) + 1
                indices = np.tril(indices, -1).flatten()
                genuine = gs.flatten()[indices == 2]
                impostor = gs.flatten()[indices == 1]
            else:
                indices = np.expand_dims(il, 1) == np.expand_dims(il, 0)
                genuine = gs.flatten()[indices.flatten()]
                indices = np.expand_dims(il, 1) != np.expand_dims(il, 0)
                impostor = gs.flatten()[indices.flatten()]
    if "genuine" not in locals():
        # genuine_or_scorematrix is an array of genuine scores
        genuine = gs.flatten()
        impostor = il.flatten()

    # convert to float32
    genuine, impostor = genuine.astype(np.float32), impostor.astype(np.float32)
    # min and max
    min_score = min(genuine.min(), impostor.min())
    max_score = max(genuine.max(), impostor.max())
    # find histogram bins and then count
    bins = np.arange(min_score, max_score, (max_score-min_score)/4646)
    genuine_bin_count = np.histogram(genuine, density=False, bins=bins)[0]
    impostor_bin_count = np.histogram(impostor, density=False, bins=bins)[0]
    genuine_bin_count = genuine_bin_count.astype(np.float32) / genuine.size
    impostor_bin_count = impostor_bin_count.astype(np.float32) / impostor.size
    if genuine.mean() < impostor.mean():  # distance bins to similarity bins
        genuine_bin_count = genuine_bin_count[::-1]
        impostor_bin_count = impostor_bin_count[::-1]
    # compute frr & grr, then far = 100 - grr & gar = 100 - frr
    gar = 1 - (1. * np.cumsum(genuine_bin_count))
    far = 1 - (1. * np.cumsum(impostor_bin_count))
    # Find gars on log scale -- 0.00001 - 1
    samples = [gar[np.argmin(np.abs(far - 10**x))] for x in range(-5, 1)]
    if print_show:
        print(("gar@far (0.00001-1.) :: " +
              "/".join(["{:1.3f}"]*6)).format(*samples))
    # interpolate and shirnk gar & far to 600 samples, for ploting
    _gar = interp.interp1d(np.arange(gar.size), gar)
    gar = _gar(np.linspace(0, gar.size-1, 599))
    _far = interp.interp1d(np.arange(far.size), far)
    far = _far(np.linspace(0, far.size-1, 599))

    gar = np.concatenate((np.array([1.]), gar), axis=0)
    far = np.concatenate((np.array([1.]), far), axis=0)

    if filename is not None:
        if not filename.endswith((".png", ".jpeg", "jpg")):
            filename += ".png"
        # TODO seaborn ?
        plt.semilogx(far, gar)
        plt.xlabel("far")
        plt.ylabel("gar")
        plt.ylim((-0.01, 1.01))
        plt.savefig(filename, dpi=300)
        if print_show:
            plt.show()

    return {"gar": gar, "far": far, "auc": abs(np.trapz(gar, far)),
            "gar_samples": samples}


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
        Blurred 4D BCHW torch.Tensor with size same as input tensor
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


def corr_1d(tensor_a: torch.Tensor, tensor_b: torch.Tensor):
    r"""Computes row wise correlation between two 2D torch.Tensor's of same
    shape. eps is added to the dinominator for numerical stability.

    Input:
        tensor_a: 2D torch.Tensor of size MxN
        tensor_b: 2D torch.Tensor of size MxN

    Return:
        A vector of length M and type torch.Tensor
    """
    assert tensor_a.dim() == 2 and tensor_b.dim() == 2, \
        "corr_1d :: tensor_a and tensor_b must be 2D"
    assert tensor_a.size(0) == tensor_b.size(0) and \
        tensor_a.dim(1) == tensor_b.dim(1), \
        "corr_1d :: tensor_a and tensor_b must have same shape"

    num = tensor_a.mul(tensor_b).mean(1) - tensor_a.mean(1)*tensor_b.mean(1)
    den = ((tensor_a.pow(2).mean(1) - tensor_a.mean(1).pow(2)).pow(0.5) *
           (tensor_b.pow(2).mean(1) - tensor_b.mean(1).pow(2)).pow(0.5))
    return num / den.add(1e-8)


def xcorr_1d(tensor: torch.Tensor):
    r"""Computes cross correlation of 2D torch.Tensor's of shape MxN, i.e,
    M vectors of lenght N. eps is added to the dinominator for numerical
    stability.

    Input:
        tensor: 2D torch.Tensor of size MxN

    Return:
        MxM torch.Tensor
    """
    assert tensor.dim() == 2, "xcorr_1d :: tensor must be 2D"

    n = tensor.size(0)
    num = (tensor.view(n, 1, -1).mul(tensor.view(1, n, -1)).mean(2) -
           tensor.view(n, 1, -1).mean(2).mul(tensor.view(1, n, -1).mean(2)))
    den = ((tensor.view(n, 1, -1).pow(2).mean(2) -
            tensor.view(n, 1, -1).mean(2).pow(2)).pow(0.5) *
           (tensor.view(1, n, -1).pow(2).mean(2) -
            tensor.view(1, n, -1).mean(2).pow(2)).pow(0.5))
    return num / den.add(1e-8)


class ImageNetNorm(nn.Module):

    def forward(self, tensor):
        if tensor.size(1) == 1:  # convert to rgb
            tensor = torch.cat((tensor, tensor, tensor), 1)
        if tensor.min() >= 0:  # do imagenet normalization
            tensor[:, 0].add_(-0.485).div_(0.229)
            tensor[:, 1].add_(-0.456).div_(0.224)
            tensor[:, 2].add_(-0.406).div_(0.225)
        return tensor


class utils:
    corr_1d = corr_1d
    xcorr_1d = xcorr_1d
    DoH = DoH
    HessianBlob = HessianBlob
    GaussianKernel = GaussianKernel
    DoG = DoG
    DoGBlob = DoGBlob
    roc = roc
    ImageNetNorm = ImageNetNorm
