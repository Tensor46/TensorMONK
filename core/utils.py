""" TensorMONK's :: utils                                                    """

__all__ = ["utils"]

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.interpolate as interp


def roc(genuine_or_scorematrix,
        impostor_or_labels,
        filename       = None,
        semilog        = True,
        lower_triangle = True,
        print_show     = False):
    """

        genuine_or_scorematrix -- genuine scores or all scores (square matrix)
            when genuine scores, impostor_or_labels must be impostor scores
            when all scores, impostor_or_labels must be labels
            accepted types - list/tuple/numpy.ndarray/torch.Tensor
        impostor_or_labels --  impostor scores or labels
            when impostor scores, genuine_or_scorematrix must be genuine scores
            when labels, genuine_or_scorematrix must be all scores
            accepted types - list/tuple/numpy.ndarray/torch.Tensor
                             list & tuple can have strings
        filename:str -- fullpath of image to save
        semilog:bool -- When True plots the roc on semilog
        lower_triangle -- To avoid duplicates in score matrix
    """
    # convert to numpy
    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy()
        elif isinstance(x, np.ndarray):
            return x
        elif isinstance(x, list) or isinstance(x, tuple):
            assert type(x[0]) in (int, float, str), \
                "list/tuple of int/float/str are accepted, given {}".format(type(x[0]))
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
                indices = il.reshape((-1,1))
                indices = np.concatenate([indices]*indices.shape[0], 1)
                indices = (indices == indices.T).astype(np.int) + 1
                indices = np.tril(indices,-1).flatten()
                genuine = gs.flatten()[indices == 2]
                impostor = gs.flatten()[indices == 1]
            else:
                indices = np.expand_dims(il, 1) == np.expand_dims(il, 0)
                genuine = gs.flatten()[indices.flatten()]
                impostor = gs.flatten()[indices.flatten() == False]
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
    if genuine.mean() < impostor.mean(): # distance bins to similarity bins
        genuine_bin_count = genuine_bin_count[::-1]
        impostor_bin_count = impostor_bin_count[::-1]
    # compute frr & grr, then far = 100 - grr & gar = 100 - frr
    gar = 1 - (1. * np.cumsum(genuine_bin_count))
    far = 1 - (1. * np.cumsum(impostor_bin_count))
    # Find gars on log scale -- 0.00001 - 1
    samples = [gar[np.argmin(np.abs(far - 10**x))] for x in range(-5, 1)]
    if print_show:
        print(("gar@far (0.00001-1.) :: "+"/".join(["{:1.3f}"]*6)).format(*samples))
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
        # need some work seaborn vs matplot?

    return {"gar": gar, "far": far, "auc": abs(np.trapz(gar, far)),
        "gar_samples": samples}


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
    roc = roc
