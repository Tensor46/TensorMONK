""" TensorMONK :: data :: transforms """

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from random import random, choices
from ..layers.dog import GaussianKernel


class Flip(nn.Module):
    r"""Does horizontal or vertical flip on a BCHW tensor.

    Args:
        horizontal (bool): If True, applies horizontal flip. Else, vertical
            flip is applied. Default = True

    ** Not recommended for CPU (Pillow/OpenCV based functions are faster).
    """
    def __init__(self, horizontal: bool = True):
        super(Flip, self).__init__()
        self.horizontal = horizontal

    def forward(self, tensor: torch.Tensor):
        return tensor.flip(3 if self.horizontal else 2)


class RandomColor(nn.Module):
    r"""Does random color channel switch when channels = 3.

    Return:
        4D BCHW torch.Tensor with size same as input tensor
    """
    def __init__(self):
        super(RandomColor, self).__init__()
        channel_shuffles = torch.Tensor([(0, 2, 1), (1, 0, 2), (1, 2, 0),
                                         (2, 0, 1), (2, 1, 0)]).long()
        self.channel_shuffles = channel_shuffles

    def forward(self, tensor: torch.Tensor):
        n, c, h, w = tensor.shape
        if c == 3:
            multiplier = torch.arange(0, n).mul(3).view(-1, 1)
            idx = self.channel_shuffles[choices(range(5), k=n)] + multiplier
            idx = idx.view(-1).to(tensor.device)
            tensor = tensor.view(-1, h, w)[idx].contiguous()
            tensor = tensor.view(n, c, h, w).contiguous()
        return tensor


class RandomBlur(nn.Module):
    r"""Does random blur with given a kernel size by varying sigma. Refer to
    GaussianKernel for details on kernel computation.

    Args:
        width (int): width of kernel, default = 5.

    Return:
        Blurred 4D BCHW torch.Tensor with size same as input tensor
    """
    def __init__(self, width: int = 5):
        super(RandomBlur, self).__init__()

        if not isinstance(width, int):
            raise TypeError("RandomBlur: width must be int: "
                            "{}".format(type(width).__name__))
        if width < 3:
            raise ValueError("RandomBlur: width must be >= 3"
                             ": {}".format(width))

        # only odd kernels
        if width % 2 == 0:
            width += 1
        # pre compute few random kernels
        self.pad = width // 2
        self.register_buffer("kernels",
                             torch.cat([GaussianKernel(x, width)
                                        for x in np.arange(1., 6., 0.1)]))
        rand_idx = torch.arange(0, self.kernels.shape[0]).view(-1, 1)
        rand_idx = rand_idx.repeat(1, 100).view(-1)
        self.rand_idx = rand_idx[torch.randperm(rand_idx.numel())]
        self.track = 0
        self.n_kernels = self.rand_idx.numel()

    def forward(self, tensor):
        n, c, h, w = tensor.shape
        if self.track + n >= self.n_kernels:
            self.track = 0
        track = self.track
        kernels = self.kernels[self.rand_idx[track]].to(tensor.device)
        self.track += 1
        if c > 1:
            kernels = kernels.repeat(c, 1, 1, 1)
        return F.conv2d(tensor, kernels, padding=[self.pad]*2, groups=c)


class RandomNoise(nn.Module):
    r"""Add Gaussian Noise to tensor.

    Args:
        mean (float): mean of gaussian distrubution, default = 0.
        std (float): std of gaussian distrubution, default = 0.01
        clamp ({float, list, tuple}, optional): when float clamps all values
            between (-clamp, +clamp). When list or tuple clamp = (min, max)
            default=None

    Return:
        4D BCHW torch.Tensor with size same as input tensor
    """
    def __init__(self, mean: float = 0., std: float = 0.01, clamp=None):
        super(RandomNoise, self).__init__()

        if not isinstance(mean, float):
            raise TypeError("RandomNoise: mean must be float: "
                            "{}".format(type(mean).__name__))
        if not isinstance(std, float):
            raise TypeError("RandomNoise: std must be float: "
                            "{}".format(type(std).__name__))
        if isinstance(clamp, float):
            clamp = (-clamp, clamp)
        if clamp is not None:
            if not (isinstance(clamp, list) or isinstance(clamp, tuple)):
                raise TypeError("RandomNoise: clamp must be float/list/tuple: "
                                "{}".format(type(clamp).__name__))
        self.mean = mean
        self.std = std
        self.clamp = clamp

    def forward(self, tensor):
        if tensor.is_cuda:
            noise = torch.cuda.FloatTensor(*tensor.shape)
        else:
            noise = torch.FloatTensor(*tensor.shape)
        noise.normal_(self.mean, self.std)
        tensor.add_(noise)
        if self.clamp is not None:
            tensor.clamp_(*self.clamp[:2])
        return tensor


class ElasticSimilarity(nn.Module):
    r"""Applies random elastic and similarity tranformation (rotation, uniform
    scale and translation) on a BCHW tensor.

    Args:
        elastic (float): Maximum elastic factor applied on a tensor. Should
            be 0 <= elastic <= 1. Default = 0.2
        angle (int/float): Random rotation range [-angle to angle] applied on a
            tensor. Should be 0 <= angle <= 180. If flip up-down is used,
            angle > 90 is not required. Default = 6.
        scale (float): Maximum zoom-in or zoom-out [1-scale, 1+scale] applied
            on a tensor. Should be 0 <= scale <= 1. If flip up-down is used,
            angle > 90 is not required. Default = 0.1. Applied scale is uniform
            in x and y direction.
        translation (float): Maximum translation applied on height
            [- translation*height, translation*height] and width
            [- translation*width, translation*width]. Should be
            0 <= translation <= 1.Default = 0.1
        horizontal_flip (bool): When True (i.e, Flip(horizontal=True)), random
            rotation range of [-angle to angle] is redundant and is limited to
            [0 to min(angle, 180)]. Default=False.
        vertical_flip (bool): When True (i.e, Flip(horizontal=False)), random
            rotation > 90 and < -90 is redundant and is limited to
            [-min(angle, 90) to min(angle, 90)]. Default=False.
        zoom_in_only (bool): When True, the range of scale is change to
            [0, 1+scale]. Default = True
        reflective_pad (bool): When True, relective padding with pad =
            min(h, w)//4 is applied on both directions of height and width
            before transformation and then center cropped to minimize zero
            padding. Default = False

    ** reflective_pad is an expensive operation.
    ** Not recommended for CPU (Pillow/OpenCV based functions are faster).

    # TODO: Add noise
    """
    def __init__(self,
                 elastic: float = 0.2,
                 angle: float = 6,
                 scale: float = 0.1,
                 translation: float = 0.1,
                 horizontal_flip: bool = False,
                 vertical_flip: bool = False,
                 zoom_in_only: bool = True,
                 reflective_pad: bool = False):
        super(ElasticSimilarity, self).__init__()
        # checks
        if not isinstance(elastic, float):
            raise TypeError("ElasticSimilarity: elastic must be float: "
                            "{}".format(type(elastic).__name__))
        if not (0. <= elastic <= 1.):
            raise ValueError("ElasticSimilarity: 0. <= elastic <= .5"
                             ": {}".format(elastic))
        if not (isinstance(angle, int) or isinstance(angle, float)):
            raise TypeError("ElasticSimilarity: angle must be int/float: "
                            "{}".format(type(angle).__name__))
        if not (0 <= angle <= 180):
            raise ValueError("ElasticSimilarity: 0 <= angle <= 180"
                             ": {}".format(angle))
        if not isinstance(scale, float):
            raise TypeError("ElasticSimilarity: scale must be float: "
                            "{}".format(type(scale).__name__))
        if not (0. <= scale <= 1.):
            raise ValueError("ElasticSimilarity: 0. <= scale <= 1."
                             ": {}".format(scale))
        if not isinstance(translation, float):
            raise TypeError("ElasticSimilarity: translation must be float: "
                            "{}".format(type(translation).__name__))
        if not (0. <= translation <= 1.):
            raise ValueError("ElasticSimilarity: 0. <= translation <= 1."
                             ": {}".format(translation))
        if not isinstance(zoom_in_only, bool):
            raise TypeError("ElasticSimilarity: zoom_in_only must be bool: "
                            "{}".format(type(zoom_in_only).__name__))
        if not isinstance(reflective_pad, bool):
            raise TypeError("ElasticSimilarity: reflective_pad must be bool: "
                            "{}".format(type(reflective_pad).__name__))

        self.reflective_pad = reflective_pad
        if elastic > 0:
            self.elastic_factor = elastic
        if translation > 0:
            # pre compute 10000 variations of translation within a given range
            self.translations = torch.FloatTensor(torch.arange(0, 1, 0.0001))
            self.translations = self.translations[torch.randperm(10000)]
            self.translations.mul_(translation*2).add_(-translation)
            self.translations = self.translations.float()
            self.track_translation = 0
            self.n_translation = 10000

        if scale > 0:
            # pre compute 6000 variations of scale within a given range
            self.scales = torch.FloatTensor(torch.arange(0, 1, 0.00016668))
            self.scales = self.scales[torch.randperm(6000)]
            if zoom_in_only:
                self.scales.mul_(scale)
            else:
                self.scales.mul_(scale*2).add_(-scale)
            self.track_scale = 0
            self.n_scale = 6000

        if angle > 0.:
            if angle < 2.:
                angle = 2
            # pre compute all possible rotation matrices
            if vertical_flip:
                angle = min(angle, 90)
            if horizontal_flip:
                radians = torch.arange(0, angle+1, 1.).mul(np.pi/180)
            else:
                radians = torch.arange(-angle, angle+1, 1.).mul(np.pi/180)

            _cos = torch.FloatTensor(radians).cos().view(-1, 1)
            _sin = torch.FloatTensor(radians).sin().view(-1, 1)
            rotate_tms = torch.cat((_cos, -_sin, _cos.mul(0),
                                    _sin, _cos, _cos.mul(0)), 1)
            n = rotate_tms.size(0)
            self.rotate_tms = rotate_tms.view(n, 2, 3)
            # pre compute ~ 2048 repetitions
            rand_rotate_idx = torch.arange(0, n).view(-1, 1)
            rand_rotate_idx = rand_rotate_idx.repeat(1, 2048//n)
            rand_rotate_idx = rand_rotate_idx.view(-1)
            _rand_idx = torch.randperm(rand_rotate_idx.numel())
            self.rand_rotate_idx = rand_rotate_idx[_rand_idx]
            self.track_angle = 0
            self.n_angle = self.rand_rotate_idx.numel()
        # identity matrix used when angle = 0
        self.identity = torch.FloatTensor([1, 0, 0, 0, 1, 0]).view(1, 2, 3)

    def forward(self, tensor: torch.Tensor):

        n, c, h, w = tensor.shape
        device = tensor.device
        if self.reflective_pad:
            # reflective padding
            pad = min(h, w) // 4
            tensor = F.pad(tensor, [pad]*4, "replicate")
            n, c, h, w = tensor.shape

        # random similarity transformations
        tms = self.random_tms(n).to(device)
        sz = torch.Size((n, c, h, w))
        # affine grid for similarity transformations
        grid = F.affine_grid(tms, sz)
        if hasattr(self, "elastic_factor"):
            # edit affine grid to do elastic transformations
            factor = self.elastic_factor * 0.1
            # row-offset
            one_180 = torch.arange(0, h, 1.).div(h-1).mul(np.pi).to(device)
            few_180s = one_180.div(random() * 0.9 + 0.1)
            few_180s = few_180s.view(1, -1) + torch.randn(n, 1).to(device)
            offset_rows = few_180s.sin().mul(factor).view(n, h, 1)
            # column-offset
            one_180 = torch.arange(0, w, 1.).div(w-1).mul(np.pi).to(device)
            few_180s = one_180.div(random() * 0.9 + 0.1)
            few_180s = few_180s.view(1, -1) * torch.randn(n, 1).to(device)
            offset_cols = few_180s.sin().mul(factor).view(n, 1, w)

            # random offsets added to the affine grid
            grid[:, :, :, 0] = grid[:, :, :, 0] + offset_cols
            grid[:, :, :, 1] = grid[:, :, :, 1] + offset_rows
        # apply transformations
        tensor = F.grid_sample(tensor, grid)
        if self.reflective_pad:
            # recrop if self.reflective_pad
            tensor = tensor[:, :, pad:-pad, pad:-pad]
        return tensor

    def random_tms(self, n):
        if hasattr(self, "track_angle"):
            # get n random similarity matrices
            if self.track_angle + n >= self.n_angle:
                self.track_angle = 0
            track = self.track_angle
            tms = self.rotate_tms[self.rand_rotate_idx[track:track+n]]
            self.track_angle += n
        else:
            # get n idenity matrices
            tms = self.identity.expand(n, 2, 3)

        if hasattr(self, "track_scale"):
            # add n random scales
            if self.track_scale + n >= self.n_scale:
                self.track_scale = 0
                device = self.scales.device
                tmp = torch.randperm(self.n_scale).to(device)
                self.scales = self.scales[tmp]
            track = self.track_scale
            scales = self.scales[track:track+n]
            self.track_scale += n
            tms = tms * (1 - scales.view(-1, 1, 1))

        if hasattr(self, "track_translation"):
            # add n random translations
            if (self.track_translation + 2*n) >= self.n_translation:
                self.track_translation = 0
                device = self.translations.device
                tmp = torch.randperm(self.n_translation).to(device)
                self.translations = self.translations[tmp]
            track = self.track_translation
            translations = self.translations[track:track + 2*n]
            self.track_translation += n
            tms[:, :, 2] = tms[:, :, 2] + translations.view(-1, 2)
        return tms


class RandomTransforms(nn.Module):
    r"""Apply transformation functions on a batch with a given probability.
    Scalable to CPU & GPU. Transformations are applied with torch.no_grad().

    Args:
        functions (required, list/tuple): a list of functions (nn.Modules).
        probabilities (required, list/tuple): a list of each function
            probability. Length must be equal to length of functions.

    ** Not recommended for CPU (Pillow/OpenCV based functions are faster).
    """
    def __init__(self, functions: list, probabilities: list):
        super(RandomTransforms, self).__init__()

        if not type(functions) in [list, tuple]:
            raise TypeError("RandomTransformations: functions must be list"
                            "/tuple: {}".format(type(functions).__name__))
        if any([not isinstance(x, nn.Module) for x in functions]):
            raise TypeError("RandomTransformations: functions must be a "
                            "list/tuple of nn.Modules")
        if not type(probabilities) in [list, tuple]:
            raise TypeError("RandomTransformations: probabilities must be list"
                            "/tuple: {}".format(type(probabilities).__name__))
        assert len(functions) == len(probabilities), \
            "RandomTransformations: len(functions) != len(probabilities)"

        self.functions = nn.ModuleList(functions)
        self.probabilities = probabilities

    def forward(self, tensor: torch.Tensor):
        n = tensor.shape[0]
        device = tensor.device
        with torch.no_grad():
            for fn, prob in zip(self.functions, self.probabilities):
                random_idx = torch.rand(n).to(device).le(prob)
                if random_idx.sum() > 0:
                    random_idx = random_idx.nonzero().view(-1)
                    tensor[random_idx] = fn(tensor[random_idx])
        return tensor


# from PIL import Image as ImPIL
# from torchvision.transforms import ToPILImage
# from tensormonk.layers.dog import GaussianKernel
# toimage = ToPILImage()
# test = RandomTransforms(
#     (ElasticSimilarity(), RandomBlur()), (0.9, 0.9))
# test = RandomTransforms(
#     (RandomNoise(0., 0.01, (0, 1)), ), (0.9, ))
# img = ImPIL.open("../data/test.png")
# npim = np.array(img, np.float32).transpose(2, 0, 1) / 255.
# toimage(test(torch.from_numpy(npim.copy()).unsqueeze(0))[0])
# %timeit test(torch.from_numpy(npim.copy()).unsqueeze(0)).shape
