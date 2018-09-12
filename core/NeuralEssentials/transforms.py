""" TensorMONK's :: NeuralEssentials                                         """

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# ============================================================================ #


class Transforms(nn.Module):
    """
        Parameters
        ----------
        p_transformations = 0-1, probability of applying transformations on the
                            batch
        v_crop            = 0-1, does a crop within +/- percentage of image's
                            height or width (values must be between 0-1)
                            Aspect ratio is not maintained
        v_affine          = 0-1, does affine transformation with identity matrix
                            varied by +/- v_affine (values must be between 0-1)
        p_blur            = 0-1, probability of applying blur. A single blur
                            kernel is applied per batch.
        sigma_blur        = 0.1-3, max sigma values of 5x5 blur kernel
                            other size can be possible but this is a faster way
        p_contrast        = 0-1, probability of applying contrast variation
        v_contrast        = 0.1 - 2, contrast range is between 1 +/- 1, least is
                            set to .4. Not conventional approach, but does the job
        p_flipud          = 0-1, probability of vertical flip
        p_fliplr          = 0-1, probability of horizontal flip
        p_channelshuffle  = 0-1, probability of channel shuffle (only for RGB)
                            only, one kind of shuffle is applied per batch
        less_pad          = True/False, when True random crop and affine will
                            have minimal padding

        *Can be scaled to mutli gpu's. torchvision.transforms are better for
        small images, small batches and cpu based training
    """
    def __init__(self,
                 p_transformations = 0.8,
                 v_crop            = 0.1,
                 v_affine          = 0.1,
                 p_blur            = 0.4,
                 sigma_blur        = 1.5,
                 p_contrast        = 0.2,
                 v_contrast        = 1.,
                 p_flipud          = 0.,
                 p_fliplr          = 0.,
                 p_channelshuffle  = 0.1,
                 less_pad          = True,
                 ):
        super(Transforms, self).__init__()

        # torch.rand or random.rand are expensive, so, random numbers are
        # generated only once and reused
        self.random01 = torch.FloatTensor(torch.arange(0, 1, 0.0001))
        self.random01 = self.random01[torch.randperm(self.random01.numel())]
        self.track = 0

        assert p_transformations >= 0 and p_transformations < 1, """Transforms ::
            p_transformations 0-1, given {}""".format(p_transformations)
        self.p_tfms = p_transformations
        # all p_transformations will have either a random crop or affine
        self.v_crop = v_crop
        self.v_affn = v_affine
        self.less_pad = less_pad
        self.p_contrast = p_contrast
        self.v_contrast = v_contrast
        self.p_blur = p_blur
        self.sigma_blur = sigma_blur
        # 5x5 (x^2 + y^2) for gaussian blur, as 5x5 convolutions are fast
        self._5x5 = torch.arange(5).sub(5//2).view(1, -1).expand(5, 5).type(torch.float32).pow(2) + \
            torch.arange(5).sub(5//2).view(1, -1).expand(5, 5).t().type(torch.float32).pow(2)
        self.p_flipud = p_flipud
        self.p_fliplr = p_fliplr
        self.p_channelshuffle = p_channelshuffle
        self.channelshuffles = [(0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]

    def forward(self, tensor):
        with torch.no_grad():
            n, c, h, w = tensor.size()
            device = tensor.device
            do_tfms = self.n_random((n,)) < self.p_tfms

            # half random crops and half random affine
            if do_tfms.sum() > 0:
                tms = torch.zeros(do_tfms.sum(), 2, 3)
                do_crop = self.n_random((do_tfms.sum(),)) < 0.5
                if do_crop.sum() > 0:
                    tms[do_crop,] = self.random_tms(do_crop.sum())
                do_affn = do_crop == 0
                if do_affn.sum() > 0:
                    tms[do_affn,] = self.random_tms(do_affn.sum(), True)
                do_tfms, tms = do_tfms.to(device), tms.to(device)
                tensor[do_tfms,] = F.grid_sample(tensor[do_tfms,],
                    F.affine_grid(tms, torch.Size((do_tfms.sum(), c, h, w))))
            # random contrast
            if self.p_contrast > 0 and do_tfms.sum() > 0:
                do_contrast = self.n_random((do_tfms.sum(),)) < self.p_contrast
                idx = do_tfms.nonzero()[do_contrast.nonzero()].view(-1).to(device)
                cvalues = self.n_random((do_contrast.sum(),)).to(device)
                cvalues.mul_(self.v_contrast*2).add_(1-self.v_contrast).clamp_(0.4)
                tmp = tensor[idx,]
                tmp.sub_(0.5).mul_(cvalues.view(-1, 1, 1, 1)).add_(0.5).clamp_(0, 1)
                tensor[idx,] = tmp
            # random blur
            if self.p_blur > 0 and do_tfms.sum() > 0:
                idx = do_tfms.nonzero()[self.n_random((do_tfms.sum(),)) < self.p_blur]
                idx = idx.view(-1).to(device)
                if idx.numel() > 0:
                    tensor[idx,] = F.conv2d(F.pad(tensor[idx,], (2, 2, 2, 2), mode="replicate"),
                        self.random_blur(1).expand(c, 1, 5, 5).to(device), groups=c)
            # random flip up-down
            if self.p_flipud > 0 and do_tfms.sum() > 0:
                idx = do_tfms.nonzero()[self.n_random((do_tfms.sum(),)) < self.p_flipud]
                if idx.numel() > 0:
                    idx = idx.to(device)
                    tensor[idx.view(-1),] = tensor[idx.view(-1),].flip(2)
            # random flip left-right
            if self.p_fliplr > 0 and do_tfms.sum() > 0:
                idx = do_tfms.nonzero()[self.n_random((do_tfms.sum(),)) < self.p_fliplr]
                if idx.numel() > 0:
                    idx = idx.to(device)
                    tensor[idx.view(-1),] = tensor[idx.view(-1),].flip(3)
            # random channel shuffle
            if self.p_channelshuffle > 0 and c == 3:
                idx = do_tfms.nonzero()[self.n_random((do_tfms.sum(),)) < self.p_channelshuffle]
                if idx.numel() > 0:
                    channelshuffle = self.channelshuffles.pop(0)
                    self.channelshuffles.append(channelshuffle)
                    idx = idx.to(device)
                    tensor[idx.view(-1),] = tensor[idx.view(-1),][:, channelshuffle]

        return tensor

    def random_blur(self, n):
        kernel = self._5x5.view(1, 1, 5, 5).expand(n, 1, 5, 5)
        sigmas = self.n_random((n,)).mul_(self.sigma_blur).clamp_(0.3)
        sigmas = sigmas.view(n, 1, 1, 1).contiguous()
        kernel = torch.exp(-kernel.div(2*sigmas)).div(2.*sigmas*sigmas*22./7)
        return kernel.div_(kernel.sum(2, True).sum(3, True))

    def random_tms(self, n, affine=False, type=torch.FloatStorage):
        tms = torch.FloatTensor([1, 0, 0, 0, 1, 0]).view(1, 2, 3).expand(n, 2, 3)
        if affine:
            tms = tms.add(self.n_random((n, 2, 3), self.v_affn))
        else: # for random crops - scale and translation
            tms = torch.cat((tms[:, : , :2], self.n_random((n, 2, 1), self.v_crop)), 2)
            scale = self.n_random((n,), self.v_crop)
            tms[:, 0, 0].add_(scale)
            tms[:, 1, 1].add_(scale)
        return tms

    def n_random(self, size, scale=None):
        n = np.prod(size)
        if self.track + n > self.random01.numel():
            self.track = 0
        track = self.track
        self.track += n
        if scale is not None:
            s = 1 if self.less_pad else 2
            return self.random01[track:track+n].view(*size).mul(scale*s).sub(scale)
        return self.random01[track:track+n]

# test = Transforms()
# from PIL import Image as ImPIL
# import torchvision.utils as tutils
# imgs = []
# for _ in range(16):
#     img = ImPIL.open("../data/test.jpeg")
#     npim = (np.array(img, np.float32).transpose(2, 0, 1)/255.)[np.newaxis, ]
#     imgs.append(torch.from_numpy(npim))
# tutils.save_image(test(torch.cat(imgs, 0)), "../test.jpeg")
