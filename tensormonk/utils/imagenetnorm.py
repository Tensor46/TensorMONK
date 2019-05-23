""" TensorMONK :: utils """

import torch


class ImageNetNorm(torch.nn.Module):

    def forward(self, tensor):
        if tensor.size(1) == 1:  # convert to rgb
            tensor = torch.cat((tensor, tensor, tensor), 1)
        if tensor.min() >= 0:  # do imagenet normalization
            tensor[:, 0].add_(-0.485).div_(0.229)
            tensor[:, 1].add_(-0.456).div_(0.224)
            tensor[:, 2].add_(-0.406).div_(0.225)
        return tensor
