""" TensorMONK :: layers :: PixelWise """

__all__ = ["PixelWise", ]

import torch


class PixelWise(torch.nn.Module):
    r""" Implemented - https://arxiv.org/pdf/1710.10196.pdf """
    def __init__(self, eps=1e-8):
        super(PixelWise, self).__init__()
        self.eps = eps

    def forward(self, tensor):
        return tensor.div(tensor.pow(2).mean(1, True).pow(.5) + self.eps)

    def __repr__(self):
        return "pixelwise"
