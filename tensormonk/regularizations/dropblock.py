""" TensorMONK :: layers :: DropBlock """

__all__ = ["DropBlock", ]

import torch
from torch import nn
import torch.nn.functional as F


class DropBlock(nn.Module):
    r"""Randomly sets block_size x block_size of input tensor to zero with a
    probability of p. Implemented - https://arxiv.org/abs/1810.12890

    Args:
        tensor_size: shape of tensor in BCHW
            (None/any integer >0, channels, height, width)
        p: dropout probability, default = 0.1
        block_size: width of the block, default = 5. Paper default = 7
        shared: When True, creates a mask with 1 channel and scales across the
            tensor channels. default = False - recommended in paper
        iterative_p: when True, iteratively increases probability from 0 to p
            till n_iterations = steps_to_max, and maintains p there after.
            default = True
        steps_to_max: iterations to reach p, default = 20000

     Return:
         torch.Tensor of shape BCHW
    """

    def __init__(self,
                 tensor_size,
                 p: float = 0.1,
                 block_size: int = 5,
                 shared: bool = False,
                 iterative_p: bool = True,
                 steps_to_max: int = 20000):

        super(DropBlock, self).__init__()
        # checks
        if not type(p) is float:
            raise TypeError("DropBlock: p must float: "
                            "{}".format(type(p).__name__))
        if not 0. <= p < 1:
            raise ValueError("DropBlock: p must be >=0 and <1: "
                             "{}".format(p))
        self.p = p

        if not type(block_size) is int:
            raise TypeError("DropBlock: block_size must int: "
                            "{}".format(type(block_size).__name__))
        if not (tensor_size[2] >= block_size and
                tensor_size[3] >= block_size):
            raise ValueError("DropBlock: tensor_size[2:3] must be greater "
                             "than block_size: "
                             "{}/{}".format(tensor_size, block_size))
        if not block_size >= 2:
            raise ValueError("DropBlock: block_size must be >= 2: "
                             "{}".format(block_size))
        if block_size % 2 == 0:
            block_size += 1
            print("DropBlock: block_size adjusted to odd: "+str(block_size))
        self.w = block_size

        if not type(shared) is bool:
            raise TypeError("DropBlock: shared must boolean: "
                            "{}".format(type(shared).__name__))
        self.shared = shared

        if not type(iterative_p) is bool:
            raise TypeError("DropBlock: iterative_p must boolean: "
                            "{}".format(type(iterative_p).__name__))
        if iterative_p:
            # steps_to_max = steps to reach p
            self.steps_to_max = steps_to_max
            self.register_buffer("n_iterations", torch.Tensor([0]).sum())

    def forward(self, tensor):
        if self.p == 0. or not self.training:
            return tensor
        n, c, h, w = tensor.shape

        if hasattr(self, "steps_to_max"):  # incremental probability
            p = min(self.p, self.p * self.n_iterations /
                    self.steps_to_max)
            self.n_iterations += 1
        else:  # constant probability = (1 - keep_prob)
            p = self.p
        # equation 1
        gamma = (p / self.w**2) * (h*w / (h-self.w+1) / (w-self.w+1))
        pad = self.w//2
        if self.shared:
            c = 1

        mask = torch.ones(n, c, h-2*pad, w-2*pad).to(tensor.device)
        mask = torch.bernoulli(mask * gamma)
        mask = F.pad(mask, (pad, pad, pad, pad))
        block_mask = F.max_pool2d(mask, self.w, 1, pad)
        block_mask = (block_mask == 0).float().detach()

        # norm = count(M)/count_ones(M)
        norm = block_mask.sum(2, True).sum(3, True) / h / w
        return tensor * block_mask * norm  # A × count(M)/count_ones(M)


# test = DropBlock((1, 3, 10, 10), 0.2, 5, True, False)
# test(torch.randn((1, 3, 10, 10)))
