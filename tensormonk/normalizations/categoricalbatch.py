""" TensorMONK :: layers :: CategoricalBNorm """

__all__ = ["CategoricalBNorm", ]

import torch
import torch.nn as nn


class CategoricalBNorm(nn.Module):
    r""" Categorical BatchNorm2d (done using targets or latents)

    Args:
        tensor_size: shape of tensor in BCHW
            (None/any integer >0, channels, height, width)
        n_labels (int): number of labels
        n_latent (int): length of latent (a special case when the latent
            conditional)

    Return:
        torch.Tensor
    """

    def __init__(self, tensor_size, n_labels: int, n_latent: int = None,
                 **kwargs):
        super(CategoricalBNorm, self).__init__()

        self.embedding = nn.Parameter(
            torch.randn(n_labels if n_latent is None else n_latent,
                        tensor_size[1]))
        self.embedding.data.normal_(0, 0.02)
        self.normalization = nn.BatchNorm2d(tensor_size[1], affine=False)

        self.n_labels = n_labels
        self.n_latent = n_latent
        self.tensor_size = tensor_size

    def forward(self, tensor, targets_or_latents):
        tensor = self.normalization(tensor)
        if self.n_latent is None:
            # targets_or_latents is targets
            # extract respective categorical embedding
            embedding = self.embedding[targets_or_latents.long()]
        else:
            # targets_or_latents is latents
            # convert latent to embedding of tensor.size(1)
            embedding = targets_or_latents @ self.embedding
        return tensor * embedding.view(-1, tensor.size(1), 1, 1)

    def __repr__(self):
        return "CategoricalBNorm"


# tensor_size = (1, 6, 4, 4)
# n_labels = 46
# test = CategoricalBNorm(tensor_size, n_labels)
# test(torch.randn(*tensor_size), torch.Tensor([6])).shape
# test = CategoricalBNorm(tensor_size, n_labels, n_latent=16)
# test(torch.randn(*tensor_size), torch.randn(1, 16)).shape
