""" TensorMONK :: utils """

import torch
from six import add_metaclass


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
        tensor_a.size(1) == tensor_b.size(1), \
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


def euclidean(tensor_a: torch.Tensor, tensor_b: torch.Tensor):
    assert tensor_a.dim() == 2 and tensor_b.dim() == 2, \
        "euclidean :: tensor_a and tensor_b must be 2D"
    assert tensor_a.size(1) == tensor_b.size(1), \
        "euclidean :: tensor_a and tensor_b must have same shape"
    tensor_a = tensor_a.unsqueeze(1)
    tensor_b = tensor_b.unsqueeze(0)
    return (tensor_a - tensor_b).pow(2).sum(2).pow(0.5)


def sq_euclidean(tensor_a: torch.Tensor, tensor_b: torch.Tensor):
    assert tensor_a.dim() == 2 and tensor_b.dim() == 2, \
        "sq_euclidean :: tensor_a and tensor_b must be 2D"
    assert tensor_a.size(1) == tensor_b.size(1), \
        "sq_euclidean :: tensor_a and tensor_b must have same shape"
    tensor_a = tensor_a.unsqueeze(1)
    tensor_b = tensor_b.unsqueeze(0)
    return (tensor_a - tensor_b).pow(2).sum(2)


def cosine(tensor_a: torch.Tensor, tensor_b: torch.Tensor):
    assert tensor_a.dim() == 2 and tensor_b.dim() == 2, \
        "cosine :: tensor_a and tensor_b must be 2D"
    assert tensor_a.size(1) == tensor_b.size(1), \
        "cosine :: tensor_a and tensor_b must have same shape"
    tensor_a = torch.nn.functional.normalize(tensor_a, 2, 1)
    tensor_b = torch.nn.functional.normalize(tensor_b, 2, 1)
    return tensor_a.mm(tensor_b.t())


class MeasuresMeta(type):
    def __init__(self, name, bases, dct):
        self.correlation = corr_1d
        self.xcorrelation = xcorr_1d
        self.cosine = cosine
        self.euclidean = euclidean
        self.sq_euclidean = sq_euclidean
        super(MeasuresMeta, self).__init__(name, bases, dct)


@add_metaclass(MeasuresMeta)
class Measures(object):
    pass
