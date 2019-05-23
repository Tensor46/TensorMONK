""" TensorMONK :: loss :: utils """

__all__ = ["compute_n_embedding", "compute_top15", "one_hot", "one_hot_idx"]

import torch
import numpy as np


def compute_n_embedding(tensor_size):
    if isinstance(tensor_size, list) or isinstance(tensor_size, tuple):
        if len(tensor_size) > 1:  # batch size is not required
            tensor_size = np.prod(tensor_size[1:])
        else:
            tensor_size = tensor_size[0]
    return int(tensor_size)


def compute_top15(responses, targets):
    predicted = responses.topk(5, 1, True, True)[1]
    predicted = predicted.t()
    correct = predicted.eq(targets.view(1, -1).expand_as(predicted))
    top1 = correct[:1].view(-1).float().sum().mul_(100.0 / responses.size(0))
    top5 = correct[:5].view(-1).float().sum().mul_(100.0 / responses.size(0))
    return top1, top5


def one_hot(targets, n_labels):
    identity = torch.eye(n_labels).to(targets.device)
    onehot_targets = identity.index_select(dim=0,
                                           index=targets.long().view(-1))
    return onehot_targets.requires_grad_()


def one_hot_idx(targets, n_labels):
    targets = targets.view(-1)
    return targets + \
        torch.arange(0, targets.size(0)).to(targets.device) * n_labels
