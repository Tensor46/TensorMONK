""" TensorMONK :: loss :: utils """

__all__ = ["compute_n_embedding", "compute_top15", "one_hot", "one_hot_idx",
           "hard_negative_mask"]

import torch
import numpy as np


def compute_n_embedding(tensor_size: tuple):
    if isinstance(tensor_size, list) or isinstance(tensor_size, tuple):
        if len(tensor_size) > 1:  # batch size is not required
            tensor_size = np.prod(tensor_size[1:])
        else:
            tensor_size = tensor_size[0]
    return int(tensor_size)


@torch.no_grad()
def compute_top15(responses: torch.Tensor, targets: torch.Tensor):
    predicted = responses.topk(5, 1, True, True)[1]
    predicted = predicted.t()
    correct = predicted.eq(targets.view(1, -1).expand_as(predicted))
    top1 = correct[:1].view(-1).float().sum().mul_(100.0 / responses.size(0))
    top5 = correct[:5].view(-1).float().sum().mul_(100.0 / responses.size(0))
    return top1, top5


@torch.no_grad()
def one_hot(targets: torch.Tensor, n_labels: int):
    identity = torch.eye(n_labels, dtype=torch.int8).to(targets.device)
    onehot_targets = identity.index_select(dim=0,
                                           index=targets.long().view(-1))
    return onehot_targets.requires_grad_(False)


@torch.no_grad()
def one_hot_idx(targets: torch.Tensor, n_labels: int):
    targets = targets.view(-1)
    return targets + \
        torch.arange(0, targets.size(0)).to(targets.device) * n_labels


@torch.no_grad()
def hard_negative_mask(prediction: torch.Tensor,
                       targets: torch.Tensor,
                       pos_to_neg_ratio: float = 0.25):
    r""" Hard negative mask generator for object detection (includes both
    positives and hard negatives).

    Args:
        prediction (torch.Tensor): label predictions of the network must be
            2D/3D tensor.
            2D: Two class problem with scores ranging from 0 to 1, where 0 is
                background.
            3D: N-class predictions before softmax where
                F.softmax(prediction, -1)[:, :, 0] are the probabilities of
                background.

        targets (torch.Tensor): A 2D tensor of labels/targets. 0 is background.

        pos_to_neg_ratio (float): Ratio of positives to negatives.
            default = 0.25
    """

    assert prediction.ndim == 2 or prediction.ndim == 3, \
        "hard_negative_mask: prediction must be 2D/3D tensor."
    assert targets.ndim == 2, \
        "hard_negative_mask: targets must be 2D tensor."
    prediction = prediction.clone()
    ns = prediction.size(0)
    foreground_mask = targets > 0

    if prediction.shape == foreground_mask.shape:
        # assumes, two class problem and sigmoid is applied to output
        assert 1. >= prediction.max() and prediction.min() >= 0., \
            "hard_negative_mask: Use torch.sigmoid(prediction)."
        # Mark the background_mask with hard negatives
        background_mask = torch.zeros_like(targets).bool()
        for i in range(ns):
            retain = max(1, int(foreground_mask[i].sum() / pos_to_neg_ratio))
            probs = prediction[i]
            # foreground prob to minimum
            probs[foreground_mask[i]] = 0.
            background_mask[i, torch.argsort(probs)[-retain:]] = True
    else:
        # assumes, N-class problem and softmax is not applied to output
        assert ~ prediction.sum(-1).eq(1).all(), \
            "hard_negative_mask: Requires predictions before softmax."
        background_probs = torch.nn.functional.softmax(prediction, -1)[:, :, 0]
        # Mark the background_mask with hard negatives
        background_mask = torch.zeros_like(targets).bool()
        for i in range(ns):
            retain = max(1, int(foreground_mask[i].sum() / pos_to_neg_ratio))
            probs = background_probs[i]
            # foreground prob to maximum
            probs[foreground_mask[i]] = 1.
            background_mask[i, torch.argsort(probs)[:retain]] = True
    mask = foreground_mask.bool() | background_mask.bool()
    return mask
