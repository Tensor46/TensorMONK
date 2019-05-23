""" TensorMONK :: loss :: CapsuleLoss """

import torch
from .utils import one_hot, compute_top15


class CapsuleLoss(torch.nn.Module):
    r""" For Dynamic Routing Between Capsules.
    Implemented  https://arxiv.org/pdf/1710.09829.pdf
    """
    def __init__(self, n_labels, *args, **kwargs):
        super(CapsuleLoss, self).__init__()
        self.n_labels = n_labels
        self.tensor_size = (1,)

    def forward(self, tensor, targets):
        onehot_targets = one_hot(targets, self.n_labels)
        # L2
        predictions = tensor.pow(2).sum(2).add(1e-6).pow(.5)
        # m+, m-, lambda, Tk all set per paper
        loss = onehot_targets*((.9 - predictions).clamp(0, 1e6)**2) + \
            (1 - onehot_targets)*.5 * ((predictions - .1).clamp(0, 1e6)**2)

        (top1, top5) = compute_top15(predictions.data, targets.data)
        return loss.sum(1).mean(), (top1, top5)
