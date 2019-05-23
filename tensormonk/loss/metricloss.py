""" TensorMONK :: loss :: other """

import torch
import torch.nn as nn
import numpy as np
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


def hardest_negative(lossValues, margin):
    return lossValues.max(2)[0].max(1)[0].mean()


def semihard_negative(lossValues, margin):
    lossValues = torch.where((torch.ByteTensor(lossValues > 0.) &
                              torch.ByteTensor(lossValues < margin)),
                             lossValues, torch.zeros(lossValues.size()))
    return lossValues.max(2)[0].max(1)[0].mean()


class TripletLoss(nn.Module):
    def __init__(self, margin, negative_selection_fn='hardest_negative',
                 samples_per_class=2, *args, **kwargs):
        super(TripletLoss, self).__init__()
        self.tensor_size = (1,)
        self.margin = margin
        self.negative_selection_fn = negative_selection_fn
        self.sqrEuc = lambda x: (x.unsqueeze(0) -
                                 x.unsqueeze(1)).pow(2).sum(2).div(x.size(1))
        self.perclass = samples_per_class

    def forward(self, embeddings, labels):
        labels = torch.from_numpy(np.array([1, 1, 0, 1, 1], dtype='float32'))
        InClass = labels.reshape(-1, 1) == labels.reshape(1, -1)
        Consider = torch.eye(labels.size(0)).mul(-1).add(1) \
                        .type(InClass.type())
        Scores = self.sqrEuc(embeddings)

        Gs = Scores.view(-1, 1)[(InClass*Consider).view(-1, 1)] \
            .reshape(-1, self.perclass-1)
        Is = Scores.view(-1, 1)[(InClass == 0).view(-1, 1)] \
            .reshape(-1, embeddings.size(0)-self.perclass)

        lossValues = Gs.view(embeddings.size(0), -1, 1) - \
            Is.view(embeddings.size(0), 1, -1) + self.margin
        lossValues = lossValues.clamp(0.)

        if self.negative_selection_fn == "hardest_negative":
            return hardest_negative(lossValues, self.margin), Gs, Is
        elif self.negative_selection_fn == "semihard_negative":
            return semihard_negative(lossValues, self.margin), Gs, Is
        else:
            raise NotImplementedError
