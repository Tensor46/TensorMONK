""" TensorMONK :: loss :: other """

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils import Measures, Checks
from torch import Tensor
from itertools import combinations
import numpy as np
import warnings


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
        warnings.warn("TripletLoss is deprecated, use MetricLoss",
                      DeprecationWarning)
        self.tensor_size = (1,)
        self.margin = margin
        self.negative_selection_fn = negative_selection_fn
        self.sqrEuc = lambda x: (x.unsqueeze(0) -
                                 x.unsqueeze(1)).pow(2).sum(2).div(x.size(1))
        self.perclass = samples_per_class

    def forward(self, embeddings, labels):
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


class MetricLoss(nn.Module):
    r""" Metric loss (triplet/anuglar triplet/n-pair) with mining (hard/semi).

    Args:
        tensor_size (int/list/tuple, required)
            Shape of tensor in (None/any integer >0, channels, height, width)
            or (None/any integer >0, in_features) or in_features
        n_labels (int, required)
            Number of labels
        loss_type (str, default="triplet")
            To replicate actual implementation of angular_triplet/triplet use
            triplet sampling, mining = None, and boundary = None. Otherwise,
            all possible triplets within a batch are used to compute the loss.
            Similarly, n-pair requires n-pair sampling, and normalize = False.
            Mining is disabled for n-pair.

            "triplet"
                (https://arxiv.org/pdf/1503.03832.pdf)
            "angular_triplet"
                (https://arxiv.org/pdf/1708.01682.pdf)
            "n_pair"
                (http://www.nec-labs.com/uploads/images/Department-Images/
                MediaAnalytics/papers/nips16_npairmetriclearning.pdf)
        measure (str, default="euclidean")
            Options = cosine/euclidean. angular_triplet will only use dot
            product. Euclidean is replaced by squared euclidean for stability.
        margin (float, default=0.5)
            margin for triplet loss. Also, used by angular_triplet when "semi"
            mining.
        boundary (float, default=None)
            When boundary is a float, margin for anchor vs positive and
            anchor vs negative must satisfy the following for a zero loss:
                anchor vs positive < boundary - margin / 2
                anchor vs negative > boundary + margin / 2
        mining (str, default=None)
            Options = None/hard/semi. When hard, hard negative mining is done.
            Essentially, the hardest negative for each unique label (must have
            multiple samples) are averaged.
            When semi, only the loss values between 0 and margin are averaged.
        normalize (bool, default=True)
            When True, converts feature vector are normalized.
        l2_factor (float, default=0.02)
            l2_factor is part of n-pair loss, and is enabled only when
            normalize is False.
        alpha (float, default=45)
            Alpha for angular triplet loss.

    Return:
        loss
    """

    def __init__(self,
                 tensor_size: tuple,
                 n_labels: int,
                 loss_type: str = "triplet",
                 measure: str = "euclidean",
                 margin: float = 0.5,
                 boundary: float = None,
                 mining: str = None,
                 normalize: bool = True,
                 l2_factor: float = 0.02,
                 alpha: float = 45.):
        super(MetricLoss, self).__init__()
        METHODS = ("triplet", "angular_triplet", "n_pair")
        MEASURES = ("cosine", "euclidean")
        MINING = ("hard", "semi")

        # checks
        checker = Checks("MetricLoss")
        # n_embedding = compute_n_embedding(tensor_size)
        checker.arg_type("n_labels", n_labels, int, "int")
        checker.arg_value("n_labels", n_labels, lambda x: x > 0, "must be > 0")
        self.n_labels = n_labels
        checker.arg_type("loss_type", loss_type, str, "str")
        checker.arg_value("loss_type", loss_type.lower(),
                          lambda x: x in METHODS, "/".join(METHODS))
        self.loss_type = loss_type.lower()
        checker.arg_type("measure", measure, str, "str")
        checker.arg_value("measure", measure.lower(),
                          lambda x: x in MEASURES, "/".join(MEASURES))
        self.measure = measure.lower()
        checker.arg_type("margin", margin, float, "float")
        checker.arg_value("margin", margin, lambda x: x > 0, "> 0")
        self.margin = margin
        if boundary is not None:
            checker.arg_type("boundary", boundary, float, "float")
            checker.arg_value("boundary", boundary, lambda x: x > 0, "> 0")
            self.pos_margin = boundary - self.margin * .5
            self.neg_margin = boundary + self.margin * .5
        self.boundary = boundary
        if mining is not None:
            checker.arg_type("mining", mining, str, "str")
            mining = mining.lower()
            checker.arg_value("mining", mining, lambda x: mining in MINING,
                              "/".join(MINING) + "/None")
        self.mining = mining
        checker.arg_type("normalize", normalize, bool, "bool")
        self.normalize = normalize
        checker.arg_type("l2_factor", l2_factor, float, "float")
        checker.arg_value("l2_factor", l2_factor, lambda x: 1 > x >= 0,
                          "1 > l2_factor >= 0")
        self.l2_factor = l2_factor
        checker.arg_type("alpha", alpha, float, "float")
        checker.arg_value("alpha", alpha, lambda x: 90 >= x >= 0,
                          "90 >= alpha >= 0")
        self.alpha = alpha
        self.tan_alpha = np.tan(np.deg2rad(alpha))
        self.ignore_zeros = True if self.mining == "semi" else False
        self.tensor_size = (1, )

    def forward(self, tensor: Tensor, targets: Tensor) -> Tensor:
        assert targets.numel() > torch.unique(targets).numel(), \
            "MetricLoss: Requires multiple samples per label in a batch."

        if self.loss_type == "triplet":
            loss = self._triplet(tensor, targets)
        elif self.loss_type == "angular_triplet":
            loss = self._angular_triplet(tensor, targets)
        else:
            loss = self._n_pair(tensor, targets)
        return loss

    def _triplet(self, tensor: Tensor, targets: Tensor):
        def measure_error(a_vs_pos: Tensor, a_vs_neg: Tensor):
            loss = (self.margin + a_vs_pos - a_vs_neg).clamp(0) \
                if self.boundary is None else \
                ((a_vs_pos - self.pos_margin).clamp(0) +
                 (self.neg_margin - a_vs_neg).clamp(0))
            return loss

        if self._is_triplet_sampling(targets) and \
           self.mining not in ("hard", "semi"):
            # batch size must be in the multiples of 3 (anchor, +ve, -ve)
            # Ex: targets = [4, 4, 6, 9, 9, 0, 6, 6, 4, ...]
            anchor_vs_positive = self._distance(tensor[0::3], tensor[1::3])
            anchor_vs_negative = self._distance(tensor[0::3], tensor[2::3])
            loss = measure_error(anchor_vs_positive, anchor_vs_negative)
            loss = self._ignore_zeros(loss.view(-1)).mean()
        else:
            # otherwise, compute all the possible triplet pairs within a batch
            # per class
            distance = self._distance_pairwise(tensor)
            loss = []
            for pos, neg in zip(*self._pos_n_neg_idx_per_label(targets)):
                possible = measure_error(distance[pos].view(-1, 1),
                                         distance[neg].view(1, -1))
                loss.append(self._mining(possible.view(-1)))
            loss = torch.cat(loss).mean()
        return loss

    def _angular_triplet(self, tensor: Tensor, targets: Tensor):
        tan_sq_alpha = self.tan_alpha ** 2
        tensor = F.normalize(tensor, 2, 1)

        if self._is_triplet_sampling(targets) and \
           self.mining not in ("hard", "semi"):
            # batch size must be in the multiples of 3 (anchor, +ve, -ve)
            # Ex: targets = [4, 4, 6, 9, 9, 0, 6, 6, 4, ...]
            a, p, n = tensor[0::3], tensor[1::3], tensor[2::3]
            fapn = 4 * tan_sq_alpha * ((a + p) * n).sum(1) - \
                2 * (1 + tan_sq_alpha) * (a * p).sum(1)
            loss = (1 + fapn.exp()).log().mean()
        else:
            loss = []
            for x in self._valid_unique(targets):
                p = tensor[targets.eq(x)]
                n = tensor[targets.ne(x)]
                idx = torch.Tensor(list(combinations(range(p.shape[0]), 2)))
                idx = idx.long().to(targets.device)
                xa_xp = (p[idx[:, 0]] * p[idx[:, 1]]).sum(1)
                xaxp_xn = (p[idx[:, 0]] + p[idx[:, 1]]) @ n.t()
                fapn = 4 * tan_sq_alpha * xaxp_xn.view(xa_xp.numel(), -1) - \
                    2 * (1 + tan_sq_alpha) * xa_xp.view(-1, 1)
                possible = (1 + fapn.exp()).log()
                loss.append(self._mining(possible.view(-1)))
            loss = torch.cat(loss).mean()
        return loss

    def _n_pair(self, tensor: Tensor, targets: Tensor):
        # Improved Deep Metric Learning with Multi-class N-pair Loss Objective
        # Works on any batch as long as at least one label has multiple samples
        # However, the actual n-pair sampling batch must be in the multiples of
        # 2 (anchor, +ve)
        # Ex: targets = [4, 4, 6, 6, 9, 9, 0, 0, 2, 2, ...]
        distance = self._distance_pairwise(tensor)
        scores = 1 - distance.div(2 if self.measure == "cosine" else 4)
        # Scores are normalized between 0-1, where 1 means identical vectors.
        # Maximum possible distance for two unit vectors is 2, when not cosine
        # we use squared euclidean (so 4 is max)
        loss = []
        for pos, neg in zip(*self._pos_n_neg_idx_per_label(targets)):
            possible = scores[neg].view(1, -1) - scores[pos].view(-1, 1)
            possible = possible.exp().sum(1).add(1).log()
            loss.append(possible.view(-1))
        loss = torch.cat(loss).mean()
        # paper adds a l2 regularizations term - given the tensor is not
        # normalized, else not required
        if not self.normalize:
            loss = loss + self.l2_factor * tensor.pow(2).sum(1).mean()
        return loss

    def _distance(self, a: Tensor, b: Tensor,
                  measure: str = None, normalize: bool = None):
        measure = self.measure if measure is None else measure
        normalize = self.normalize if normalize is None else normalize
        if measure == "cosine":
            return 1 - Measures.cosine(a, b)
        return Measures.sqr_euclidean(a, b, normalize)

    def _distance_pairwise(self, a: Tensor, b: Tensor = None,
                           measure: str = None, normalize: bool = None):
        b = a if b is None else b
        measure = self.measure if measure is None else measure
        normalize = self.normalize if normalize is None else normalize
        if measure == "cosine":
            return 1 - Measures.cosine_pairwise(a, b)
        return Measures.sqr_euclidean_pairwise(a, b, normalize)

    def _mining(self, loss: Tensor):
        if self.mining == "hard":
            return loss.max().unsqueeze(0)
        else:
            if loss.numel() > loss.gt(self.margin).sum() and \
               self.mining == "semi":
                loss = loss[loss.le(self.margin)]
            return self._ignore_zeros(loss).view(-1)

    def _ignore_zeros(self, loss: Tensor):
        if self.ignore_zeros and \
           (loss.numel() > loss.nonzero().numel() > 0):
            loss = loss[loss.nonzero()]
        return loss

    @torch.no_grad()
    def _pos_n_neg_idx_per_label(self, targets: Tensor):
        same_label = self._same_label(targets)
        pos_valid = self._valid_comparisons(targets, only_lower=True)
        neg_valid = self._valid_comparisons(targets, only_lower=False)
        pos_per_label, neg_per_label = [], []
        for x in self._valid_unique(targets):
            pos_per_label.append((targets.eq(x).view(-1, 1) *
                                  same_label * pos_valid))
            neg_per_label.append((targets.eq(x).view(-1, 1) *
                                  same_label.eq(0) * neg_valid))
        return pos_per_label, neg_per_label

    @torch.no_grad()
    def _valid_unique(self, targets: Tensor):
        unique = torch.unique(targets).view(-1, 1)
        unique = unique[(unique == targets.view(1, -1)).sum(1).gt(1)]
        return unique.view(-1)

    @torch.no_grad()
    def _same_label(self, targets: Tensor):
        return targets.view(-1, 1).eq(targets.view(1, -1))

    @torch.no_grad()
    def _valid_comparisons(self, targets: Tensor, only_lower: bool = True):
        valid = torch.eye(targets.numel()).to(targets.device).eq(0)
        if only_lower:
            idx = valid.nonzero()
            idx = idx[idx[:, 0] >= idx[:, 1]]
            valid = valid.view(-1)
            valid[idx[:, 0] + idx[:, 1] * targets.numel()] = 0
            valid = valid.view(targets.numel(), -1)
        return valid

    @torch.no_grad()
    def _is_triplet_sampling(self, targets: Tensor):
        # batch size must be in the multiples of 3 (anchor, +ve, -ve)
        # Ex: targets = [4, 4, 6, 9, 9, 0, 6, 6, 4, ...]
        return ((targets.shape[0] % 3) == 0 and
                all(targets[0::3] == targets[1::3]) and
                all(targets[0::3] != targets[2::3]) and
                all(targets[1::3] != targets[2::3]))

    @torch.no_grad()
    def _is_n_pair_sampling(self, targets: Tensor):
        # batch size must be in the multiples of 2 (anchor, +ve)
        # Ex: targets = [4, 4, 6, 6, 9, 9, 0, 0, 6, 6, ...]
        return ((targets.shape[0] % 2) == 0 and
                all(targets[0::2] == targets[1::2]) and
                all(targets[0::2][:-1] != targets[0::2][1:]))


# tensor = F.normalize(torch.Tensor([[0.10, 0.60, 0.20, 0.10],
#                                    [0.15, 0.50, 0.22, 0.11],
#                                    [0.90, 0.50, 0.96, 0.11],
#                                    [0.90, 0.10, 0.26, 0.71],
#                                    [0.85, 0.20, 0.27, 0.78],
#                                    [0.01, 0.90, 0.91, 0.92],
#                                    [0.80, 0.45, 0.86, 0.16],
#                                    [0.92, 0.56, 0.99, 0.06],
#                                    [0.08, 0.56, 0.16, 0.16]]), 2, 1)
# targets = torch.Tensor([4, 4, 6, 9, 9, 0, 6, 6, 4]).long()
# idx = torch.Tensor([0, 2, 1, 3, 5, 4, 6, 8, 7]).long()
# MetricLoss((1, 4), 4, "triplet", "euclidean", 0.5)(tensor, targets)
# MetricLoss((1, 4), 4, "triplet", "euclidean", 0.5)(tensor[idx], targets)
# MetricLoss((1, 4), 4, "triplet", "cosine", 0.5)(tensor, targets)
# MetricLoss((1, 4), 4, "triplet", "euclidean", 0.5, .8, None)(tensor, targets)
# MetricLoss((1, 4), 4, "triplet", mining="semi")(tensor, targets)
# MetricLoss((1, 4), 4, "triplet", mining="hard")(tensor, targets)
# MetricLoss((1, 4), 4, "angular_triplet")(tensor, targets)
# MetricLoss((1, 4), 4, "angular_triplet", mining="semi")(tensor, targets)
# MetricLoss((1, 4), 4, "angular_triplet", mining="hard")(tensor, targets)
# MetricLoss((1, 4), 4, "n_pair", "euclidean", margin=0.5)(tensor, targets)
# MetricLoss((1, 4), 4, "n_pair", "cosine", margin=0.5)(tensor, targets)
