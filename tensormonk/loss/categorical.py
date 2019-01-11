""" TensorMONK's :: loss :: Categorical """

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from .utils import compute_n_embedding, compute_top15, one_hot_idx


def nlog_likelihood(tensor, targets):
    return F.nll_loss(tensor.log_softmax(1), targets)


class Categorical(nn.Module):
    r""" Categorical with weight's to convert embedding to n_labels
    categorical responses.

    Args:
        tensor_size (int/list/tuple): shape of tensor in
            (None/any integer >0, channels, height, width) or
            (None/any integer >0, in_features) or in_features
        n_labels (int): number of labels
        type (str): loss function, options = entr/smax/tsmax/lmcl,
            default = entr
            entr  - categorical cross entropy
            smax  - softmax + negative log likelihood
            tsmax - taylor softmax + negative log likelihood
                 https://arxiv.org/pdf/1511.05042.pdf
            lmcl  - large margin cosine loss
                 https://arxiv.org/pdf/1801.09414.pdf  eq-4
            lmgm  - large margin Gaussian Mixture
                 https://arxiv.org/pdf/1803.02988.pdf  eq-17
        measure (str): cosine/dot, cosine similarity / matrix dot product,
            default = dot
        center (bool): center loss https://ydwen.github.io/papers/WenECCV16.pdf
        scale (float): lambda in center loss / lmgm / s in lcml, default = 0.5
        margin (float): margin for lcml, default = 0.3
        alpha (float): center or lmgm, default = 0.5
        defaults (float): deafults center, lcml, & lmgm parameters

    Return:
        loss, (top1, top5)
    """
    def __init__(self,
                 tensor_size,
                 n_labels,
                 type: str = "entr",
                 measure: str = "dot",
                 center: bool = False,
                 scale: float = 0.5,
                 margin: float = 0.3,
                 alpha: float = 0.5,
                 defaults: bool = False,
                 *args, **kwargs):
        super(Categorical, self).__init__()

        n_embedding = compute_n_embedding(tensor_size)
        self.type = type.lower()
        if "distance" in kwargs.keys():  # add future warning
            measure = kwargs["distance"]
        self.measure = measure.lower()
        assert self.type in ("entr", "smax", "tsmax", "lmcl", "lmgm"), \
            "Categorical :: type != entr/smax/tsmax/lmcl/lmgm"
        assert self.measure in ("dot", "cosine"), \
            "Categorical :: measure != dot/cosine"

        if defaults:
            if self.type == "lmcl":
                margin, scale = 0.35, 10
            if center:
                scale, alpha = 0.5, 0.01
            if self.type == "lmgm":
                alpha, scale = 0.01, 0.1

        self.center = center
        if center:
            self.register_buffer("center_alpha", torch.Tensor([alpha]).sum())
            self.centers = nn.Parameter(
                F.normalize(torch.randn(n_labels, n_embedding), p=2, dim=1))
            self.center_function = CenterFunction.apply

        self.scale = scale
        self.margin = margin
        self.alpha = alpha
        self.n_labels = n_labels

        self.weight = nn.Parameter(torch.randn(n_labels, n_embedding))
        self.tensor_size = (1, )

    def forward(self, tensor, targets):

        if self.type == "lmgm":
            # TODO euclidean computation is not scalable to larger n_labels
            # mahalanobis with identity covariance per paper = squared
            # euclidean -- does euclidean for stability
            # Switch to measure="cosine" if you have out of memory issues
            if self.measure == "cosine":
                self.weight.data = F.normalize(self.weight.data, p=2, dim=1)
                tensor = F.normalize(tensor, p=2, dim=1)
                responses = 1 - tensor.mm(self.weight.t())
            else:
                responses = (tensor.unsqueeze(1) - self.weight.unsqueeze(0))
                responses = responses.pow(2).sum(2).pow(0.5)
            (top1, top5) = compute_top15(- responses.data, targets.data)

            true_idx = one_hot_idx(targets, self.n_labels)
            responses = responses.view(-1)
            loss = self.scale * (responses[true_idx]).mean()
            responses[true_idx] = responses[true_idx] * (1 + self.alpha)
            loss = loss + nlog_likelihood(- responses.view(tensor.size(0), -1),
                                          targets)
            return loss, (top1, top5)

        if self.measure == "cosine" or self.type == "lmcl":
            self.weight.data = F.normalize(self.weight.data, p=2, dim=1)
            tensor = F.normalize(tensor, p=2, dim=1)
        responses = tensor.mm(self.weight.t())
        if self.measure == "cosine" or self.type == "lmcl":
            responses = responses.clamp(-1., 1.)
        (top1, top5) = compute_top15(responses.data, targets.data)

        if self.type == "tsmax":  # Taylor series
            responses = 1 + responses + 0.5*(responses**2)

        if self.type == "entr":
            loss = F.cross_entropy(responses, targets.view(-1))

        elif self.type in ("smax", "tsmax"):
            loss = nlog_likelihood(responses, targets)

        elif self.type == "lmcl":
            m, s = min(0.5, self.margin), max(self.scale, 1.)
            true_idx = one_hot_idx(targets, self.n_labels)
            responses = responses.view(-1)
            responses[true_idx] = responses[true_idx] - m
            responses = (responses * s).view(tensor.size(0), -1)
            loss = nlog_likelihood(responses, targets)
        else:
            raise NotImplementedError

        if self.center:
            loss = loss + self.center_function(tensor, targets.long(),
                                               self.centers, self.center_alpha)

        return loss, (top1, top5)


class CenterFunction(Function):

    @staticmethod
    def forward(ctx, tensor, targets, centers, alpha):
        ctx.save_for_backward(tensor, targets, centers, alpha)
        target_centers = centers.index_select(0, targets)
        return 0.5 * (tensor - target_centers).pow(2).sum()

    @staticmethod
    def backward(ctx, grad_output):

        tensor, targets, centers, alpha = ctx.saved_variables
        grad_tensor = grad_centers = None
        grad_centers = torch.zeros(centers.size()).to(tensor.device)
        grad_tensor = tensor - centers.index_select(0, targets.long())

        unique = torch.unique(targets.long())
        for j in unique:
            grad_centers[j] += centers[j] - \
                tensor.data[targets == j].mean(0).mul(alpha)

        return grad_tensor * grad_output, None, grad_centers, None


# tensor = torch.rand(3, 256)
# test = Categorical(256, 10, "smax", center=True)
# targets = torch.tensor([1, 3, 6])
# test(tensor, targets)
