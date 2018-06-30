""" TensorMONK's :: NeuralLayers :: LossFunctions                            """

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd  import Variable

import numpy as np
#==============================================================================#


class CapsuleLoss(nn.Module):
    def __init__(self, n_labels, *args, **kwargs):
        super(CapsuleLoss, self).__init__()

        self.n_labels = n_labels
        self.tensor_size = (1,)

    def forward(self, features, targets):
        identity = Variable(torch.eye(self.n_labels))
        if targets.is_cuda:
            identity = identity.cuda()
        onehot_targets = identity.index_select(dim=0, index=targets.view(-1))
        # L2
        predictions = features.pow(2).sum(2).pow(.5)
        # m+, m-, lambda, Tk all set per paper
        loss = onehot_targets*((.9-predictions).clamp(0, 1e6)**2) + \
               (1-onehot_targets)*.5*((predictions-.1).clamp(0, 1e6)**2)

        predicted = predictions.topk(5, 1, True, True)[1]
        predicted = predicted.t()
        correct = predicted.eq(targets.view(1,-1).expand_as(predicted))
        top1 = correct[:1].view(-1).float().sum().mul_(100.0/features.size(0))
        top5 = correct[:5].view(-1).float().sum().mul_(100.0/features.size(0))

        return loss.sum(1).mean(), (top1, top5)
#==============================================================================#


class CategoricalLoss(nn.Module):
    """

        Parameters
        ----------
        type         :: entr/smax/tsmax/tentr/lmcl
                        entr  - categorical cross entropy
                        smax  - softmax
                        tsmax - taylor softmax
                                https://arxiv.org/pdf/1511.05042.pdf
                        tentr - taylor entropy
                        lmcl  - large margin cosine loss
                                https://arxiv.org/pdf/1801.09414.pdf

        distance     :: cosine/dot
                        cosine similarity / matrix dot product
        center       :: True/False
                        https://ydwen.github.io/papers/WenECCV16.pdf

        Other inputs
        ------------
        tensor_size         :: feature length
        n_labels            :: number of output labels

    """
    def __init__(self, tensor_size=128, n_labels=10, type="entr",
                 distance="dot", center=False, *args, **kwargs):
        super(CategoricalLoss, self).__init__()
        if isinstance(tensor_size, list) or isinstance(tensor_size, tuple):
            if len(tensor_size)>1: # batch size is not required
                tensor_size = tensor_size[1:]

        self.type = type.lower()
        self.distance = distance.lower()
        self.center = center

        self.n_labels = n_labels
        self.weight = nn.Parameter(torch.Tensor(int(np.prod(tensor_size)),n_labels))
        nn.init.orthogonal_(self.weight, gain=1./np.sqrt(np.prod(tensor_size)))
        self.m, self.s = .3, 10
        self.tensor_size = (1,)

    def forward(self, features, targets):
        BSZ = features.size(0)

        if self.distance == "cosine" or self.type == "lmcl":
            responses = features.mm(self.weight).div( ((features**2).sum(1,keepdim=True)**.5) * ((self.weight**2).sum(0,keepdim=True)**.5) )
            responses = responses.clamp(-1., 1.)
        else:
            responses = features.mm(self.weight)

        predicted = responses.topk(5, 1, True, True)[1]
        predicted = predicted.t()
        correct = predicted.eq(targets.view(1,-1).expand_as(predicted))
        top1 = correct[:1].view(-1).float().sum().mul_(100.0/BSZ)
        top5 = correct[:5].view(-1).float().sum().mul_(100.0/BSZ)

        # Taylor series
        if self.type == "tsmax" or self.type == "tentr":
            responses = 1 + responses + 0.5*(responses**2)

        if self.type.endswith("entr"):
            loss = F.cross_entropy(responses,targets.view(-1))

        elif self.type.endswith("smax"):
            identity = Variable(torch.eye(self.n_labels))
            if targets.is_cuda:
                identity = identity.cuda()
            onehot_targets = identity.index_select(dim=0, index=targets.view(-1))
            responses = torch.exp(responses)
            loss = -torch.log( (responses*onehot_targets).sum(1) / responses.sum(1) ).sum() / BSZ

        elif self.type == "lmcl":
            m, s = 0.35, 10  # From https://arxiv.org/pdf/1801.09414.pdf
            genuineIDX = Variable(torch.from_numpy(np.arange(BSZ)))
            if targets.is_cuda: genuineIDX = genuineIDX.cuda()
            genuineIDX = targets.view(-1) + genuineIDX * self.n_labels
            responses = responses.view(-1)
            responses[genuineIDX] = responses[genuineIDX] - m
            responses = responses.view(BSZ,-1)
            responses = torch.exp(responses*s)
            loss = - torch.log(responses.view(-1)[genuineIDX] / responses.sum(1)).sum() / BSZ
        else:
            raise NotImplementedError

        return loss, (top1, top5)


# tensor = torch.rand(3,256)
# test = CategoricalLoss(256, 10, "smax",)
# targets = torch.tensor([1,3,6])
# test(tensor,targets)
