""" TensorMONK's :: NeuralLayers :: CategoricalLoss                          """

import torch,torch.nn as nn, torch.nn.functional as F, numpy as np
from   torch.autograd  import Variable

#==============================================================================#


class CapsuleLoss(nn.Module):
    def __init__(self, n_labels, *args, **kwargs):
        super(CapsuleLoss,self).__init__()

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


class SupervisedLoss(nn.Module):
    """
        type
            entropy         ::
            softmax         ::
            tayloredSoftmax ::
            triplet         ::

        distance
            cosine          ::
            dot             ::

        center

    """
    def __init__(self, tensor_size=128, n_labels=10, *args, **kwargs):
        super(CategoricalLoss,self).__init__()
        if type(tensor_size) in [list, tuple]:
            if len(tensor_size)>1: tensor_size = tensor_size[1:]
        self.n_labels = n_labels
        self.weight = nn.Parameter(torch.Tensor(int(np.prod(tensor_size)),n_labels))
        nn.init.orthogonal_(self.weight, gain=1./np.sqrt(np.prod(tensor_size)))
        self.m, self.s = .3, 10
        self.tensor_size = (1,)

    def forward(self, tensor, targets):
        BSZ = tensor.size(0)
        responses = tensor.mm(self.weight)
        responses = responses.clamp(-1., 1.)

        predicted = responses.topk(5, 1, True, True)[1]
        predicted = predicted.t()
        correct = predicted.eq(targets.view(1,-1).expand_as(predicted))

        top1 = correct[:1].view(-1).float().sum().mul_(100.0/BSZ)
        top5 = correct[:5].view(-1).float().sum().mul_(100.0/BSZ)

        genuineIDX = Variable(torch.from_numpy(np.arange(BSZ)))
        if targets.is_cuda: genuineIDX = genuineIDX.cuda()
        genuineIDX = targets.view(-1) + genuineIDX * self.n_labels

        # Cost = F.cross_entropy(responses,targets.view(-1))
        responses = torch.exp(responses)
        Cost = - torch.log(responses.view(-1)[genuineIDX] / responses.sum(1)).sum() / BSZ
        return Cost, (top1, top5)


# tensor = torch.rand(3,256)
# test = categoricalLoss(256, 10)
# targets = torch.tensor([1,3,6])
# test(tensor,targets)
