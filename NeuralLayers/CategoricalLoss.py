""" TensorMONK's :: NeuralLayers :: CategoricalLoss                          """

import torch,torch.nn as nn, torch.nn.functional as F, numpy as np
from   torch.autograd  import Variable

#==============================================================================#


class CategoricalLoss(nn.Module):
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
