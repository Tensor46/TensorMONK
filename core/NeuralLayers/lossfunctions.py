""" TensorMONK's :: NeuralLayers :: LossFunctions                            """

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#==============================================================================#

def hardest_negative(lossValues,margin):
    return lossValues.max(2)[0].max(1)[0].mean()

def semihard_negative(lossValues, margin):
    lossValues = torch.where((torch.ByteTensor(lossValues>0.) & torch.ByteTensor(lossValues<margin)), lossValues, torch.zeros(lossValues.size()))
    return lossValues.max(2)[0].max(1)[0].mean()


class TripletLoss(nn.Module):
    def __init__(self, margin, negative_selection_fn='hardest_negative', samples_per_class = 2, *args, **kwargs):
        super(TripletLoss, self).__init__()
        self.tensor_size = (1,)
        self.margin = margin
        self.negative_selection_fn = negative_selection_fn
        self.sqrEuc = lambda x : (x.unsqueeze(0) - x.unsqueeze(1)).pow(2).sum(2).div(x.size(1))
        self.perclass = samples_per_class

    def forward(self, embeddings, labels):
        InClass     = labels.reshape(-1,1) == labels.reshape(1,-1)
        Consider    = torch.eye(labels.size(0)).mul(-1).add(1).type(InClass.type())
        Scores      = self.sqrEuc(embeddings)
        Gs          = Scores.view(-1, 1)[(InClass*Consider).view(-1, 1)].reshape(-1, self.perclass-1)
        Is          = Scores.view(-1, 1)[(InClass == 0).view(-1, 1)].reshape(-1, embeddings.size(0)-self.perclass)
        lossValues = Gs.view(embeddings.size(0), -1, 1) - Is.view(embeddings.size(0), 1, -1) + self.margin
        lossValues = lossValues.clamp(0.)
        if self.negative_selection_fn == "hardest_negative":
            return hardest_negative(lossValues, self.margin), Gs, Is
        elif self.negative_selection_fn == "semihard_negative":
            return semihard_negative(lossValues, self.margin), Gs, Is
        else:
            raise NotImplementedError
#==============================================================================#


class DiceLoss(nn.Module):
    """
    Implemented from https://arxiv.org/pdf/1803.11078.pdf
    """
    def __init__(self, type = "tversky", *args, **kwargs):
        self.tensor_size = (1,)
        if type == "tversky":
            self.beta = 2.0
        elif type == "dice":
            self.beta = 1.0         # below Eq(6)
        else:
            raise NotImplementedError
    def forward(self, prediction, targets):
        top1, top5 = 0., 0.
        p_i = prediction
        p_j = prediction.mul(-1).add(1)

        g_i = targets
        g_j = targets.mul(-1).add(1)

        num = (p_i*g_i).sum(1).sum(1).mul((1 + self.beta**2))   # eq(5)
        den = num.add((p_i*g_j).sum(1).sum(1).mul((self.beta**2))).add((p_j*g_i).sum(1).sum(1).mul((self.beta)))    # eq(5)
        loss = num/den
        return loss.mean(), (top1, top5)
#==============================================================================#


class CapsuleLoss(nn.Module):
    """
        Implemented  https://arxiv.org/pdf/1710.09829.pdf
    """
    def __init__(self, n_labels, *args, **kwargs):
        super(CapsuleLoss, self).__init__()

        self.n_labels = n_labels
        self.tensor_size = (1,)

    def forward(self, features, targets):
        identity = torch.eye(self.n_labels)
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


class CenterLoss(nn.Module):
    """
        Implemented https://ydwen.github.io/papers/WenECCV16.pdf
    """
    def __init__(self,
                 tensor_size = 46,
                 n_labels    = 10,
                 distance    = "euclidean",
                 **kwargs):
        super(CenterLoss, self).__init__()

        distance = distance.lower()
        assert distance in ["cosine", "euclidean"], \
            "CenterLoss :: Distance must be cosine/euclidean"

        if isinstance(tensor_size, list) or isinstance(tensor_size, tuple):
            if len(tensor_size)>1: # batch size is not required
                tensor_size = np.prod(tensor_size[1:])
            else:
                tensor_size = tensor_size[0]
        n_embedding = tensor_size
        self.distance = distance.lower()

        self.n_labels = n_labels
        self.weight = nn.Parameter(torch.Tensor(n_labels, int(np.prod(n_embedding))))
        nn.init.orthogonal_(self.weight, gain=1./np.sqrt(np.prod(n_embedding)))
        self.weight.data.div_(self.weight.pow(2).sum(1, True))
        self.tensor_size = (1,)

    def forward(self, tensor, targets):
        # onehot targets
        identity = torch.eye(self.n_labels).to(tensor.device)
        onehot = identity.index_select(dim=0, index=targets.view(-1))
        idx = onehot.view(-1).nonzero()

        if self.distance == "cosine":
            # l2 weights and features
            self.weight.data.div_(self.weight.pow(2).sum(1, True))
            tensor = tensor.div(tensor.pow(2).sum(1, True))
            distances = tensor.mm(self.weight.t()).clamp(-1., 1.)
            loss = 1 - distances.view(-1)[idx].mean()
        else: # euclidean
            distances = (tensor.unsqueeze(1) - self.weight.unsqueeze(0))
            distances = distances.pow(2).sum(2).add(1e-6).pow(0.5)
            loss = distances.view(-1)[idx].mean()
        return loss
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
        distance = distance.lower()
        if isinstance(tensor_size, list) or isinstance(tensor_size, tuple):
            if len(tensor_size)>1: # batch size is not required
                tensor_size = np.prod(tensor_size[1:])
            else:
                tensor_size = tensor_size[0]
        n_embedding = tensor_size

        self.type = type.lower()
        self.distance = distance
        if center:
            self.center = CenterLoss(n_embedding, n_labels,
                "cosine" if distance == "cosine" else "euclidean")

        self.n_labels = n_labels
        self.weight = nn.Parameter(torch.Tensor(n_labels, int(np.prod(n_embedding))))
        nn.init.orthogonal_(self.weight, gain=1./np.sqrt(np.prod(n_embedding)))
        self.m, self.s = .3, 10
        self.tensor_size = (1,)

    def forward(self, features, targets):
        BSZ = features.size(0)

        if self.distance == "cosine" or self.type == "lmcl":
            self.weight.data = self.weight.data.div(self.weight.pow(2).sum(1, True).pow(.5).add(1e-8))
            features = features.div(features.pow(2).sum(1, True).pow(.5).add(1e-8))
            responses = features.mm(self.weight.t())
            responses = responses.clamp(-1., 1.)
        else:
            responses = features.mm(self.weight.t())

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
            identity = torch.eye(self.n_labels)
            if targets.is_cuda:
                identity = identity.cuda()
            onehot_targets = identity.index_select(dim=0, index=targets.view(-1))
            responses = torch.exp(responses)
            loss = -torch.log( (responses*onehot_targets).sum(1) / responses.sum(1) ).sum() / BSZ

        elif self.type == "lmcl":
            m, s = 0.35, 10  # From https://arxiv.org/pdf/1801.09414.pdf
            genuineIDX = torch.from_numpy(np.arange(BSZ))
            if targets.is_cuda: genuineIDX = genuineIDX.cuda()
            genuineIDX = targets.view(-1) + genuineIDX * self.n_labels
            responses = responses.view(-1)
            responses[genuineIDX] = responses[genuineIDX] - m
            responses = responses.view(BSZ,-1)
            responses = torch.exp(responses*s)
            loss = - torch.log(responses.view(-1)[genuineIDX] / responses.sum(1)).sum() / BSZ
        else:
            raise NotImplementedError

        if hasattr(self, "center"):
            loss += self.center(features, targets)

        return loss, (top1, top5)


# tensor = torch.rand(3,256)
# test = CategoricalLoss(256, 10, "smax",)
# targets = torch.tensor([1,3,6])
# test(tensor,targets)
