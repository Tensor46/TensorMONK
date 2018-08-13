""" TensorMONK's :: NeuralArchitectures                                      """

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from ..NeuralLayers import *
#==============================================================================#


class CapsuleNet(nn.Module):
    """
        Implemented https://arxiv.org/pdf/1710.09829.pdf for MNIST
    """
    def __init__(self, tensor_size=(6, 1, 28, 28), n_labels=10,
                 primary_n_capsules=8, primary_capsule_length=32,
                 routing_capsule_length=16, routing_iterations=3,
                 replicate_paper=True, *args, **kwargs):
        super(CapsuleNet, self).__init__()

        if replicate_paper:
            primary_n_capsules = 8
            primary_capsule_length = 32
            routing_capsule_length = 16
            routing_iterations = 3
            block = Convolution
            self.InitialConvolutions = Convolution(tensor_size, filter_size=9, out_channels= 256,
                                                   strides=1, pad=False, activation="relu")
            _tensor_size = self.InitialConvolutions.tensor_size
        else: # You can be creative!
            block = Convolution
            self.InitialConvolutions = nn.Sequential(Convolution(tensor_size, 5, 64, 1, False, "relu", 0., None, False),
                                                     ResidualComplex((6,  64, 24, 24), 3, 256, 1, True, "relu", 0., None, False),
                                                     ResidualComplex((6, 256, 24, 24), 3, 256, 1, True, "relu", 0., None, False),
                                                     ResidualComplex((6, 256, 24, 24), 3, 256, 1, True, "relu", 0., None, False),
                                                     ResidualComplex((6, 256, 24, 24), 3, 256, 1, True, "relu", 0., None, False),
                                                     Convolution((6, 256, 24, 24), 5, 64, 1, False, "", 0., None, False),)
            _tensor_size = self.InitialConvolutions[-1].tensor_size
        print("InitialConvolutions output size :: ", _tensor_size)

        # block can be replaced with any module available in NeuralLayers
        self.PrimaryCapsule = PrimaryCapsule(_tensor_size,
                                             filter_size=9, out_channels=256, strides=2,
                                             pad=False, activation="", dropout=0.,
                                             batch_nm=False, pre_nm=False,
                                             block=block, n_capsules=primary_n_capsules,
                                             capsule_length=primary_capsule_length)
        print("Primary capsule output size :: ", self.PrimaryCapsule.tensor_size)
        self.RoutingCapsule = RoutingCapsule(self.PrimaryCapsule.tensor_size,
                                             n_capsules=n_labels, capsule_length=routing_capsule_length,
                                             iterations=routing_iterations)

        print("Routing capsule output size :: ", self.RoutingCapsule.tensor_size)
        self.Reconstruction = nn.Sequential(nn.Linear(n_labels*routing_capsule_length, 512), nn.ReLU(),
                                            nn.Linear(512, 1024), nn.ReLU(),
                                            nn.Linear(1024, int(np.prod(tensor_size[1:]))), nn.Sigmoid())

        self.tensor_size = (6, n_labels)
        self.input_tensor_size = tensor_size

    def forward(self, tensor, targets):
        tensor_size = tensor.size()

        # CapsuleNet
        tensor_deep = self.InitialConvolutions(tensor)
        tensor_primary = self.PrimaryCapsule(tensor_deep)
        features = self.RoutingCapsule(tensor_primary)

        # Reconstruction (only during training)
        #   remove 'if' loop if you like to view the rec_tensor on test data
        rec_tensor = None
        rec_loss = 0.
        if self.training:
            identity = Variable(torch.eye(self.tensor_size[1]))
            if targets.is_cuda:
                identity = identity.cuda()
            onehot_targets = identity.index_select(dim=0, index=targets.view(-1))
            rec_tensor = self.Reconstruction((features * onehot_targets[:, :, None]).view(tensor_size[0], -1))
            rec_loss = F.mse_loss(rec_tensor.view(tensor_size[0], -1), tensor.view(tensor_size[0], -1))
            rec_tensor = rec_tensor.view(*tensor_size)

        return features, rec_tensor, rec_loss


# from core.NeuralLayers import *
# tensor_size = (2, 1, 28, 28)
# targets = torch.LongTensor([1, 2])
# tensor = torch.rand(*tensor_size)
# test = CapsuleNet(tensor_size)
# test(tensor, targets)[2]
