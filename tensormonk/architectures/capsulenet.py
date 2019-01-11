""" TensorMONK :: architectures """

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers import Convolution, ResidualComplex, PrimaryCapsule, \
    RoutingCapsule


class CapsuleNet(nn.Module):
    r"""Dynamic routing between capsules - https://arxiv.org/pdf/1710.09829.pdf
    for MNIST (tensor_size = (1, 1, 28, 28)) -- works for FashionMNIST

    Args:
        tensor_size: shape of tensor in BCHW
            (None/any integer >0, channels, height, width)
        n_labels: 10, MNIST/FashionMNIST
        primary_n_capsules: number of capsules in primary capsule layer
        primary_capsule_length: length of each capsule in primary capsule layer
        routing_capsule_length: number of capsules in routing capsule
        routing_iterations: length of each capsule in routing capsule
        replicate_paper: replicates https://arxiv.org/pdf/1710.09829.pdf
        activation: None/relu/relu6/lklu/elu/prelu/tanh/sigm/maxo/rmxo/swish
        dropout: 0. - 1., default = 0.1 with dropblock=True
        normalization: None/batch/group/instance/layer/pixelwise
        pre_nm: if True, normalization -> activation -> convolution else
            convolution -> normalization -> activation

    Return:
        embedding (a torch.Tensor), rec_tensor (a torch.Tensor),
            rec_loss (a torch.Tensor)
    """
    def __init__(self,
                 tensor_size: tuple = (6, 1, 28, 28),
                 n_labels: int = 10,
                 primary_n_capsules: int = 8,
                 primary_capsule_length: int = 32,
                 routing_capsule_length: int = 16,
                 routing_iterations: int = 3,
                 replicate_paper: bool = True,
                 activation: str = "relu",
                 dropout: float = 0.,
                 normalization: str = None,
                 pre_nm: bool = False,
                 *args, **kwargs):
        super(CapsuleNet, self).__init__()

        if replicate_paper:
            primary_n_capsules = 8
            primary_capsule_length = 32
            routing_capsule_length = 16
            routing_iterations = 3
            block = Convolution
            self.InitialConvolutions = \
                Convolution(tensor_size, filter_size=9, out_channels=256,
                            strides=1, pad=False, activation="relu")
            _tensor_size = self.InitialConvolutions.tensor_size
        else:  # You can be creative!
            block = Convolution
            kwargs = {"activation": activation, "dropout": dropout,
                      "normalization": normalization, "pre_nm": pre_nm}
            tp = [Convolution(tensor_size, 5, 64, 1, False, **kwargs),
                  ResidualComplex((6,  64, 24, 24), 3, 256, 1, True, **kwargs),
                  ResidualComplex((6, 256, 24, 24), 3, 256, 1, True, **kwargs),
                  ResidualComplex((6, 256, 24, 24), 3, 256, 1, True, **kwargs),
                  ResidualComplex((6, 256, 24, 24), 3, 256, 1, True, **kwargs),
                  Convolution((6, 256, 24, 24), 5, 64, 1, False, "", **kwargs)]
            self.InitialConvolutions = nn.Sequential(tp)
            _tensor_size = self.InitialConvolutions[-1].tensor_size
        print("InitialConvolutions output size :: ", _tensor_size)

        # block can be replaced with any module available in NeuralLayers
        self.Primary = PrimaryCapsule(_tensor_size, filter_size=9,
                                      out_channels=256, strides=2,
                                      pad=False, activation="",
                                      dropout=dropout, batch_nm=False,
                                      pre_nm=False, block=block,
                                      n_capsules=primary_n_capsules,
                                      capsule_length=primary_capsule_length)
        print("Primary capsule output size :: ",
              self.Primary.tensor_size)
        self.Routing = RoutingCapsule(self.Primary.tensor_size,
                                      n_capsules=n_labels,
                                      capsule_length=routing_capsule_length,
                                      iterations=routing_iterations)

        print("Routing capsule output size :: ", self.Routing.tensor_size)
        self.Reconstruction = \
            nn.Sequential(nn.Linear(n_labels*routing_capsule_length, 512),
                          nn.ReLU(),
                          nn.Linear(512, 1024),
                          nn.ReLU(),
                          nn.Linear(1024, int(np.prod(tensor_size[1:]))))

        self.tensor_size = (6, n_labels)
        self.input_tensor_size = tensor_size

    def forward(self, tensor, targets):
        tensor_size = tensor.size()

        # CapsuleNet
        tensor_deep = self.InitialConvolutions(tensor)
        tensor_primary = self.Primary(tensor_deep)
        embedding = self.Routing(tensor_primary)

        # Reconstruction (only during training)
        #   remove 'if' loop if you like to view the rec_tensor on test data
        rec_tensor = None
        rec_loss = 0.
        if self.training:
            identity = torch.eye(self.tensor_size[1])
            if targets.is_cuda:
                identity = identity.cuda()
            onehot_targets = identity.index_select(dim=0,
                                                   index=targets.view(-1))

            rec_tensor = (embedding *
                          onehot_targets[:, :, None]).view(tensor_size[0], -1)
            rec_tensor = torch.tanh(self.Reconstruction(rec_tensor))
            rec_loss = F.mse_loss(rec_tensor.view(tensor_size[0], -1),
                                  tensor.view(tensor_size[0], -1))
            rec_tensor = rec_tensor.view(*tensor_size)

        return embedding, rec_tensor, rec_loss


# from tensormonk.layers import *
# tensor_size = (2, 1, 28, 28)
# targets = torch.LongTensor([1, 2])
# tensor = torch.rand(*tensor_size)
# test = CapsuleNet(tensor_size)
# test(tensor, targets)[2]
