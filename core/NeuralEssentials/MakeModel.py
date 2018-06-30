""" TensorMONK's :: NeuralEssentials                                         """

import os
import numpy as np
from .BaseModel import BaseModel
from .CudaModel import CudaModel
from .LoadModel import LoadModel
import torch
is_cuda = torch.cuda.is_available()
#==============================================================================#


def MakeModel(file_name, tensor_size, n_labels,
              embedding_net, embedding_net_kwargs,
              loss_net=None, loss_net_kwargs={},
              default_gpu=0, gpus=1, ignore_trained=False):
    Model = BaseModel()
    Model.file_name = file_name
    print("...... making PyTORCH model!")
    embedding_net_kwargs["tensor_size"] = tensor_size
    Model.netEmbedding = CudaModel(is_cuda, gpus, embedding_net, embedding_net_kwargs)
    loss_net_kwargs["tensor_size"] = Model.netEmbedding.tensor_size
    loss_net_kwargs["n_labels"] = n_labels
    if loss_net is not None:
        Model.netLoss = CudaModel(is_cuda, gpus, loss_net, loss_net_kwargs)

    if os.path.isfile(Model.file_name+("" if Model.file_name.endswith(".t7") else ".t7")) and not ignore_trained:
        print("...... loading pretrained Model!")
        Model = LoadModel(Model)
    if is_cuda and gpus > 0:
        if gpus == 1:
            torch.cuda.set_device(default_gpu)
        Model.netEmbedding = Model.netEmbedding.cuda()
        if Model.netLoss is not None:
            Model.netLoss = Model.netLoss.cuda()
    print(" --- Total parameters in netEmbedding :: {} ---".format(np.sum([p.cpu().data.numel() \
          for p in Model.netEmbedding.parameters()])))
    if Model.netLoss is not None:
        print(" --- Total parameters in netLoss      :: {} ---\n".format(np.sum([p.cpu().data.numel() \
              for p in Model.netLoss.parameters()])))
    Model.is_cuda = is_cuda
    return Model
#==============================================================================#


def MakeCNN(file_name, tensor_size, n_labels,
              embedding_net, embedding_net_kwargs,
              loss_net, loss_net_kwargs,
              default_gpu=0, gpus=1, ignore_trained=False):
    Model = BaseModel()
    Model.file_name = file_name
    print("...... making PyTORCH model!")
    embedding_net_kwargs["tensor_size"] = tensor_size
    Model.netEmbedding = CudaModel(is_cuda, gpus, embedding_net, embedding_net_kwargs)
    loss_net_kwargs["tensor_size"] = Model.netEmbedding.tensor_size
    loss_net_kwargs["n_labels"] = n_labels
    Model.netLoss = CudaModel(is_cuda, gpus, loss_net, loss_net_kwargs)

    if os.path.isfile(Model.file_name+("" if Model.file_name.endswith(".t7") else ".t7")) and not ignore_trained:
        print("...... loading pretrained Model!")
        Model = LoadModel(Model)
    if is_cuda:
        if gpus == 1:
            torch.cuda.set_device(default_gpu)
        Model.netEmbedding = Model.netEmbedding.cuda()
        Model.netLoss = Model.netLoss.cuda()
    print(" --- Total parameters in netEmbedding :: {} ---".format(np.sum([p.cpu().data.numel() \
          for p in Model.netEmbedding.parameters()])))
    print(" --- Total parameters in netLoss      :: {} ---\n".format(np.sum([p.cpu().data.numel() \
          for p in Model.netLoss.parameters()])))
    Model.is_cuda = is_cuda
    return Model
#==============================================================================#


def MakeAE(file_name, tensor_size, n_labels,
           autoencoder_net, autoencoder_net_kwargs,
           default_gpu=0, gpus=1, ignore_trained=False):
    Model = BaseModel()
    Model.file_name = file_name
    print("...... making PyTORCH model!")
    autoencoder_net_kwargs["tensor_size"] = tensor_size
    Model.netAE = CudaModel(is_cuda, gpus, autoencoder_net, autoencoder_net_kwargs)

    if os.path.isfile(Model.file_name+("" if Model.file_name.endswith(".t7") else ".t7")) and not ignore_trained:
        print("...... loading pretrained Model!")
        Model = LoadModel(Model)
    if is_cuda:
        if gpus == 1:
            torch.cuda.set_device(default_gpu)
        Model.netAE = Model.netAE.cuda()
    print(" --- Total parameters in netAE :: {} ---".format(np.sum([p.cpu().data.numel() \
          for p in Model.netAE.parameters()])))
    Model.is_cuda = is_cuda
    return Model
