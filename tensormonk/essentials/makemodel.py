""" TensorMONK's :: essentials """

import os
import torch
from .cudamodel import CudaModel
is_cuda = torch.cuda.is_available()


class BaseModel:
    netEmbedding = None
    netLoss = None
    netAdversarial = None
    meterTop1 = []
    meterTop5 = []
    meterLoss = []
    meterTeAC = []
    meterSpeed = []
    meterIterations = 0
    fileName = None
    isCUDA = False


def MakeModel(file_name,
              tensor_size,
              n_labels,
              embedding_net,
              embedding_net_kwargs={},
              loss_net=None,
              loss_net_kwargs: dict = {},
              default_gpu: int = 0,
              gpus: int = 1,
              ignore_trained: bool = False,
              old_weights: bool = False):
    r"""Using BaseModel structure build CudaModel's for embedding_net and
    loss_net.

    Args:
        file_name: full path + name of the file to save model
        tensor_size: input tensor size to build embedding_net
        n_labels: number of labels for loss_net, use None when loss_net is None
        embedding_net: feature embedding network that requires input of size
            tensor_size
        embedding_net_kwargs: all the additional kwargs required to build the
            embedding_net, default = {}
        loss_net: loss network. default = None
        loss_net_kwargs: all the additional kwargs required to build the
            loss_net. When loss_net_kwargs["tensor_size"] and
            loss_net_kwargs["n_labels"] are not available uses the
            Model.netEmbedding.tensor_size and n_labels
        default_gpu: gpus used when gpus = 1 and cuda is available, default = 0
        gpus: numbers of gpus used for training - used by CudaModel for
            multi-gpu support, default = 1
        ignore_trained: when True, ignores the trained model
        old_weights: converts old_weights from NeuralLayers.Linear and
            NeuralLayers.CenterLoss to new format, default = False

    Return:
        BaseModel with networks
    """
    Model = BaseModel()
    Model.file_name = file_name

    print("...... making PyTORCH model!")
    embedding_net_kwargs["tensor_size"] = tensor_size
    Model.netEmbedding = CudaModel(is_cuda, gpus,
                                   embedding_net, embedding_net_kwargs)

    if "tensor_size" not in loss_net_kwargs.keys():
        loss_net_kwargs["tensor_size"] = Model.netEmbedding.tensor_size
    if "n_labels" not in loss_net_kwargs.keys():
        loss_net_kwargs["n_labels"] = n_labels
    if loss_net is not None:
        Model.netLoss = CudaModel(is_cuda, gpus, loss_net, loss_net_kwargs)

    file_name = Model.file_name
    if not file_name.endswith(".t7"):
        file_name += ".t7"

    if os.path.isfile(file_name) and not ignore_trained:
        print("...... loading pretrained Model!")
        Model = LoadModel(Model, old_weights)

    for x in dir(Model):  # count parameters
        if x.startswith("net") and getattr(Model, x) is not None:
            count = 0
            for p in getattr(Model, x).parameters():
                count += p.cpu().data.numel()
            print(" --- Total parameters in {} :: {} ---".format(x, count))

    if is_cuda and gpus > 0:  # cuda models
        if gpus == 1:
            torch.cuda.set_device(default_gpu)
        for x in dir(Model):
            if x.startswith("net") and getattr(Model, x) is not None:
                eval("Model." + x + ".cuda()")
        Model.is_cuda = is_cuda
    return Model


def convert(state_dict):
    new_state_dict = {}
    for name in state_dict.keys():
        # fix for new Linear layer
        new_name = name.replace(".Linear.weight", ".weight")
        new_name = new_name.replace(".Linear.bias", ".bias")
        # fix for new Center Loss
        new_name = name.replace(".center.centers", ".centers")

        new_state_dict[new_name] = state_dict[name]
    return new_state_dict


def LoadModel(Model, old_weights):
    r""" Loads the following from Model.file_name:
        1. state_dict of any value whose key starts with "net" & value != None
        2. values of keys that starts with "meter".
    """
    file_name = Model.file_name
    if not file_name.endswith(".t7"):
        file_name += ".t7"
    dict_stuff = torch.load(file_name)

    for x in dir(Model):
        if x.startswith("net") and getattr(Model, x) is not None:
            if old_weights:
                dict_stuff[x] = convert(dict_stuff[x])
            eval("Model." + x + '.load_state_dict(dict_stuff["'+x+'"])')
        if x.startswith("meter") and dict_stuff[x] is not None:
            setattr(Model, x, dict_stuff[x])
    return Model


def SaveModel(Model, remove_weight_nm=False):
    r""" Saves the following to Model.file_name:
        1. state_dict of any value whose key starts with "net" & value != None
        2. values of keys that starts with "meter".
    """
    file_name = Model.file_name
    if not file_name.endswith(".t7"):
        file_name += ".t7"
    dict_stuff = {}

    for x in dir(Model):
        if x.startswith("net") and getattr(Model, x) is not None:
            net = getattr(Model, x)
            if remove_weight_nm:
                for name, p in net.named_parameters():
                    if "weight_v" in name:
                        eval("torch.nn.utils.remove_weight_norm(net." +
                             name.rstrip(".weight_v") + ", 'weight')")
            state_dict = net.state_dict()
            for y in state_dict.keys():
                state_dict[y] = state_dict[y].cpu()
            dict_stuff.update({x: state_dict})
        if x.startswith("meter"):
            dict_stuff.update({x: getattr(Model, x)})

    torch.save(dict_stuff, file_name)
