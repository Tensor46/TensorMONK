""" TensorMONK's :: NeuralEssentials                                         """

import os
import torch
import copy
#==============================================================================#


def LoadModel(Model):
    file_name = Model.file_name
    if not file_name.endswith(".t7"):
        file_name += ".t7"
    dict_stuff = torch.load(file_name)
    what2load = [x for x in dir(Model) if x.startswith("net") or x.startswith("meter")]
    for x in what2load:
        if "net" in x.lower() and getattr(Model, x) is not None:
            eval("Model."+x+'.load_state_dict(dict_stuff["'+x+'"])')
        else:
            if dict_stuff[x] is not None: setattr(Model,x,dict_stuff[x])
    return Model
