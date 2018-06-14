""" TensorMONK's :: NeuralEssentials                                         """

import os
import copy
import torch
#==============================================================================#


def SaveModel(Model):
    what2save = [x for x in dir(Model) if x.startswith("net") or x.startswith("meter")]
    dict_stuff = {}
    file_name = Model.file_name
    if not file_name.endswith(".t7"):
        file_name += ".t7"

    for x in what2save:
        if "net" in x.lower() and getattr(Model, x) is not None:
            eval("Model."+x+".zero_grad()")
            this_network  = getattr(Model, x).state_dict()
            # this_network  = copy.deepcopy(getattr(Model, x)).state_dict()
            for y in this_network.keys():
                this_network[y] = this_network[y].cpu()
            dict_stuff.update({x : this_network})
        else:
            dict_stuff.update({x : getattr(Model, x)})
    torch.save(dict_stuff, file_name)
