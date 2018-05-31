""" tensorMONK's :: neuralEssentials                                         """

import os
import torch
import torchvision.datasets as DataSET
import torchvision
import torchvision.transforms as DataMods
from random import random as rand01
from PIL import Image as ImPIL
#==============================================================================#


def FolderITTR(data_path, BSZ, tensor_size=(6, 3, 28, 28), cpus=6, functions=[],
               random_flip=True):
    flip = lambda x : x.transpose(ImPIL.FLIP_LEFT_RIGHT) if rand01()>.5 else x
    resize = lambda x : x.resize((tensor_size[3], tensor_size[2]), ImPIL.BILINEAR)
    mods = list(functions) + [resize, ] + ([flip,] if random_flip else []) + [DataMods.ToTensor(),]
    data = DataSET.ImageFolder(data_path, DataMods.Compose(mods))
    data_loader = torch.utils.data.DataLoader(data, batch_size=BSZ, shuffle=True, num_workers=cpus)
    n_labels = len(next(os.walk(data_path))[1])
    return (data_loader, n_labels)
