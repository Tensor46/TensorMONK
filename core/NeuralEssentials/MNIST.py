""" TensorMONK's :: NeuralEssentials                                         """

import os
import torch
import torchvision
import torchvision.transforms as DataMods
# ============================================================================ #


def MNIST(data_path="./INs_OUTs/", tensor_size = (6, 1, 28, 28), BSZ=64, cpus=4):
    n_labels = 10
    dataset = torchvision.datasets.MNIST(root=data_path, train=True, download=True,
                transform=DataMods.Compose([DataMods.ToTensor(), DataMods.Normalize((0.1307,), (0.3081,)),]))
    trDataLoader = torch.utils.data.DataLoader(dataset, batch_size=BSZ, shuffle=True, num_workers=cpus)

    dataset = torchvision.datasets.MNIST(root=data_path, train=False, download=False,
                transform=DataMods.Compose([DataMods.ToTensor(), DataMods.Normalize((0.1307,), (0.3081,)),]))
    teDataLoader = torch.utils.data.DataLoader(dataset, batch_size=BSZ, shuffle=False, num_workers=cpus)

    return trDataLoader, teDataLoader, n_labels
