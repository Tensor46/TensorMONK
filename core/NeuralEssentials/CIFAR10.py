""" TensorMONK's :: NeuralEssentials                                         """

import os
import torch
import torchvision
import torchvision.transforms as DataMods
# ============================================================================ #


def CIFAR10(data_path="./data/CIFAR10", tensor_size = (6, 3, 32, 32), BSZ=64, cpus=4):
    n_labels = 10
    dataset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True,
                transform=DataMods.Compose([DataMods.ToTensor(), DataMods.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),]))
    trDataLoader = torch.utils.data.DataLoader(dataset, batch_size=BSZ, shuffle=True, num_workers=cpus)

    dataset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=False,
                transform=DataMods.Compose([DataMods.ToTensor(), DataMods.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),]))
    teDataLoader = torch.utils.data.DataLoader(dataset, batch_size=BSZ, shuffle=False, num_workers=cpus)

    return trDataLoader, teDataLoader, n_labels
