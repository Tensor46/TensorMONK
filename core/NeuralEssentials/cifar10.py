""" TensorMONK's :: NeuralEssentials                                         """

import torch
import torchvision
import torchvision.transforms as transforms
# ============================================================================ #


def CIFAR10(data_path="./data/CIFAR10", tensor_size = (6, 3, 32, 32), BSZ=64, cpus=4,
            normalize_01=False):
    n_labels = 10
    if normalize_01:
        transform = transforms.Compose([transforms.ColorJitter(.5, .5, .2),
                                        transforms.RandomGrayscale(p=.16),
                                        transforms.RandomVerticalFlip(p=0.46),
                                        transforms.ToTensor(),])
    else:
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                             (0.2023, 0.1994, 0.2010)), ])
    dataset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True,
                                           transform=transform)
    trDataLoader = torch.utils.data.DataLoader(dataset, batch_size=BSZ, shuffle=True, num_workers=cpus)

    dataset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=False, transform=transform)
    teDataLoader = torch.utils.data.DataLoader(dataset, batch_size=BSZ, shuffle=False, num_workers=cpus)

    return trDataLoader, teDataLoader, n_labels
