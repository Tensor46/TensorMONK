""" TensorMONK :: data """

import os
from torch.utils.data import DataLoader
import multiprocessing
from torchvision import datasets
from torchvision.transforms import RandomApply, ColorJitter, \
    RandomResizedCrop, RandomRotation, Compose, ToTensor, Normalize, \
    RandomHorizontalFlip, Resize


def DataSets(dataset: str = "MNIST",
             data_path: str = "../data",
             tensor_size: tuple = None,
             n_samples: int = 64,
             cpus: int = multiprocessing.cpu_count(),
             augment: bool = False,
             normalize: bool = True):
    r"""Train, validation and test dataset iterator for
    MNIST/FashionMNIST/CIFAR10/CIFAR100

    Args:
        dataset (string): name of dataset, MNIST/FashionMNIST/CIFAR10/CIFAR100
        data_path (string): path to dataset, default = "../data"
        tensor_size (list/tuple, optional): BCHW of output, default = based on
            dataset
        n_samples (int): samples per batch
        cpus (int, optional): numbers of cpus used by dataloader,
            default = cpu_count
        augment (bool, optional): when True, does color jitter, random crop
            and random rotation
        normalize (bool, optional): When True, uses default mean and std.
            When False, out tensor values range between 0-1

    Return:
        train data iterator, test data iterator and n_labels
    """

    dataset = dataset.lower()
    assert dataset in ["mnist", "fashionmnist", "cifar10", "cifar100"],\
        "DataSets: available options MNIST/FashionMNIST/CIFAR10/CIFAR100"

    # data path
    folder = os.path.join(data_path, dataset)
    if not os.path.isdir(folder):
        os.mkdir(folder)

    basics = [ToTensor()]
    if dataset in ["mnist", "fashionmnist", "cifar10", "cifar100"]:
        n_labels = 10
        if dataset == "mnist":
            loader = datasets.MNIST
            if tensor_size is None:
                tensor_size = (1, 1, 28, 28)
            elif tensor_size[2] != 28 or tensor_size[3] != 28:
                basics = [Resize(tensor_size[2:][::-1])] + basics
            if normalize:
                basics += [Normalize((0.1307,), (0.3081,))]
        if dataset == "fashionmnist":
            loader = datasets.FashionMNIST
            if tensor_size is None:
                tensor_size = (1, 1, 28, 28)
            elif tensor_size[2] != 32 or tensor_size[3] != 32:
                basics = [Resize(tensor_size[2:][::-1])] + basics
            if normalize:
                basics += [Normalize((0.5,), (0.5,))]
        if dataset == "cifar10":
            loader = datasets.CIFAR10
            if tensor_size is None:
                tensor_size = (1, 3, 32, 32)
            elif tensor_size[2] != 32 or tensor_size[3] != 32:
                basics = [Resize(tensor_size[2:][::-1])] + basics
            if normalize:
                basics += [Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))]
        if dataset == "cifar100":
            n_labels = 100
            loader = datasets.CIFAR100
            if tensor_size is None:
                tensor_size = (1, 3, 32, 32)
            elif tensor_size[2] != 32 or tensor_size[3] != 32:
                basics = [Resize(tensor_size[2:][::-1])] + basics
            if normalize:
                basics += [Normalize((0.5071, 0.4867, 0.4408),
                                     (0.2675, 0.2565, 0.2761))]
    elif dataset == "emnist":
        # TODO
        pass

    # test data
    teData = loader(root=folder, train=False, download=True,
                    transform=Compose(basics))
    teData = DataLoader(teData, batch_size=n_samples,
                        shuffle=False, num_workers=cpus)
    # validation data
    vaData = None

    # train data
    if augment:
        h, w = tensor_size[2:]
        few_augs = [ColorJitter(.5, .5, .2),
                    RandomResizedCrop((w, h), scale=(0.7, 1.0),
                                      ratio=(0.75, 1.33), interpolation=2),
                    RandomRotation(16, resample=False)]
        if dataset not in ["mnist", "fashionmnist"]:
            few_augs += [RandomHorizontalFlip(p=0.25)]
        basics = [RandomApply(few_augs, p=0.8)] + basics

    trData = loader(root=folder, train=True, download=False,
                    transform=Compose(basics))
    trData = DataLoader(trData, batch_size=n_samples,
                        shuffle=True, num_workers=cpus)
    return trData, vaData, teData, n_labels, tensor_size


# import torchvision.utils as tutils
# tr, va, te, n_labels, tensor_size = DataSets(dataset="mnist")
# for x, y in tr:
#     break
# from torchvision import utils as tutils
# tutils.save_image(x, "./test.png")
