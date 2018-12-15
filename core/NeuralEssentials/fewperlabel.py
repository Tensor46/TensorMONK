""" TensorMONK's :: NeuralEssentials                                         """

import os
import sys
from random import shuffle
from functools import reduce
from PIL import Image as ImPIL
import multiprocessing
from tqdm import trange
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")
list_images = lambda folder: [os.path.join(folder, x) for x in \
    next(os.walk(folder))[2] if x.lower().endswith(IMAGE_EXTENSIONS)]


class FewPerLabel(Dataset):
    r"""Folder iterator to sample n consecutive samples per label. Batch size
    must be equal a multiplier of n to get valid sets. On a dataset with 10
    labels, with n = 2, an example batch can yeild the following labels
    0 0 2 2 6 6 1 1 6 6

    Args:
        path: full path to folders, where each folder represents a class
        tensor_size: a list/tuple of tensor shape in BCHW Ex: (None, 3, 64, 64)
        n_consecutive: delivers n_consecutive samples per label, must be >= 2
        process_image: None (reads and resizes to tensor_size) or function to
                       read and modify
        augmentations: a list/tuple of functions to augment pil image

    Returns:
        a torch.Tensor image with values in the range [0, 1] and
        torch.LongTensor label
    """
    def __init__(self, path, tensor_size, n_consecutive, process_image=None,
            augmentations=[], n_samples=int(1e6)):
        # get all folders
        if isinstance(path, str): path = [path]
        folders = []
        for p in path:
            for folder in next(os.walk(p))[1]: # only immediate folders
                folders.append(os.path.join(p, folder))
        self.folders = sorted(folders)

        # read all images -- creates a list of lists
        self.dataset = []
        self.n_per_label = []
        for i in trange(len(self.folders), desc="CouplePerClass"):
            images = list_images(folders[i])
            if len(images) > 0:
                self.dataset.append(images)
                self.n_per_label.append(len(images))

        # random list to pick labels and samples
        self.n_labels = len(self.dataset)
        self.n_consecutive = n_consecutive
        self.true_n_samples = np.sum(self.n_per_label)
        self.random_labels = self.get_random_list(self.n_labels, None,
            self.n_consecutive)

        # process_image
        self.tensor_size = tensor_size
        if process_image is None:
            process_image = lambda x: ImPIL.open(x).resize(
                (tensor_size[3], tensor_size[2]), ImPIL.BILINEAR)
        self.process_image = process_image

        # augmentations
        self.augmentations = augmentations
        self.n_samples = n_samples
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        label = self.random_labels.pop(0)
        file_name = self.dataset[label][idx % self.n_per_label[label]]
        image = self.process_image(file_name)

        for fn in self.augmentations:
            image = fn(image)

        if self.tensor_size[1] == 1:
            image = image.convert("L")

        if len(self.random_labels) == 0:
            # generates random order of labels
            self.random_labels = self.get_random_list(self.n_labels, label,
                self.n_consecutive)
        return self.to_tensor(image), label

    @staticmethod
    def get_random_list(n_labels, last=None, n_consecutive=1):
        random_labels = list(range(0, n_labels))
        shuffle(random_labels)
        if last is not None and last == random_labels[0]:
            random_labels = random_labels[1:] + random_labels[:1]
        if n_consecutive > 1:
            random_labels = [[x] * n_consecutive for x in random_labels]
            random_labels = reduce(lambda x, y: x + y, random_labels)
        return random_labels


# trData = FewPerLabel("../data/test_folders",
#     (1, 3, 128, 128), 3, process_image=None, augmentations=[], n_samples=int(1e6))
# trDataLoader = torch.utils.data.DataLoader(trData,
#     batch_size=16, shuffle=True, num_workers=multiprocessing.cpu_count())
#
# for x, y in trDataLoader:
#     break
#
# ImPIL.fromarray(x.mul(255)[0,].data.numpy().transpose(1, 2, 0).astype(np.uint8))
# ImPIL.fromarray(x.mul(255)[1,].data.numpy().transpose(1, 2, 0).astype(np.uint8))
# ImPIL.fromarray(x.mul(255)[2,].data.numpy().transpose(1, 2, 0).astype(np.uint8))
# ImPIL.fromarray(x.mul(255)[3,].data.numpy().transpose(1, 2, 0).astype(np.uint8))
# ImPIL.fromarray(x.mul(255)[4,].data.numpy().transpose(1, 2, 0).astype(np.uint8))
# ImPIL.fromarray(x.mul(255)[5,].data.numpy().transpose(1, 2, 0).astype(np.uint8))
