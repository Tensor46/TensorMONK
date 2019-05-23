""" TensorMONK :: data :: utils """

import os
import errno
import torch
import torch.nn.functional as F
from PIL import Image as ImPIL
from torchvision import transforms
import threading
import requests
DEBUG = False
_totensor = transforms.ToTensor()


def totensor(input, t_size: tuple = None):
    r"""Converts image_file or PIL image to torch tensor.

    Args:
        input (str/pil-image): full path of image or pil-image
        t_size (list, optional): tensor_size in BCHW, used to resize the input
    """
    if isinstance(input, torch.Tensor):
        if t_size is not None:
            if len(t_size) == input.dim() == 4:
                if t_size[2] != input.size(2) or t_size[3] != input.size(3):
                    input = F.interpolate(input, size=t_size[2:])
        return input

    if isinstance(input, str):
        if not os.path.isfile(input):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                    input)
        input = ImPIL.open(input).convert("RGB")

    if ImPIL.isImageType(input):
        if t_size is not None:
            if t_size[1] == 1:
                input = input.convert("L")
            if t_size[2] != input.size[1] or t_size[3] != input.size[0]:
                input = input.resize((t_size[3], t_size[2]), ImPIL.BILINEAR)
    else:
        raise TypeError("totensor: input must be str/pil-imgage: "
                        "{}".format(type(input).__name__))
    tensor = _totensor(input)
    if tensor.dim() == 2:
        tensor.unsqueeze_(0)
    return tensor


def check_download(file_name):
    try:
        ImPIL.open(file_name)
        print(" ... downloaded | existing {}".format(file_name.split("/")[-1]))
    except Exception as e:
        if os.path.isfile(file_name):
            os.remove(file_name.split("/")[-1])
        if DEBUG:
            print(e)
        print(" ... downloaded & deleted  {}".format(file_name.split("/")[-1]))


class ThreadedDownload(threading.Thread):
    def __init__(self, file_names, urls, redownload):
        super().__init__()

        # checks
        if not (isinstance(file_names, list) or isinstance(file_names, tuple)):
            raise TypeError("ThreadedDownload: file_names is not list/tuple")
        if not (isinstance(urls, list) or isinstance(urls, tuple)):
            raise TypeError("ThreadedDownload: urls is not list/tuple")
        assert len(file_names) == len(urls), \
            "ThreadedDownload: len(file_names) != len(urls)"

        self.file_names = file_names
        self.urls = urls
        self.redownload = redownload

    def run(self):
        for file_name, url in zip(self.file_names, self.urls):
            if os.path.isfile(file_name) and not self.redownload:
                continue

            with open(file_name, "wb") as f:
                try:
                    content = requests.get(url).content
                    f.write(content)
                except Exception as e:
                    if DEBUG:
                        print(e, file_name.split("/")[-1])
            check_download(file_name)


def urls_2_images(file_names, urls, num_threads=64, redownload=False):
    """
    Image downloader for datasets like:
        Im2Text: Describing Images Using 1 Million Captioned Photographs
        Flicker datasets
        ...

    Args:
        file_names (list/tuple): A list/tuple of file name with full path.
        urls (list/tuple): A list/tuple of image url's
    """
    threads = [ThreadedDownload([], [], redownload)for i in range(num_threads)]
    while urls:
        for t in threads:
            try:
                t.file_names.append(file_names.pop())
                t.urls.append(urls.pop())
            except IndexError as e:
                print(e)
                break
    threads = [t for t in threads if t.urls]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
