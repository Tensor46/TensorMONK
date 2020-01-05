""" TensorMONK :: data :: SuperResolutionData """

__all__ = ["SuperResolutionData"]

import os
from PIL import Image as ImPIL
from torchvision import transforms


class SuperResolutionData(object):
    r"""Super-Resolution train and test data.

    Train data:
        DIV2K
        Flickr2K

    Test data:
        BSDS100
        Historical
        Manga109
        Set5
        Set14
        Urban100

    Args:
        path (str): path to data folder, when not available it downloads
            automatically (DIV2K & Flickr2K takes longer).

        t_size (tuple, optional): BCHW or HW of low resolution image.
            default: (1, 3, 32, 32)

        n_upscale (int, optional): upscales required on low to generate high
            resolution image. HR-height = LR-height * (2^n_upscale).
            default: 2

        test (bool, optional): When True, loads test data.
            default: False

        add_flickr2k (bool, optional): When True, adds Flickr2K to training.
            default: True
    """

    def __init__(self,
                 path: str = "../data/sr_data",
                 t_size: tuple = (1, 3, 32, 32),
                 n_upscale: int = 2,
                 test: bool = False,
                 add_flickr2k: bool = True):

        if not isinstance(path, str):
            raise TypeError("SuperResolutionData: path must be str.")
        if not os.path.isdir(path):
            raise ValueError("SuperResolutionData: path is not valid dir.")
        if not isinstance(t_size, tuple):
            raise TypeError("SuperResolutionData: t_size must be tuple.")
        if not (len(t_size) == 2 or len(t_size) == 4):
            raise ValueError("SuperResolutionData: len(t_size) != 2/4.")
        if not isinstance(n_upscale, int):
            raise TypeError("SuperResolutionData: n_upscale must be int.")
        if not (n_upscale >= 1):
            raise ValueError("SuperResolutionData: n_upscale must be >= 1.")
        if not isinstance(test, bool):
            raise TypeError("SuperResolutionData: test must be bool.")
        if not isinstance(add_flickr2k, bool):
            raise TypeError("SuperResolutionData: add_flickr2k must be bool.")

        self.__dict__.update(locals())
        self.dataset = []
        if self.test:
            self.get_test()
        else:
            self.get_div2k()
            if add_flickr2k:
                self.get_flickr2k()

        sz = (t_size[-2] * (2 ** n_upscale), t_size[-1] * (2 ** n_upscale))
        if self.test:
            self.transforms = transforms.Compose([
                ImPIL.open,
                transforms.Resize(sz, interpolation=ImPIL.BICUBIC)])
        else:
            self.transforms = transforms.Compose([
                ImPIL.open,
                transforms.RandomResizedCrop(sz, scale=(0.5, 1.0),
                                             interpolation=ImPIL.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip()])
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        image = self.dataset[index % len(self.dataset)]
        hr = self.transforms(image)
        lr = hr.resize(reversed(self.t_size[-2:]), ImPIL.BICUBIC)
        # return hr, lr
        hr = self.to_tensor(hr).add_(-0.5).div_(0.25)
        lr = self.to_tensor(lr).add_(-0.5).div_(0.25)
        return hr, lr

    def get_div2k(self):
        r"""DIV2K dataset.

        Paper: Ntire 2017 challenge on single image super-resolution: Dataset
               and study
        URL:   http://www.vision.ee.ethz.ch/~timofter/publications
               /Agustsson-CVPRW-2017.pdf
        """
        import wget
        path = os.path.abspath(self.path)
        urlpath = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
        zippath = os.path.join(path, "DIV2K_train_HR.zip")
        imgpath = os.path.join(path, "DIV2K_train_HR")

        if not (os.path.isfile(zippath) or
                os.path.isdir(imgpath)):
            print(" ... downloading div2k")
            wget.download(urlpath, path, bar=wget.bar_adaptive)
        if os.path.isfile(zippath):
            if not os.path.isdir(imgpath):
                print("... unzipping")
                os.system("unzip {} -d {}".format(zippath, path))

        for x in next(os.walk(imgpath))[-1]:
            if x.endswith((".png", ".jpg", ".jpeg")):
                self.dataset.append(os.path.join(imgpath, x))

    def get_flickr2k(self):
        r"""Flickr2K dataset.

        Paper: Ntire 2017 challenge on single image super-resolution: Methods
               and results
        URL:   http://www.vision.ee.ethz.ch/~timofter/publications
               /Timofte-CVPRW-2017.pdf
        """
        import wget
        path = os.path.abspath(self.path)
        urlpath = "http://cv.snu.ac.kr/research/EDSR/Flickr2K.tar"
        zippath = os.path.join(path, "Flickr2K.tar")
        imgpath = os.path.join(path, "Flickr2K")

        if not (os.path.isfile(zippath) or
                os.path.isdir(imgpath)):
            print(" ... downloading flickr2k")
            wget.download(urlpath, path, bar=wget.bar_adaptive)
        if os.path.isfile(zippath):
            if not os.path.isdir(imgpath):
                print("... unzipping")
                os.system("tar -xf {} -C {}".format(zippath, path))

        for x in next(os.walk(imgpath + "/Flickr2K_HR"))[-1]:
            if x.endswith((".png", ".jpg", ".jpeg")):
                self.dataset.append(os.path.join(imgpath + "/Flickr2K_HR", x))

    def get_test(self):
        r"""Set5, Set14, BSDS100, Urban100, Manga109, Historical dataset.

        Paper: Single image super-resolution from transformed self-exemplars
        URL:   https://www.cv-foundation.org/openaccess/content_cvpr_2015
               /papers/Huang_Single_Image_Super-Resolution_2015_CVPR_paper.pdf
        """
        import wget
        path = os.path.abspath(self.path)
        urlpath = ("http://vllab.ucmerced.edu/wlai24/LapSRN/results"
                   "/SR_testing_datasets.zip")
        zippath = os.path.join(path, "SR_testing_datasets.zip")
        imgpath = os.path.join(path, "SR_testing_datasets")
        if not (os.path.isfile(zippath) or
                os.path.isdir(imgpath)):
            print(" ... downloading urban100")
            wget.download(urlpath, path, bar=wget.bar_adaptive)
        if os.path.isfile(zippath) and not os.path.isdir(imgpath):
            print("... unzipping")
            os.system("unzip {} -d {}".format(zippath, path))

        for r, ds, fs in os.walk(imgpath):
            for f in fs:
                if f.endswith((".png", ".jpg", ".jpeg")):
                    self.dataset.append(os.path.join(r, f))
