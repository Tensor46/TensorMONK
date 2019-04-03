""" TensorMONK :: data :: PascalVOC """

import torch
import numpy as np
import PIL.Image as ImPIL
import xml.etree.ElementTree as ET
from ..utils import PillowUtils, ObjectUtils


VOC_LABELS = ("BACKGROUND", "aeroplane", "bicycle", "bird", "boat", "bottle",
              "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
              "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
              "train", "tvmonitor")


class PascalVOC(object):

    def __init__(self, path: str = "../data/VOCdevkit/VOC2012",
                 tensor_size: tuple = (1, 3, 320, 320),
                 train: bool = True, retain_difficult: bool = False,
                 **kwargs):

        assert tensor_size[2] == 300 or tensor_size[2] == 320
        self.t_size = tensor_size
        self.train = train
        self.retain_difficult = retain_difficult

        with open(path + "/ImageSets/Main/" +
                  ("trainval.txt" if train else "val.txt")) as txt:
            file_names = txt.readlines()
        file_names = [x.strip() for x in file_names]

        # read all files
        self.images = [path + "/JPEGImages/" + x + ".jpg" for x in file_names]
        self.xmls = [path + "/Annotations/" + x + ".xml" for x in file_names]

        # random brightness, contrast, saturation, hue and grey transformations
        from torchvision import transforms
        self.random_transforms = transforms.RandomApply(
            [transforms.ColorJitter(.6, .6, .6, 0.36),
             transforms.RandomGrayscale(p=0.36)], p=0.9)
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image, xml = self.images[idx], self.xmls[idx]
        image = ImPIL.open(image).convert("RGB")
        np_labels, ltrb_boxes = self.parse_xml(xml, self.retain_difficult)
        if self.train:
            # image augmentations that require box adjustment
            image, ltrb_boxes = PillowUtils.random_pad(image, ltrb_boxes, 0.6)
            image, ltrb_boxes = PillowUtils.random_crop(image, ltrb_boxes)
            image, ltrb_boxes = PillowUtils.random_flip(image, ltrb_boxes, 0.5)
            # image augmentations
            image = self.random_transforms(image)

        # resize and adjust boxes
        image, ltrb_boxes = PillowUtils.to_pil(image, self.t_size, ltrb_boxes)

        # normalize the boxes
        ltrb_boxes = ObjectUtils.pixel_to_norm01(ltrb_boxes, self.t_size[3],
                                                 self.t_size[2], "numpy")

        # to torch.Tensor's
        ltrb_boxes, labels, image = torch.from_numpy(ltrb_boxes).float(), \
            torch.from_numpy(np_labels).long(), self.to_tensor(image).float()
        return image, ltrb_boxes, labels

    @staticmethod
    def parse_xml(file_name, retain_difficult):
        def np_box(x):
            return map(float, [x.find("xmin").text, x.find("ymin").text,
                               x.find("xmax").text, x.find("ymax").text])
        content = ET.parse(file_name).findall("object")

        labels, ltrb_boxes = [], []
        for x in content:
            if x.find("difficult").text == "1" and not retain_difficult:
                continue
            labels += [x.find("name").text.lower().strip()]
            ltrb_boxes += [list(np_box(x.find("bndbox")))]

        ltrb_boxes = np.array(ltrb_boxes).astype(np.float32) - 1
        np_labels = np.array([VOC_LABELS.index(x) for x in labels])
        return np_labels.astype(np.int64), ltrb_boxes
