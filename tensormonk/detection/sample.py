""" TensorMONK's :: data :: Sample """

import os
import random
import numpy as np
from typing import Union
from PIL import Image as ImPIL
from ..utils import PillowUtils, ObjectUtils


class Sample(object):
    r"""Sample is an object that contains image path, bounding boxes and points
    for object detection tasks that can localize landmark. It can augment data
    (random 90/180/270 rotates, random pad and random cropping) during training
    -- boxes and points are adjusted accordingly. The image can be resized if
    Sample.OSIZE is initialized.


    Sample Options (are set once):
    -----------------------------
        INVALID (float): In cases where some points are not available set the
            value to float("nan"). This allows to track those points after
            augmentation (must be filtered during loss computation --
            tensormonk.loss.PointLoss automatically handles it)

            default = float("nan")

        OSIZE (tuple): (width, height) of output image, when not set returns
            image without resize along with its attributes (boxes and
            points) after augmentation.

            default = None

        RESIZE (bool): When True along with OSIZE != None will resize the image
            during augmentation and adjust the boxes and points to new image
            size.

        ROTATE_90 (bool): Enables random rotation (90/180/270)

            options = True | False
            default = True

        ROTATE_90_PROBS (tuple): Probability of ROTATE_90

            default = (0.4, 0.6, 0.8)
                40%, 20%, 20% and 20% probable to rotate 0, 90, 180, and 270
                degrees respectively

        PAD (bool): Does random padding

            options = True | False
            default = True

        PAD_PERCENTAGE (float): Maximum percentage of height and width that
            is padded.

            options = 0 < PAD_PERCENTAGE < 1
            default = 0.1

        CROP (bool): Does random cropping

            options = True | False
            default = True

        CROP_MIN_SIDE_PERCENTAGE (float): Minimum percentage of the size that
            must be retained

            options = 0 < CROP_MIN_SIDE_PERCENTAGE < 1
            default = 0.3

        CROP_MIN_OBJECT_SIDE (int): Minimum side of the object that has to be
            maintained after crop and resize. In case of multiple objects, at
            least one object will have min(w, h) >= CROP_MIN_OBJECT_SIDE

            options = 0 < CROP_MIN_OBJECT_SIDE < min(Sample.OSIZE)
            default = 16

        CROP_N_ATTEMPTS (int): Number of attempts to find random crop, when
            failed randomly selects one object and extracts a crop around it.

            options = depends on cpu (a larger number can slow down dataloader)
            default = 16

        RETAIN_AREA (float): An object is retained only if
            original area * RETAIN_AREA >= visible area after a crop

            default = 0.5



    Sample Args:
    -----------
        image (str, required): Full path to image (does not accept ndarray or
            pillow image -- larger dataset can run out of memory)

        labels (list/tuple/np.ndarray, required): labels of all the objects in
            the image. In order to use tensormonk.loss.LabelLoss use 0 for
            background.

        boxes (list/tuple/np.ndarray, required): bounding boxes of all the
            labels. Must be in pixel coordinates and ltrb form (left, top,
            right, bottom)

        points (list/tuple/np.ndarray, optional): [x, y, x, y, ...] points of
            all the bounding boxes. If points for some objects are missing use
            float("nan") and maintain all the labels to have same number of
            points. When not required use None.



    Sample Properties:
    -----------------
        image         = returns pil image (reads every time)
        image_name    = returns full image path
        labels        = returns labels (np ndarray) -- copy
        is_boxes      = returns True/False (True indicates presence of boxes)
        boxes         = returns boxes in ltrb format (np ndarray) -- copy
        boxes_ltrb    = returns boxes in ltrb format (np ndarray) -- copy
        boxes_cxcywh  = returns boxes in cxcywh format (np ndarray) -- copy
        is_points     = returns True/False (True indicates presence of points)
        points        = returns points in xy format (np ndarray) -- copy
        points_cxcy   = returns points in cxcy format (np ndarray) -- copy

    *copy -- provides a copy of np array (maintains original in case of inplace
    operations).



    Example:
    -------
    import torch
    from tensormonk.detection import Sample
    from torchvision import transforms

    Sample.OSIZE = 320, 320
    Sample.RESIZE = True
    Sample.ROTATE_90 = False
    Sample.PAD = False
    Sample.CROP = True
    Sample.CROP_MIN_SIDE_PERCENTAGE = 0.3
    Sample.CROP_MIN_OBJECT_SIDE = 16
    Sample.CROP_N_ATTEMPTS = 8

    data = [["./image1.jpg", [1], [[4, 6, 4, 6]]],
            ["./image2.jpg", [4, 6], [[4, 6, 4, 6], [2, 6, 3, 6]]]]


    class SomeDB(object):
        def __init__(self, data, osize: tuple):

            self.samples = []
            for x in data:
                self.samples.append(
                    Sample(image=x[0], labels=x[1], boxes=x[2], points=None))

            self.transforms = transforms.RandomApply(
                [transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
                 transforms.RandomGrayscale(p=0.25),
                 transforms.ToTensor()])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image, labels, boxes, points = self.samples[idx].augmented()
        tensor = self.transforms(image)
        labels = torch.from_numpy(labels).long()
        boxes = torch.from_numpy(boxes).float()
        if points is None:
            return image, labels, boxes

        points = torch.from_numpy(points).float()
        return image, labels, boxes, points


    dataset = SomeDB(data, (320, 320))

    # To check how augmentation is working use the following to visualize
    dataset.samples[0].annotate_augmented()
    # To visualize original data
    dataset.samples[0].annotate()

    """
    INVALID: float = float("nan")  # no change required
    # to resize
    OSIZE: tuple = None  # Has to be defined (width, height)
    RESIZE: bool = True
    # random rotate
    ROTATE_90: bool = True
    ROTATE_90_PROBS: tuple = (0.4, 0.6, 0.8)
    # random padding
    PAD: bool = True
    PAD_PERCENTAGE: float = 0.1
    # random crop
    CROP: bool = True
    CROP_MIN_SIDE_PERCENTAGE: float = 0.3
    CROP_MIN_OBJECT_SIDE: int = 16
    CROP_N_ATTEMPTS: int = 16
    RETAIN_AREA: float = 0.5

    def __init__(self,
                 image: str,
                 labels: np.ndarray,
                 boxes: np.ndarray,
                 points: np.ndarray = None):

        self._image = self._labels = self._boxes = self._points = None
        self._is_boxes = self._is_points = False
        self.image = image
        self.labels = labels
        self.boxes = boxes
        self.points = points

    def data(self):
        r"""Provides a copy of original data."""
        return self.image, self.labels, self.boxes, self.points

    def augmented(self):
        r"""Provides augmented data."""
        image, labels, boxes, points = self.data()
        if self.ROTATE_90:
            image, boxes, points = self._rotate_90(image, boxes, points)
        if self.PAD:
            image, boxes, points = self._pad(image, boxes, points)
        if self.CROP and self.OSIZE is not None:
            try:
                image, boxes, points = self._crop(image, boxes, points)
            except ValueError:
                pass
        if self.RESIZE and self.OSIZE is not None:
            image, boxes, points = self._resize(image, boxes, points)

        image, labels, boxes, points = self._validate_augmented(
            image, labels, boxes, points)
        return image, labels, boxes, points

    def annotate_augmented(self):
        r"""To visualize augmented data."""
        image, labels, boxes, points = self.augmented()
        return self.annotate([], image, boxes, points)

    def _validate_boxes(self, boxes: np.ndarray, w: int, h: int):
        r"""Return valid boxes."""
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        visible_boxes = boxes.copy()
        visible_boxes[:, 0::2] = visible_boxes[:, 0::2].clip(0, w)
        visible_boxes[:, 1::2] = visible_boxes[:, 1::2].clip(0, h)
        visible_area = ((visible_boxes[:, 2] - visible_boxes[:, 0]) *
                        (visible_boxes[:, 3] - visible_boxes[:, 1]))
        valid = (visible_area / (area + 1e-6)) > self.RETAIN_AREA
        return valid

    def _validate_augmented(self, image: ImPIL.Image, labels: np.ndarray,
                            boxes: np.ndarray, points: np.ndarray):
        r"""Return valid boxes, points, labels using boxes."""
        if 0.9 >= self.RETAIN_AREA >= 0.1:
            valid = self._validate_boxes(boxes, *image.size)
            labels = labels[valid]
            boxes = boxes[valid]
            points = points[valid]
        return image, labels, boxes, points

    def _rotate_90(self, image: ImPIL.Image, boxes: np.ndarray,
                   points: np.ndarray):
        r"""Does 0/90/180/270 rotation."""
        p = random.random()
        w, h = image.size

        if self.ROTATE_90_PROBS[1] >= p > self.ROTATE_90_PROBS[0]:
            image = image.transpose(ImPIL.ROTATE_90)
            if self.is_boxes:
                l, t, r, b = np.split(boxes, 4, 1)
                boxes = np.concatenate((t, w - r, b, w - l), 1)
            if self.is_points:
                x, y = np.split(points, 2, 2)
                points = np.concatenate((y, w - x), -1)
        elif self.ROTATE_90_PROBS[2] >= p > self.ROTATE_90_PROBS[1]:
            image = image.transpose(ImPIL.ROTATE_270)
            if self.is_boxes:
                l, t, r, b = np.split(boxes, 4, 1)
                boxes = np.concatenate((h - b, l, h - t, r), 1)
            if self.is_points:
                x, y = np.split(points, 2, 2)
                points = np.concatenate((h - y, x), -1)
        elif p > self.ROTATE_90_PROBS[2]:
            image = image.transpose(ImPIL.ROTATE_180)
            if self.is_boxes:
                l, t, r, b = np.split(boxes, 4, 1)
                boxes = np.concatenate((w - r, h - b, w - l, h - t), 1)
            if self.is_points:
                x, y = np.split(points, 2, 2)
                points = np.concatenate((w - x, h - y), -1)
        return image, boxes, points

    def _pad(self, image: ImPIL.Image, boxes: np.ndarray, points: np.ndarray):
        r"""Pads a maximum of Sample.PAD_PERCENTAGE * (w + h)/2 pixels."""
        w, h = image.size
        pad = min(1, int(self.PAD_PERCENTAGE * (w + h) / 2.))
        ox, oy = random.randint(0, pad), random.randint(0, pad)
        image = image.crop((-ox, -oy, w + ox, h + oy))
        if boxes is not None:
            boxes[:, 0::2] += ox
            boxes[:, 1::2] += oy
        if points is not None:
            points[:, :, 0] += ox
            points[:, :, 1] += oy
        return image, boxes, points

    def _crop(self, image: ImPIL.Image, boxes: np.ndarray, points: np.ndarray):
        r"""Does random image crop, and adjusts boxes and points."""
        (w, h), (ow, oh) = image.size, self.OSIZE
        new_points = None
        for _ in range(self.CROP_N_ATTEMPTS):
            # random side of square crop
            nw = (min(w, h) * (1 if random.random() <= 0.2 else
                               random.uniform(self.CROP_MIN_SIDE_PERCENTAGE,
                                              1.)))
            if (ow / oh) >= 1.:
                nh = nw * oh / ow
            else:
                nw, nh = nw * ow / oh, nw
            # aspect ratio variation
            p = random.random()
            if 0. < p < 0.4:
                nw = nw * random.uniform(0.8, 1.)
            elif 0.4 < p < 0.8:
                nh = nh * random.uniform(0.8, 1.)
            # random crop
            crop = random.randint(0, int(w-nw)), random.randint(0, int(h-nh))
            crop = crop + (crop[0] + nw, crop[1] + nh)
            ious, iofs = ObjectUtils.compute_iou(
                boxes, np.array(crop).reshape(-1, 4), True)
            within_the_crop = (iofs >= 0.9)
            if ~ within_the_crop.any():
                # No boxes (90% of the area) are within the crop
                continue

            # check if at least one box has minimum required size after resize
            rw = (boxes[:, 2] - boxes[:, 0]) / nw * ow
            rh = (boxes[:, 3] - boxes[:, 1]) / nh * oh
            valid_boxes = np.minimum(rw, rh) > self.CROP_MIN_OBJECT_SIDE
            if ~ (within_the_crop * valid_boxes).any():
                continue

            new_image = image.crop(crop)
            new_boxes = boxes.copy()
            new_boxes[:, 0::2] = new_boxes[:, 0::2] - crop[0]
            new_boxes[:, 1::2] = new_boxes[:, 1::2] - crop[1]
            if points is not None:
                new_points = points.copy()
                new_points[:, :, 0] = new_points[:, :, 0] - crop[0]
                new_points[:, :, 1] = new_points[:, :, 1] - crop[1]
            return new_image, new_boxes, new_points

        # if the above fails pick a random box and build a crop around it
        pick = random.randint(0, len(boxes)-1)
        anchor = boxes[pick].copy()
        # adjust crop - change w & h till the resized width makes sense
        nw = sum(anchor[2:] - anchor[:2]) / 2.
        nw = random.uniform(nw * ow / (ow * 0.8),
                            nw * ow / (self.CROP_MIN_OBJECT_SIDE * 1.25))
        nh = nw * oh / ow
        left = random.randint(max(0, int(anchor[2] - nw)), int(anchor[0]))
        top = random.randint(max(0, int(anchor[3] - nw)), int(anchor[1]))
        crop = (left, top, int(left + nw), int(top + nh))

        new_image = image.crop(crop)
        new_boxes, new_points = boxes.copy(), points.copy()
        new_boxes[:, 0::2] = new_boxes[:, 0::2] - crop[0]
        new_boxes[:, 1::2] = new_boxes[:, 1::2] - crop[1]
        if points is not None:
            new_points[:, :, 0] = new_points[:, :, 0] - crop[0]
            new_points[:, :, 1] = new_points[:, :, 1] - crop[1]
        return new_image, new_boxes, new_points

    def _resize(self, image: ImPIL.Image, boxes: np.ndarray,
                points: np.ndarray):
        r"""Resize image to Sample.OSIZE, and adjusts boxes and points."""
        (w, h), (ow, oh) = image.size, self.OSIZE
        image = image.resize(self.OSIZE, ImPIL.BILINEAR)
        if boxes is not None:
            boxes[:, 0::2] = boxes[:, 0::2] / w * ow
            boxes[:, 1::2] = boxes[:, 1::2] / h * oh
        if points is not None:
            points[:, :, 0] = points[:, :, 0] / w * ow
            points[:, :, 1] = points[:, :, 1] / h * oh
        return image, boxes, points

    @property
    def image(self):
        return ImPIL.open(self._image).convert("RGB")

    @image.setter
    def image(self, value: str):
        if not isinstance(value, str):
            raise TypeError("Sample: image must be str (full path)")
        if not os.path.isfile(value):
            raise FileNotFoundError
        self._image = value

    @property
    def image_name(self):
        return self._image

    @property
    def labels(self):
        return self._labels.copy()

    @labels.setter
    def labels(self, value):
        if isinstance(value, (int, float, list, tuple, np.ndarray)):
            self._labels = np.array(value).astype(np.int)
        else:
            raise TypeError("Sample: labels must be int/list/tuple/ndarray")

    @property
    def is_boxes(self):
        return self._is_boxes

    @property
    def boxes(self):
        return self._boxes.copy() if ~ self.is_boxes else None

    @boxes.setter
    def boxes(self, value: Union[list, tuple, np.ndarray]):
        if value is None:
            self._boxes = None
        elif isinstance(value, (list, tuple, np.ndarray)):
            if self.labels is None:
                raise ValueError("Sample: boxes requires labels")
            value = np.array(value).astype(np.float32)
            assert self.labels.size * 4 == value.size, \
                "boxes must be of shape (n_labels x 4)"
            self._boxes = value.reshape(self.labels.size, 4)
            self._is_boxes = True
        else:
            raise TypeError("Sample: boxes must be None/list/tuple/ndarray")

    @property
    def boxes_ltrb(self):
        return self.boxes

    @property
    def boxes_cxcywh(self):
        boxes = self.boxes
        if self.is_boxes:
            boxes = (boxes[:, 0::2].mean(1), boxes[:, 1::2].mean(1),
                     boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])
            boxes = np.vstack(boxes).T
        return boxes

    @property
    def is_points(self):
        return self._is_points

    @property
    def points(self):
        return self._points.copy() if ~ self.is_points else None

    @points.setter
    def points(self, value: Union[list, tuple, np.ndarray]):
        if value is None:
            self._points = None
        elif isinstance(value, (list, tuple, np.ndarray)):
            self._points = np.array(value).astype(np.float32).reshape(
                self.labels.size, -1, 2)
            self._is_points = True
        else:
            raise TypeError("Sample: points must be None/list/tuple/ndarray")

    @property
    def fake_boxes(self):
        if self.is_boxes:
            return self.boxes
        # create fake boxes for augmentation purpose
        fake_boxes = []
        for xys in self.points:
            valid = ~ np.isnan(xys).prod(1).astype(bool)
            if valid.size == xys.shape[0]:
                fake_boxes.append(np.array([0., 0., 1., 1.]))
                continue
            fake_boxes.append(np.concatenate((xys[valid].min(0) * 0.8,
                                              xys[valid].max(0) * 1.25)))
        return np.stack(fake_boxes)

    @property
    def points_cxcy(self):
        if self.is_points and self.is_boxes:
            boxes, points = self.boxes_cxcywh, self.points
            points[:, :, 0] -= boxes[:, [0]]
            points[:, :, 1] -= boxes[:, [1]]
            return points
        return None

    def annotate(self, ids: list = [], image: ImPIL.Image = None,
                 boxes: np.ndarray = None, points: np.ndarray = None):
        r"""Annotates boxes and points on the image."""
        if image is None and boxes is None:
            image = self.image
            if self.labels is None:
                return image
            boxes, points = self.boxes, self.points
            if len(ids) > 0 and isinstance(ids, (list, tuple)):
                if self.is_boxes:
                    boxes = boxes[ids]
                if self.is_points:
                    points = points[ids]
        return PillowUtils.annotate_boxes(
            image, boxes, self.avoid_nans_to_visualize(points))

    def avoid_nans_to_visualize(self, points: np.ndarray):
        r"""Removes nan's in the points."""
        if points is None:
            return None
        if np.isnan(points).any():
            if len(points[~ np.isnan(points)]) == 0:
                points = None
            else:
                points = points[~ np.isnan(points)]
        return points
