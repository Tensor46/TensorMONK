""" TensorMONK :: utils """

import torch
import numpy as np
import PIL.Image as ImPIL
from random import random
from PIL import ImageDraw, ImageOps
from torchvision import transforms


class PillowUtils:
    tensor_to_pil = transforms.ToPILImage()

    @staticmethod
    def to_pil(image, t_size: tuple = None, ltrb_boxes: np.ndarray = None):
        """
        Converts file_name or np.ndarray or 3D torch.Tensor to pillow image.
        Adjusts the ltrb_boxes when ltrb_boxes are provided along with t_size.

        Args:
            image ({str/np.ndarray/torch.Tensor}): input
            t_size (tuple, optional): BCHW (Ex: (None, 3, 60, 60)) used to
                convert to grey scale or resize
            ltrb_boxes (np.ndarray, optional): Must be pixel locations in
                (left, top, right, bottom).
        """
        if ImPIL.isImageType(image):
            o = image
        elif isinstance(image, str):
            o = ImPIL.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            o = ImPIL.fromarray(image)
        elif isinstance(image, torch.Tensor):
            o = PillowUtils.tensor_to_pil(image)
        else:
            raise TypeError("to_pil: image must be str/np.ndarray/torch.Tensor"
                            ": {}".format(type(image).__name__))
        if t_size is not None and len(t_size) == 4:
            w, h = o.size
            if t_size[1] == 1:
                o = o.convert("L")
            if not (t_size[2] == w and t_size[3] == h):
                o = o.resize((t_size[3], t_size[2]), ImPIL.BILINEAR)
            if ltrb_boxes is not None:
                ltrb_boxes[:, 0::2] *= t_size[3] / w
                ltrb_boxes[:, 1::2] *= t_size[2] / h
                return o, ltrb_boxes
        return o

    @staticmethod
    def random_pad(image: ImPIL.Image, ltrb_boxes: np.ndarray = None,
                   pad: float = 0.36, fill: int = 0):
        """
        Does random padding with fill value. When ltrb_boxes is not None, the
        locations are adjusted to new image.

        Args:
            image ({str/np.ndarray/torch.Tensor}): input
            ltrb_boxes (np.ndarray, optional): Must be pixel locations in
                (left, top, right, bottom).
            pad (float, optional): (w + h) / 2 * pad is the max padding, must
                be in the range of 0-1, default=0.2
            fill (int, optional): used to fill the extra region, default=0

        Return:
            (image, ltrb_boxes) or image
        """
        # random padding
        w, h = image.size
        in_pixels = int((w + h)*0.5 * pad * random())
        image = ImageOps.expand(image, border=in_pixels, fill=fill)
        if ltrb_boxes is None:
            return image
        ltrb_boxes += in_pixels
        return image, ltrb_boxes

    @staticmethod
    def random_crop(image: ImPIL.Image, ltrb_boxes: np.ndarray = None,
                    retain: float = 0.5, maintain_aspect_ratio: bool = False):
        """
        Does random crop and adjusts the boxes when not None (crops are
        limited such that the objects are within the cropped image)!

        Args:
            image (pil-image): pillow input image
            ltrb_boxes (np.ndarray, optional): object locations in ltrb format.
                Requires 2D-array, with rows of (left, top, right, bottom) in
                pixels.
            retain (float, optional): retain pecentage (0.2-0.8) - ignored when
                boxes is not None. default=0.5
            maintain_aspect_ratio (bool, optional): Retains aspect ratio when
                True

        Return:
            image, boxes
        """
        w, h = image.size
        x_, y_, _x, _y = np.random.rand(4).tolist()
        if ltrb_boxes is None:
            retain = max(0.2, retain)
            retain = min(1.0, retain)
            x_, _x = int(w * x_*(1-retain)*.5), int(w * (1 - _x*(1-retain)*.5))
            y_, _y = int(h * y_*(1-retain)*.5), int(h * (1 - _y*(1-retain)*.5))
        else:
            x_ = int(x_*ltrb_boxes[:, 0].min())
            y_ = int(y_*ltrb_boxes[:, 1].min())
            _x = w - int(_x * (w - ltrb_boxes[:, 2].max()))
            _y = h - int(_y * (h - ltrb_boxes[:, 3].max()))
        if maintain_aspect_ratio and (0.9 < (_x - x_)/(_y - y_) < 1.1):
            if (_x - x_)/(_y - y_) > 1:
                extra = (_x - x_) - (_y - y_)
                y_ -= extra//2
                _y += extra//2
            else:
                extra = (_y - y_) - (_x - x_)
                x_ -= extra//2
                _x += extra//2
        image = image.crop((x_, y_, _x, _y))
        if ltrb_boxes is None:
            return image
        ltrb_boxes[:, 0::2] -= x_
        ltrb_boxes[:, 1::2] -= y_
        return image, ltrb_boxes

    @staticmethod
    def random_flip(image: ImPIL.Image, ltrb_boxes: np.ndarray = None,
                    probability: float = 0.75, vertical_flip: bool = True):
        """ Does random flip and adjusts the boxes when not None!

        Args:
            image (pil-image): pillow input image
            ltrb_boxes (np.ndarray, optional): object locations in ltrb format.
                Requires 2D-array, with rows of (left, top, right, bottom) in
                pixels.
            probability (float, optional): proability of flip, default=0.75
            vertical_flip (bool, optional): When True does, vertical flip,
                default=True

        Return:
            image, boxes
        """
        if random() < probability:
            w, h = image.size
            prob = random()
            if not (0.33 <= prob <= 0.66):  # horizontal
                image = image.transpose(ImPIL.FLIP_LEFT_RIGHT)
                if ltrb_boxes is not None:
                    ltrb_boxes[:, 0::2] = w - ltrb_boxes[:, [2, 0]]
            if prob > 0.33:
                image = image.transpose(ImPIL.FLIP_TOP_BOTTOM)
                if ltrb_boxes is not None:
                    ltrb_boxes[:, 1::2] = h - ltrb_boxes[:, [3, 1]]
        if ltrb_boxes is None:
            return image
        return image, ltrb_boxes

    @staticmethod
    def annotate_boxes(image, ltrb_boxes, text: list = None):
        """ Annotates the boxes for visualization!

        Args:
            image ({pillow image, 3D torch.Tensor}): input image to annotate
            ltrb_boxes ({2D torch.Tensor/np.ndarray}): annotation boxes
            text (list): a list of strings to label

        Return:
            annotated pillow image
        """
        if isinstance(image, torch.Tensor):
            image = PillowUtils.tensor_to_pil(image)
        if isinstance(ltrb_boxes, torch.Tensor):
            ltrb_boxes = ltrb_boxes.data.cpu().numpy()

        _show = image.copy()
        boxes = ltrb_boxes.copy()
        if boxes.max() <= 1:
            # convert normalized ltrb_boxes to pixel locations
            boxes[:, 0::2] *= image.size[0]
            boxes[:, 1::2] *= image.size[1]

        w, h = _show.size
        boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, w)
        boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, h)
        draw = ImageDraw.Draw(_show)
        for i, x in enumerate(boxes.astype(np.int64)):
            draw.rectangle((tuple(x[:2].tolist()), tuple(x[2:].tolist())),
                           outline=(0, 255, 0))
            if text is not None:
                if isinstance(text[i], str):
                    draw.text(tuple((x[:2]).tolist()), text[i],
                              fill=(255, 0, 0))
        del draw
        return _show
