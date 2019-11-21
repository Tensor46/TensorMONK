""" TensorMONK :: utils """

import torch
import numpy as np
import PIL.Image as ImPIL
import random
from typing import Union
from PIL import ImageDraw, ImageOps
from torchvision import transforms


class PillowUtils:
    tensor_to_pil = transforms.ToPILImage()

    @staticmethod
    def to_pil(image: Union[str, ImPIL.Image, np.ndarray, torch.Tensor],
               t_size: tuple = None, ltrb_boxes: np.ndarray = None):
        r"""Converts file_name or ndarray or 3D torch.Tensor to pillow image.
        Adjusts the ltrb_boxes when ltrb_boxes are provided along with t_size.

        Args:
            image (str/np.ndarray/torch.Tensor):
                input
            t_size (tuple, optional)
                BCHW (Ex: (None, 3, 60, 60)) used to convert to grey scale or
                resize
            ltrb_boxes (np.ndarray, optional)
                Must be pixel locations in (left, top, right, bottom).
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
    def random_pad(image: ImPIL.Image,
                   ltrb_boxes: np.ndarray = None,
                   points: np.ndarray = None,
                   pad: float = 0.36, fill: int = 0):
        r"""Does random padding with fill value. When ltrb_boxes and/or points
        are not None, the boxes and/or points are adjusted to new image.

        Args:
            image ({str/np.ndarray/torch.Tensor}): input
            ltrb_boxes (np.ndarray, optional): Must be pixel locations in
                (left, top, right, bottom).
            points (np.ndarray, optional): (x, y) locations of landmarks within
                a box. Expects a 3D-array with points.shape[0] = ltrb.shape[0]
                and points.shape[2] = 2
            pad (float, optional): (w + h) / 2 * pad is the max padding, must
                be in the range of 0-1, default=0.2
            fill (int, optional): used to fill the extra region, default=0

        Return:
            (image, ltrb_boxes) or image
        """
        # random padding
        w, h = image.size
        in_pixels = int((w + h)*0.5 * pad * random.random())
        image = ImageOps.expand(image, border=in_pixels, fill=fill)
        if ltrb_boxes is None:
            return image
        ltrb_boxes += in_pixels
        if points is None:
            return image, ltrb_boxes
        points += in_pixels
        return image, ltrb_boxes, points

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
    def extend_random_crop(image: ImPIL.Image,
                           labels: np.ndarray,
                           ltrb: np.ndarray,
                           points: np.ndarray = None,
                           osize: tuple = (320, 320),
                           min_box_side: int = 30,
                           ignore_intersection: tuple = (0.5, 0.9),
                           aspect_ratio_bounds: tuple = (0.5, 2)):

        r"""Does random crop and adjusts the boxes while maintaining minimum
        box size. When not None, the points (xy locations within a bounding
        box) are adjusted.

        Args:
            image (pil-image): pillow input image
            labels (np.ndarray): labels of each box
            ltrb (np.ndarray): object locations in ltrb format.
                Requires 2D-array, with rows of (left, top, right, bottom) in
                pixels.
            points (np.ndarray, optional): (x, y) locations of landmarks within
                a box. Expects a 3D-array with points.shape[0] = ltrb.shape[0]
                and points.shape[2] = 2
            osize (tuple/list): Output image size (width, height)
            min_box_side (int): Minimum size of the box predicted by the model.
                Default = 30 -- SSD minimum size
            ignore_intersection (tuple/list of floats): avoids objects within
                the intersection range in the final crop.
            aspect_ratio_bounds (tuple/list of floats): allowed crop ratios
                given an image

        Return:
            image, labels, ltrb, points

        ** requires some speed-up
        """
        valid_points = None
        # filter boxes with negative width -- not usual but a safe check
        _valid = np.stack((ltrb[:, 2] - ltrb[:, 0], ltrb[:, 3] - ltrb[:, 1]))
        _valid = _valid.min(0) > 2
        labels, ltrb, points = labels[_valid], ltrb[_valid], points[_valid]
        w, h = image.size
        # minimum ltrb side on actual image
        mbox = min((ltrb[:, 3] - ltrb[:, 1]).min(),
                   (ltrb[:, 2] - ltrb[:, 0]).min())
        # min & max possible crop size to maintain min_box_side
        mincw = int(mbox*1.1)
        maxcw = int(min(mincw * min(osize) / min_box_side, min(w, h)))
        if mincw > maxcw:
            mincw = maxcw - 1
        # random width and height given all the above conditions
        nw = random.randint(mincw, maxcw)
        nh = random.randint(int(nw*aspect_ratio_bounds[0]),
                            int(nw*aspect_ratio_bounds[1]))
        nh = min(max(nh, int(mbox*1.1)), h)
        # find all possible boxes, given nw and nh
        all_ls, all_ts = np.arange(0, w-nw, 10), np.arange(0, h-nh, 10)
        all_ls = all_ls.repeat(len(all_ts))
        all_ts = np.tile(all_ts[None, :],
                         (len(np.arange(0, w-nw, 10)), 1)).reshape(-1)
        possible = np.concatenate((all_ls[None, ], all_ts[None, ])).T
        possible = np.concatenate([possible[:, [0]],
                                   possible[:, [1]],
                                   possible[:, [0]]+nw,
                                   possible[:, [1]]+nh], 1)

        # intersection in percentage to validate all possible boxes
        lt = np.maximum(ltrb[:, :2][:, np.newaxis],
                        possible[:, :2][np.newaxis, :])
        rb = np.minimum(ltrb[:, 2:][:, np.newaxis],
                        possible[:, 2:][np.newaxis, :])
        intersection = np.multiply(*np.split(np.clip(rb - lt, 0, None), 2, 2))
        intersection = intersection.squeeze(2)
        area = ((ltrb[:, 2] - ltrb[:, 0]) * (ltrb[:, 3] - ltrb[:, 1]))
        intersection = intersection / area[:, None]
        idx = np.where((intersection > ignore_intersection[1]).sum(0))[0]
        idx = [x for x in idx
               if not ((intersection[:, x] > ignore_intersection[0]) *
                       (intersection[:, x] < ignore_intersection[1])).any()]

        if len(idx) > 0:
            # randomly pick one valid possible box
            pick = random.randint(0, len(idx)-1)
            crop = possible[idx[pick]]
            valid = intersection[:, idx[pick]] > ignore_intersection[1]
            valid_ltrb = ltrb[valid].copy()
            if points is not None:
                valid_points = points[valid].copy()
            valid_labels = labels[valid].copy()

        else:
            # if the above fails -- fall back to a single object
            pick = random.randint(0, len(ltrb)-1)
            crop = ltrb[pick].copy()
            # adjust crop - add some width and some height
            rw_ = (crop[2] - crop[0]) * (random.random() * 0.2) + 0.05
            _rw = (crop[2] - crop[0]) * (random.random() * 0.2) + 0.05
            rh_ = (crop[3] - crop[1]) * (random.random() * 0.2) + 0.05
            _rh = (crop[3] - crop[1]) * (random.random() * 0.2) + 0.05
            crop[0] -= rw_
            crop[1] -= rh_
            crop[2] += _rw
            crop[3] += _rh
            valid_ltrb = ltrb[[pick]].copy()
            if points is not None:
                valid_points = points[[pick]].copy()
            valid_labels = labels[[pick]].copy()

        # adjust xy's
        valid_ltrb[:, 0::2] -= crop[0]
        valid_ltrb[:, 1::2] -= crop[1]
        if points is not None:
            valid_points[:, :, 0] -= crop[0]
            valid_points[:, :, 1] -= crop[1]

        image = image.crop(list(map(int, crop)))
        w, h = image.size
        image = image.resize(osize)
        valid_ltrb[:, 0::2] *= osize[0] / w
        valid_ltrb[:, 1::2] *= osize[1] / h
        if points is not None:
            valid_points[:, :, 0] *= osize[0] / w
            valid_points[:, :, 1] *= osize[1] / h
        valid_ltrb[:, 0::2] = np.clip(valid_ltrb[:, 0::2], 0, osize[0]-1)
        valid_ltrb[:, 1::2] = np.clip(valid_ltrb[:, 1::2], 0, osize[1]-1)
        if points is not None:
            valid_points[:, :, 0] = np.clip(valid_points[:, :, 0], 0, osize[0])
            valid_points[:, :, 1] = np.clip(valid_points[:, :, 1], 0, osize[1])
        return image, valid_labels, valid_ltrb, valid_points

    @staticmethod
    def random_flip(image: ImPIL.Image,
                    ltrb_boxes: np.ndarray = None,
                    points: np.ndarray = None,
                    probability: float = 0.75,
                    vertical_flip: bool = True):
        r"""Does random flip and adjusts the boxes & points when not None.

        Args:
            image (pil-image): pillow input image
            ltrb_boxes (np.ndarray, optional): object locations in ltrb format.
                Requires 2D-array, with rows of (left, top, right, bottom) in
                pixels.
            points (np.ndarray, optional): (x, y) locations of landmarks within
                a box. Expects a 3D-array with points.shape[0] = ltrb.shape[0]
                and points.shape[2] = 2
            probability (float, optional): proability of flip, default=0.75
            vertical_flip (bool, optional): When True does, vertical flip,
                default=True

        Return:
            image, boxes
        """
        ph, pv = (0.66, 0.33) if vertical_flip else (1., 1.)
        if random.random() < probability:
            w, h = image.size
            prob = random.random()
            if prob <= ph:  # horizontal
                image = image.transpose(ImPIL.FLIP_LEFT_RIGHT)
                if ltrb_boxes is not None:
                    ltrb_boxes[:, 0::2] = w - ltrb_boxes[:, [2, 0]]
                if points is not None:
                    points[:, :, 0] = w - points[:, :, 0]
            if prob >= pv:  # vertical
                image = image.transpose(ImPIL.FLIP_TOP_BOTTOM)
                if ltrb_boxes is not None:
                    ltrb_boxes[:, 1::2] = h - ltrb_boxes[:, [3, 1]]
                if points is not None:
                    points[:, :, 1] = h - points[:, :, 1]
        if ltrb_boxes is None:
            return image
        if points is None:
            return image, ltrb_boxes
        return image, ltrb_boxes, points

    @staticmethod
    def random_rotate(image: ImPIL.Image,
                      ltrb_boxes: np.ndarray = None,
                      points: np.ndarray = None,
                      probability: float = 0.5,
                      vertical_flip: bool = True):
        r"""Does random 90/-90 rotation and adjusts the boxes & points when not
        None!

        Args:
            image (pil-image): pillow input image
            ltrb_boxes (np.ndarray, optional): object locations in ltrb format.
                Requires 2D-array, with rows of (left, top, right, bottom) in
                pixels.
            points (np.ndarray, optional): (x, y) locations of landmarks within
                a box. Expects a 3D-array with points.shape[0] = ltrb.shape[0]
                and points.shape[2] = 2
            probability (float, optional): proability of flip, default=0.75

        Return:
            image, boxes
        """
        if random.random() < probability:
            w, h = image.size
            if random.random() > 0.5:  # rotate left
                image = image.rotate(90)
                if ltrb_boxes is not None:
                    ltrb_boxes = np.concatenate([ltrb_boxes[:, [1]],
                                                 w - ltrb_boxes[:, [2]],
                                                 ltrb_boxes[:, [3]],
                                                 w - ltrb_boxes[:, [0]]], 1)
                if points is not None:
                    points = np.concatenate((points[:, :, [1]],
                                             w - points[:, :, [0]]), 2)
            else:
                image = image.rotate(-90)
                if ltrb_boxes is not None:
                    ltrb_boxes = np.concatenate([h - ltrb_boxes[:, [3]],
                                                 ltrb_boxes[:, [0]],
                                                 h - ltrb_boxes[:, [1]],
                                                 ltrb_boxes[:, [2]]], 1)
                if points is not None:
                    points = np.concatenate((h - points[:, :, [1]],
                                             points[:, :, [0]]), 2)
        if ltrb_boxes is None:
            return image
        if points is None:
            return image, ltrb_boxes
        return image, ltrb_boxes, points

    @staticmethod
    def annotate_boxes(image: Union[ImPIL.Image, torch.Tensor],
                       ltrb_boxes: Union[np.ndarray, torch.Tensor, list],
                       points: Union[np.ndarray, torch.Tensor, list] = None,
                       text: list = None,
                       box_color: str = "#F1C40F",
                       point_color: str = "#00FFBB"):
        r"""Annotates the boxes and points for visualization.

        Args:
            image ({pillow image, 3D torch.Tensor}): input image to annotate
            ltrb_boxes ({2D torch.Tensor/np.ndarray}): annotation boxes
            points ({2D torch.Tensor/np.ndarray}): annotation points
            text (list): a list of strings to label

        Return:
            annotated pillow image
        """
        if isinstance(image, torch.Tensor):
            image = PillowUtils.tensor_to_pil(image)
        if isinstance(ltrb_boxes, torch.Tensor):
            ltrb_boxes = ltrb_boxes.data.cpu().numpy()
        if isinstance(points, torch.Tensor):
            points = points.data.cpu().numpy()

        assert isinstance(image, ImPIL.Image), \
            "image must be pillow image / 3D torch.Tensor"
        _show = image.copy()
        w, h = _show.size
        draw = ImageDraw.Draw(_show)

        if ltrb_boxes is not None:
            if isinstance(ltrb_boxes, (list, tuple)):
                ltrb_boxes = np.array(ltrb_boxes)
            assert isinstance(ltrb_boxes, np.ndarray), \
                "ltrb_boxes must be None/list/ndarray/torch.Tensor"

            boxes = ltrb_boxes.copy()
            if boxes.max() <= 2:
                # convert normalized ltrb_boxes to pixel locations
                boxes[:, 0::2] *= w
                boxes[:, 1::2] *= h
            boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, w)
            boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, h)
            for x in boxes.astype(np.int64):
                draw.rectangle((tuple(x[:2].tolist()), tuple(x[2:].tolist())),
                               outline=box_color)

        if points is not None:
            if isinstance(points, (list, tuple)):
                points = np.array(points)
            assert isinstance(points, np.ndarray), \
                "points must be None/list/ndarray/torch.Tensor"

            points = points.copy().reshape(-1, 2)
            if points.max() <= 2:
                points[:, 0] *= w
                points[:, 1] *= h
            points[:, 0] = np.clip(points[:, 0], 0, w)
            points[:, 1] = np.clip(points[:, 1], 0, h)
            r = 2
            for x, y in points.astype(np.int64):
                draw.ellipse((int(x) - r, int(y) - r, int(x) + r, int(y) + r),
                             fill=point_color)

        if text is not None:
            for txt in text:
                if isinstance(txt, str):
                    draw.text(tuple((x[:2]).tolist()), txt, fill="#E74C3C")
        del draw
        return _show
