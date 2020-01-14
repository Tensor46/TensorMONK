""" TensorMONK's :: detection :: utils """

__all__ = ["pixel_to_norm01", "norm01_to_pixel", "auto_convert",
           "ltrb_to_cxcywh", "cxcywh_to_ltrb",
           "compute_intersection", "compute_area",
           "compute_iou", "compute_iof", "nms",
           "centers_per_layer",
           "encode_boxes", "decode_boxes",
           "encode_point", "decode_point",
           "compute_centerness", "compute_objectness",
           "ObjectUtils"]


import torch
from torch import Tensor
import numpy as np
from typing import Type, Union


def pixel_to_norm01(boxes: Type[Union[Tensor, np.ndarray]], w: int, h: int):
    r"""Normalizes bounding boxes (ltrb/cxcywh) from pixel coordinates to
    normalized 0-1 form given width (w) and height (h) of an image.
    *Makes a copy of boxes.

    Args:
        boxes (torch.Tensor / np.ndarray): Nx4 array/Tensor of boxes

        w (int): width of image feed to network

        h (int): height of image feed to network

    Return:
        Nx4 array of boxes
    """
    assert isinstance(boxes, (Tensor, np.ndarray))
    # to normalized 0-1
    boxes = boxes.clone() if isinstance(boxes, Tensor) else boxes.copy()
    boxes[:, 0::2] /= w
    boxes[:, 1::2] /= h
    return boxes


def norm01_to_pixel(boxes: Type[Union[Tensor, np.ndarray]], w: int, h: int):
    r"""Normalizes bounding boxes (ltrb/cxcywh) from normalized 0-1 form to
    pixel coordinates given width (w) and height (h) of an image.
    *Makes a copy of boxes.

    Args:
        boxes (torch.Tensor / np.ndarray): Nx4 array/Tensor of boxes

        w (int): width of image feed to network

        h (int): height of image feed to network

    Return:
        Nx4 array of boxes
    """
    # to pixel coordinates
    boxes = boxes.clone() if isinstance(boxes, Tensor) else boxes.copy()
    boxes[:, 0::2] *= w
    boxes[:, 1::2] *= h
    return boxes


def auto_convert(boxes: Type[Union[Tensor, np.ndarray]], w: int, h: int):
    r"""Normalizes bounding boxes (ltrb/cxcywh) from (pixel coordinates to
    normalized 0-1) or (normalized 0-1 form to pixel coordinates) given
    width (w) and height (h) of image """

    if boxes.max() < 2:
        # to pixel coordinates
        boxes[:, 0::2] *= w
        boxes[:, 1::2] *= h
    else:
        # to normalized 0-1
        boxes[:, 0::2] /= w
        boxes[:, 1::2] /= h
    return boxes


@torch.jit.script
def ltrb_to_cxcywh_pt(boxes: Tensor):
    boxes = (boxes.reshape(-1, 2, 2).mean(1), boxes[:, 2:] - boxes[:, :2])
    return torch.cat(boxes, 1)


def ltrb_to_cxcywh(boxes: Type[Union[Tensor, np.ndarray]]):
    r"""Converts bounding boxes from
    (left, top, right, bottom) to (center x, center y, width, height).

    Args:
        boxes (torch.Tensor / np.ndarray): Nx4 array/Tensor of boxes

    Return:
        Nx4 array of boxes
    """
    if isinstance(boxes, Tensor):
        return ltrb_to_cxcywh_pt(boxes)
    boxes = (boxes.reshape(-1, 2, 2).mean(1), boxes[:, 2:] - boxes[:, :2])
    return np.concatenate(boxes, 1)


@torch.jit.script
def cxcywh_to_ltrb_pt(boxes: Tensor):
    boxes = (boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2)
    return torch.cat(boxes, 1)


def cxcywh_to_ltrb(boxes: Type[Union[Tensor, np.ndarray]]):
    r"""Converts bounding boxes from
    (center x, center y, width, height) to (left, top, right, bottom).

    Args:
        boxes (torch.Tensor / np.ndarray): Nx4 array/Tensor of boxes

    Return:
        Nx4 array of boxes
    """
    if isinstance(boxes, Tensor):
        return cxcywh_to_ltrb_pt(boxes)
    boxes = (boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2)
    return np.concatenate(boxes, 1)


def intersection_pt(boxes1: Tensor, boxes2: Tensor):
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    intersection = torch.mul(*(rb - lt).clamp(0).split(1, dim=2))
    return intersection.squeeze_(2)


def intersection_np(boxes1: np.ndarray, boxes2: np.ndarray):
    lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    intersection = np.multiply(*np.split(np.clip(rb - lt, 0, None), 2, 2))
    return intersection.squeeze(2)


def compute_intersection(boxes1: Type[Union[Tensor, np.ndarray]],
                         boxes2: Type[Union[Tensor, np.ndarray]]):
    r"""Computes intersection for ltrb boxes.

    Args:
        boxes1 (torch.Tensor / np.ndarray): Nx4 array/Tensor of boxes
        boxes2 (torch.Tensor / np.ndarray): Mx4 array/Tensor of boxes

    Return:
        NxM array/Tensor of intersection
    """
    if isinstance(boxes1, Tensor):
        return intersection_pt(boxes1, boxes2)
    return intersection_np(boxes1, boxes2)


@torch.jit.script
def compute_area_pt(boxes: Tensor):
    return ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))


def compute_area(boxes: Type[Union[Tensor, np.ndarray]]):
    r"""Computes area for ltrb boxes.

    Args:
        boxes (torch.Tensor / np.ndarray): Nx4 array/Tensor of boxes

    Return:
        N element array/Tensor of area's
    """
    if isinstance(boxes, Tensor):
        return compute_area_pt(boxes)
    return ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))


def compute_iou(ltrb_boxes1: Type[Union[Tensor, np.ndarray]],
                ltrb_boxes2: Type[Union[Tensor, np.ndarray]],
                return_iof: bool = False):
    r"""Computes all combinations of intersection over union for two sets of
    boxes. Accepts np.ndarray or torch.Tensor.

    Args:
        ltrb_boxes1 (torch.Tensor / np.ndarray): Nx4 array of boxes
        ltrb_boxes2 (torch.Tensor / np.ndarray): Mx4 array of boxes
        return_iof (bool): When True, returns iou and iof (intersection over
            foreground)

            default: False

    Return:
        NxM array/Tensor of iou's
    """

    intersection = compute_intersection(ltrb_boxes1, ltrb_boxes2)
    area_1 = compute_area(ltrb_boxes1)
    area_2 = compute_area(ltrb_boxes2)

    union = area_1[:, None] + area_2[None, ] - intersection
    iou = intersection / union
    if return_iof:
        iof = intersection / (area_1[:, None] + 1e-15)
        return iou, iof
    return iou


def compute_iof(ltrb_boxes1: Type[Union[Tensor, np.ndarray]],
                ltrb_boxes2: Type[Union[Tensor, np.ndarray]]):
    r"""Computes intersection over foreground - ltrb_boxes1 is foreground.

    Args:
        ltrb_boxes1 (torch.Tensor / np.ndarray): Nx4 array of boxes
        ltrb_boxes2 (torch.Tensor / np.ndarray): Mx4 array of boxes

    Return:
        NxM array/Tensor of iof's
    """
    intersection = compute_intersection(ltrb_boxes1, ltrb_boxes2)
    iof = intersection / (compute_area(ltrb_boxes1)[:, None] + 1e-15)
    return iof


def nms_pt(boxes: Tensor, scores: Tensor,
           iou_threshold: float = 0.5, n_objects: int = -1):

    if not boxes.numel():
        return torch.Tensor([]).long()
    retain = []
    scores.squeeze_()
    l, t, r, b = [boxes[:, x] for x in range(4)]
    area = compute_area_pt(boxes)
    idx = torch.sort(scores)[1]
    while idx.numel():
        # add best scored
        retain.append(idx[-1].item())
        if idx.numel() == 1 or len(retain) == n_objects:
            break
        # compute iou
        intersection = intersection_pt(boxes[idx[:-1]], boxes[[idx[-1]]])
        iou = intersection / (area[idx[:-1]] + area[idx[-1]] - intersection)
        # remove boxes with iou above iou_threshold
        idx = idx[:-1][iou < iou_threshold]
    return torch.Tensor(retain).long().to(boxes.device)


def nms_np(boxes: np.ndarray, scores: np.ndarray,
           iou_threshold: float = 0.5, n_objects: int = -1):
    if not boxes.size:
        return np.array([]).astype(np.int32)

    retain = []
    scores = scores.squeeze()
    l, t, r, b = [boxes[:, x] for x in range(4)]
    area = compute_area(boxes)
    idx = np.argsort(scores)
    while idx.size:
        # add best scored
        retain.append(idx[-1])
        if idx.size == 1 or len(retain) == n_objects:
            break
        # compute iou
        intersection = intersection_np(boxes[idx[:-1]], boxes[[idx[-1]]])
        iou = intersection / (area[idx[:-1]] + area[idx[-1]] -
                              intersection)
        # remove boxes with iou above iou_threshold
        idx = idx[:-1][iou < iou_threshold]
    return np.array(retain).astype(np.int32)


def nms(boxes: Type[Union[Tensor, np.ndarray]],
        scores: Type[Union[Tensor, np.ndarray]],
        iou_threshold: float = 0.5, n_objects: int = -1):
    r"""Non-maximal suppression.

    Args:
        boxes (np.ndarray/torch.Tensor): Nx4 ltrb boxes (left, top, right,
            bottom)

        scores (np.ndarray/torch.Tensor): Array of N probabilities.

        iou_threshold (float, optional): iou threshold to eliminate overlaps
            default = 0.5

        n_objects (float, optional): When > 0, quits after n_objects are
            found
            default = -1

    Returns:
         np.ndarray/torch.Tensor of retained indices
    """
    if isinstance(boxes, np.ndarray):
        return nms_np(boxes, scores, iou_threshold, n_objects)
    return nms_pt(boxes, scores, iou_threshold, n_objects)


def centers_per_layer_np(t_size: tuple, c_size: tuple, is_pad: bool):
    (h, w), (ch, cw) = t_size[2:], c_size[2:]
    if is_pad and h / (2 ** np.floor(np.log2(h / ch))) == ch:
        # a corner stretches to a corner if
        #      padding is applied to all convolutions with kernel size > 1
        #      padding is applied to all pooling layers with kernel size > 1
        #      all the kernel sizes are odd (convolution & pooling)
        #      cw * 2^n == w (where n is an integer)
        xs = np.array([w / 2.], dtype=np.float32) if cw == 1 else \
            np.linspace(0, w - 1, num=cw)
        ys = np.array([h / 2.], dtype=np.float32) if ch == 1 else \
            np.linspace(0, h - 1, num=ch)
    else:
        xs = np.arange(0, cw, dtype=np.float32) / cw * w + (w / cw * 0.5) - 0.5
        ys = np.arange(0, ch, dtype=np.float32) / ch * h + (h / ch * 0.5) - 0.5
    xs, ys = np.meshgrid(xs, ys)
    centers_per_layer = np.stack((xs.reshape(-1), ys.reshape(-1)), axis=1)
    return centers_per_layer.astype(np.float32)


def centers_per_layer(t_size: tuple, c_size: tuple, is_pad: bool = True):
    r"""Centers of each location per layer. Since padding is common in most
    networks the centers are stretched.

    Args:
        t_size (tuple): shape of the input tensor in BCHW
            (None/any integer >0, channels, height, width)

        c_size (tuple): shape of the tensor at prediction layer in BCHW
            (None/any integer >0, channels, height, width)

        is_pad (bool): when True, assumes all the convolutions with filter size
            > 1 are odd and use padding. In such a case, a corner at (0, 39) in
             a 40x40 image is mapped to a corner at (0, 319) in 320x320 image.
            default = True

    Return:
        torch.Tensor of shape (c_size[2]*c_size[3])x2
    """
    centers_per_layer = centers_per_layer_np(t_size, c_size, is_pad)
    return torch.from_numpy(centers_per_layer).float()


def encode_boxes(format: str,
                 centers: Tensor,
                 pix2pix_delta: Tensor,
                 anchor_wh: Tensor,
                 r_boxes: Tensor,
                 boxes2centers_mapping: Tensor,
                 var1: float,
                 var2: float):
    r"""Encodes raw boxes given centers, pix2pix_delta and anchor_wh.

    Args:
        format (str): Encoding format
            options = "normalized_offset" | "normalized_gcxcywh"

        centers (Tensor): Centers of all prediction locations

        pix2pix_delta (Tensor): pixel to pixel delta at the prediction layer

        anchor_wh (Tensor): width and height of the anchor

        r_boxes (Tensor): raw bounding boxes from data loader

        boxes2centers_mapping (Tensor): r_boxes to centers mapping based on max
            iou. When None, computes ious and boxes2centers_mapping.

        var1 (float): var1 from SSD/YOLO/R-CNN to derive gcxcywh.

        var2 (float): var1 from SSD/YOLO/R-CNN to derive gcxcywh.
    """
    if boxes2centers_mapping is None:
        # compute ious
        ious = compute_iou(torch.cat(
            (centers - anchor_wh / 2, centers + anchor_wh / 2), 1), r_boxes)
        boxes2centers_mapping = ious.max(1)[1].view(-1)
    if format == "normalized_offset":
        # Similar to FCOS / any IOU based loss functions
        t_boxes = torch.cat((
            (centers - r_boxes[boxes2centers_mapping, :2]) / anchor_wh,
            (r_boxes[boxes2centers_mapping, 2:] - centers) / anchor_wh), 1)
    elif format == "normalized_gcxcywh":
        r_boxes = ObjectUtils.ltrb_to_cxcywh(r_boxes)
        if var1 is not None and var2 is not None:
            # Similar to SSD/YoloV3
            t_boxes = torch.cat((
                (r_boxes[boxes2centers_mapping, :2] - centers) /
                (var1 * anchor_wh),
                ((r_boxes[boxes2centers_mapping, 2:] + 1) / anchor_wh).log()
                / var2), 1)
        else:
            t_boxes = torch.cat((
                (r_boxes[boxes2centers_mapping, :2] - centers) / pix2pix_delta,
                ((r_boxes[boxes2centers_mapping, 2:] + 1) / anchor_wh).log()),
                1)
    else:
        raise NotImplementedError("format = {}?".format(format))
    return t_boxes


def decode_boxes(format: str,
                 centers: Tensor,
                 pix2pix_delta: Tensor,
                 anchor_wh: Tensor,
                 p_boxes: Tensor,
                 var1: float,
                 var2: float):
    r"""Decodes predicted boxes given centers, pix2pix_delta and anchor_wh.

    Args:
        format (str): Encoding format
            options = "normalized_offset" | "normalized_gcxcywh"

        centers (Tensor): Centers of all prediction locations

        pix2pix_delta (Tensor): pixel to pixel delta at the prediction layer

        anchor_wh (Tensor): width and height of the anchor

        p_boxes (Tensor): predicted bounding boxes from network

        var1 (float): var1 from SSD/YOLO/R-CNN to derive gcxcywh.

        var2 (float): var1 from SSD/YOLO/R-CNN to derive gcxcywh.
    """
    if format == "normalized_offset":
        p_boxes = torch.cat((centers - p_boxes[:, :2] * anchor_wh,
                             centers + p_boxes[:, 2:] * anchor_wh), 1)
    elif format == "normalized_gcxcywh":
        if var1 is not None and var2 is not None:
            p_boxes = torch.cat((
                centers + p_boxes[:, :2] * var1 * anchor_wh,
                anchor_wh * torch.exp(p_boxes[:, 2:] * var2)), 1)
        else:
            p_boxes = torch.cat((p_boxes[:, :2] * pix2pix_delta + centers,
                                 p_boxes[:, 2:].exp() * anchor_wh), 1)
    else:
        raise NotImplementedError("format = {}?".format(format))
    return p_boxes


def encode_point(format: str,
                 centers: Tensor,
                 pix2pix_delta: Tensor,
                 anchor_wh: Tensor,
                 r_point: Tensor,
                 boxes2centers_mapping: Tensor,
                 var: float = 0.5):
    r"""Encodes raw point given centers, pix2pix_delta and anchor_wh.

    Args:
        format (str): Encoding format
            options = "normalized_xy_offsets"

        centers (Tensor): Centers of all prediction locations

        pix2pix_delta (Tensor): pixel to pixel delta at the prediction layer
            Not used currently, may be, in future!

        anchor_wh (Tensor): width and height of the anchor

        r_point (Tensor): raw points on the image from data loader
            (non normalized)

        boxes2centers_mapping (Tensor): r_boxes to centers mapping based on max
            iou.

        var (float): var from SSD/YOLO/R-CNN to derive gcxcy.
    """

    if format == "normalized_xy_offsets":
        r_point = r_point.to(centers.device).view(r_point.size(0), -1)
        t_point = r_point[boxes2centers_mapping]
        t_point = t_point.view(centers.size(0), -1, 2)
        t_point -= centers[:, None]
        t_point /= anchor_wh[:, None] * var
        t_point = t_point.view(centers.size(0), -1, 2)
    else:
        raise NotImplementedError("format = {}?".format(format))
    return t_point


def decode_point(format: str,
                 centers: Tensor,
                 pix2pix_delta: Tensor,
                 anchor_wh: Tensor,
                 p_point: Tensor,
                 var: float = 0.5):
    r"""Decodes predicted point given centers, pix2pix_delta and anchor_wh.

    Args:
        format (str): Encoding format
            options = "normalized_xy_offsets"

        centers (Tensor): Centers of all prediction locations

        pix2pix_delta (Tensor): pixel to pixel delta at the prediction layer
            Not used currently, may be, in future!

        anchor_wh (Tensor): width and height of the anchor

        p_point (Tensor): predicted points from network

        var (float): var from SSD/YOLO/R-CNN to derive gcxcy.
    """

    p_point = p_point.view(centers.size(0), -1, 2)
    if format == "normalized_xy_offsets":
        p_point = p_point * (anchor_wh[:, None] * var) + centers[:, None]
    else:
        raise NotImplementedError("format = {}?".format(format))
    return p_point


def compute_centerness(centers: Tensor,
                       r_boxes: Tensor,
                       boxes2centers_mapping: Tensor):
    r"""Computes centerness.
    Paper: FCOS: Fully Convolutional One-Stage Object Detection
    URL:   https://arxiv.org/pdf/1904.01355.pdf

    Args:
        centers (Tensor): Centers of all prediction locations

        r_boxes (Tensor): raw bounding boxes from data loader

        boxes2centers_mapping (Tensor): r_boxes to centers mapping based on max
            iou.
    """

    ltrb_star = torch.cat((
        centers - r_boxes[boxes2centers_mapping, :2],
        r_boxes[boxes2centers_mapping, 2:] - centers), 1)
    centerness = (
        (ltrb_star[:, 0::2].min(1)[0] / ltrb_star[:, 0::2].max(1)[0]) *
        (ltrb_star[:, 1::2].min(1)[0] / ltrb_star[:, 1::2].max(1)[0]))
    return centerness.pow_(0.5)


def compute_objectness(centers: Tensor,
                       pix2pix_delta: Tensor,
                       r_boxes: Tensor):
    r"""Computes objectness -- in the lines of YoloV3 but can also be used
    like a mask.

    Args:
        centers (Tensor): Centers of all prediction locations

        pix2pix_delta (Tensor): pixel to pixel delta at the prediction layer

        r_boxes (Tensor): raw bounding boxes from data loader
    """

    ious, iofs = ObjectUtils.compute_iou(torch.cat(
        (centers - pix2pix_delta / 2, centers + pix2pix_delta / 2), 1),
        r_boxes, return_iof=True)
    objectness = iofs.max(1)[0]
    return objectness


class ObjectUtils:
    r"""Utils required for object detection that accepts numpy array or torch
    Tensor.

    Few Basics
    ----------
    ltrb = (left, top, right, bottom) / corner-form
    cxcywh = (center x, center y, width, height) / center-form
    pixel coordinates = ltrb or cxcywh in pixel locations
    norm01 = ltrb or cxcywh in normalized form (max height & width is 1)


        (0, 0)                                (0, image_width)
        * ---------------- x ---------------- *
        |                                     | i
        |        (l, t)              (r, t)   | m
        |        *---------w---------*        | a
        |        |   w is box width  |        | g
        |        |  h is box height  |        | e
        y        h         *(cx, cy) h        y
        |        |                   |        | h
        |        |                   |        | e
        |        *---------w---------*        | i
        |        (l, b)              (r, b)   | g
        |                                     | h
        * ---------------- x ---------------- * t
                 i m a g e   w i d t h        (image_height, image_width)


    Available box transformations:
    -----------------------------
        pixel_to_norm01  - Normalizes pixel coordinates to 0-1.
        norm01_to_pixel  - Inverse of pixel_to_norm01.
        ltrb_to_cxcywh   - Converts pixel/norm01 ltrb boxes to cxcywh
        cxcywh_to_ltrb   - Converts pixel/norm01 cxcywh boxes to ltrb

    compute_intersection - Computes intersection
    compute_area         - Computes area
    compute_iou          - Computes intersection of union given two sets of
                           boxes
    compute_iof          - Computes intersection of foreground given two sets
                           of boxes
    nms                  - Non-maximal suppression
    centers_per_layer    - Centers of each location per layer
    encode_boxes         - Encodes raw boxes
    decode_boxes         - Decodes predicted boxes
    encode_point         - Encodes raw points
    decode_point         - Decodes predicted points
    compute_centerness   - Computes centerness (FCOS paper)
    compute_objectness   - Objectness per pixel for all prediction layers
    """

    pixel_to_norm01 = pixel_to_norm01
    pixel_to_norm01_np = pixel_to_norm01
    norm01_to_pixel = norm01_to_pixel
    norm01_to_pixel_np = norm01_to_pixel
    ltrb_to_cxcywh = ltrb_to_cxcywh
    ltrb_to_cxcywh_np = ltrb_to_cxcywh
    cxcywh_to_ltrb = cxcywh_to_ltrb
    cxcywh_to_ltrb_np = cxcywh_to_ltrb
    compute_intersection = compute_intersection
    compute_intersection_np = intersection_np
    compute_area = compute_area
    compute_area_np = compute_area
    compute_iou = compute_iou
    compute_iou_np = compute_iou
    compute_iof = compute_iof
    compute_iof_np = compute_iof
    nms_np = nms_np
    nms = nms
    centers_per_layer_np = centers_per_layer_np
    centers_per_layer = centers_per_layer
    encode_boxes = encode_boxes
    decode_boxes = decode_boxes
    encode_point = encode_point
    decode_point = decode_point
    compute_centerness = compute_centerness
    compute_objectness = compute_objectness
