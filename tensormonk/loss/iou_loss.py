""" TensorMONK's :: loss :: IOULoss """

import torch
import torch.nn as nn


class IOULoss(nn.Module):
    r"""Intersection over union for bounding box predictions.

    "log_iou" -> UnitBox: An Advanced Object Detection Network
    Paper -- https://arxiv.org/pdf/1608.01471.pdf

    "iou" -> Optimizing Intersection-Over-Union in Deep Neural Networks for
    Image Segmentation
    Paper -- http://www.cs.umanitoba.ca/~ywang/papers/isvc16.pdf

    "giou" -> Generalized Intersection over Union: A Metric and A Loss for
    Bounding Box Regression
    Paper -- https://arxiv.org/pdf/1902.09630.pdf

    "log_giou" -> extension of "giou"

    Args:
        method (str): Variations of iou based loss functions.
            options = "iou" | "log_iou" | "giou" | "log_giou"
            default = "log_iou"

        reduction (str): The reduction applied to the output.
            options = None | "mean" | "sum"
            default = None

        box_form (str): The format of p_boxes and t_boxes.
            options = "d_ltrb" | "cxcywh"
            default = "d_ltrb"

            d_ltrb = (dl, dt, dr, db) are the distances between the location
            pixel (i, j) and the left, top, right, and bottom boundaries.
            cxcywh = (cx, cy, w, h) corner form boxes
    """
    METHODS = ("iou", "log_iou", "giou", "log_giou")

    def __init__(self,
                 method: str = "log_iou",
                 reduction: str = None,
                 box_form: str = "d_ltrb",
                 **kwargs):
        super(IOULoss, self).__init__()

        if not isinstance(method, str):
            raise TypeError("IOULoss: method must be str: "
                            "{}".format(type(method).__name__))
        method = method.lower()
        if method not in IOULoss.METHODS:
            raise ValueError("IOULoss: method must be "
                             "iou/log_iou/giou/log_giou: "
                             "{}".format(method))

        if not isinstance(box_form, str):
            raise TypeError("IOULoss: box_form must be str: "
                            "{}".format(type(box_form).__name__))
        box_form = box_form.lower()
        if box_form not in ["d_ltrb", "cxcywh"]:
            raise ValueError("IOULoss: box_form must be d_ltrb/cxcywh: "
                             "{}".format(box_form))
        if reduction is not None:
            if not isinstance(reduction, str):
                raise TypeError("IOULoss: reduction must be str/None: "
                                "{}".format(type(reduction).__name__))
            reduction = reduction.lower()
            if reduction not in ["sum", "mean"]:
                raise ValueError("IOULoss: reduction must be sum/mean/None: "
                                 "{}".format(reduction))

        self.method = method
        self.is_offsets = box_form == "d_ltrb"
        self.reduction = reduction
        self.tensor_size = 1,

    def forward(self,
                p_boxes: torch.Tensor,
                t_boxes: torch.Tensor,
                weights: torch.Tensor = None):
        r"""
        p_boxes: Must have Nx4 shape (N can vary per batch).
            Each box must be in the d_ltrb form; d_ltrb = (dl, dt, dr, db)
            (dl, dt, dr, db) are the distances between the location
            pixel (i, j) and the left, top, right, and bottom boundaries.
            Must be valid boxes not all the possible boxes on an image.
        t_boxes: Same as p_boxes.
        weights: None or tensor of shape Nx1 or N.
        """

        assert p_boxes.size(-1) == 4 and t_boxes.size(-1) == 4, \
            "IOULoss: p_boxes/t_boxes last dimension length != 4"
        p_boxes, t_boxes = p_boxes.view(-1, 4), t_boxes.view(-1, 4)
        assert p_boxes.size(0) == t_boxes.size(0), \
            "IOULoss: p_boxes.size(0) ~= t_boxes.size(0)"

        if self.is_offsets:
            p_dl, p_dt, p_dr, p_db = [p_boxes[:, i] for i in range(4)]
            t_dl, t_dt, t_dr, t_db = [t_boxes[:, i] for i in range(4)]
            assert not (p_dl.lt(0).any() or p_dt.lt(0).any() or
                        p_dr.lt(0).any() or p_db.lt(0).any()), \
                "IOULoss: p_boxes boxes offset must be >= 0 (use relu)"
            assert not (t_dl.lt(0).any() or t_dt.lt(0).any() or
                        t_dr.lt(0).any() or t_db.lt(0).any()), \
                "IOULoss: target boxes offset must be >= 0"
            # intersection
            intersection = ((torch.min(p_dl, t_dl) + torch.min(p_dr, t_dr)) *
                            (torch.min(p_dt, t_dt) + torch.min(p_db, t_db)))
            # union
            p_area = (p_dl + p_dr) * (p_dt + p_db)
            t_area = (t_dl + t_dr) * (t_dt + t_db)
            union = p_area + t_area - intersection

            if self.method in ["giou", "log_giou"]:
                ac = ((torch.max(p_dl, t_dl) + torch.max(p_dr, t_dr)) *
                      (torch.max(p_dt, t_dt) + torch.max(p_db, t_db)))

        else:
            p_cx, p_cy, p_w, p_h = [p_boxes[:, i] for i in range(4)]
            t_cx, t_cy, t_w, t_h = [t_boxes[:, i] for i in range(4)]
            assert not (p_w.le(0).any() or p_h.le(0).any()), \
                "IOULoss: p_boxes boxes are not in center form (use relu)"
            assert not (t_w.le(0).any() or t_h.le(0).any()), \
                "IOULoss: target boxes are not in center form"
            # intersection
            intersection = ((torch.min(p_cx + p_w, t_cx + t_w) -
                             torch.max(p_cx - p_w, t_cx - t_w)) *
                            (torch.min(p_cy + p_h, t_cy + t_h) -
                             torch.max(p_cy - p_h, t_cy - t_h)))
            # union
            p_area = (p_w * 2) * (p_h * 2)
            t_area = (t_w * 2) * (t_h * 2)
            union = p_area + t_area - intersection
            ious = (intersection + 1.0) / (union + 1.0)

            if self.method in ["giou", "log_giou"]:
                ac = ((torch.max(p_cx + p_w, t_cx + t_w) -
                       torch.min(p_cx - p_w, t_cx - t_w)) *
                      (torch.max(p_cy + p_h, t_cy + t_h) -
                       torch.min(p_cy - p_h, t_cy - t_h)))

        # iou or generalized iou computation
        ious = intersection / union.add(1e-8)
        if self.method in ["giou", "log_giou"]:
            ious = ious - ((ac - union) / ac.add(1e-8))

        # - ious.log() / 1 - ious
        if self.method in ["iou", "giou"]:
            iou_loss = 1 - ious
        else:
            iou_loss = - torch.log(ious.clamp(1e-8, 1))

        # weighted loss
        if weights is not None:
            iou_loss = iou_loss * weights.view(-1)

        # reduction
        if self.reduction == "sum":
            return iou_loss.sum()
        elif self.reduction == "mean":
            return iou_loss.mean()
        else:
            return iou_loss

    def __repr__(self):
        return "IOU Loss ({}): reduction = {}".format(self.method,
                                                      self.reduction)


# p_boxes = torch.Tensor([[0.1, 0.2, 0.1, 0.3]])
# t_boxes = torch.Tensor([[0.2, 0.1, 0.2, 0.4]])
# IOULoss("log_iou")(p_boxes, t_boxes)
# IOULoss("iou")(p_boxes, t_boxes)
# IOULoss("log_giou")(p_boxes, t_boxes)
# IOULoss("giou")(p_boxes, t_boxes)
