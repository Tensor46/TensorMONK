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
        iou_type (str): Variations of iou based loss functions.
            options = "iou" | "log_iou" | "giou" | "log_giou"
            default = "log_iou"

        reduction (str): The reduction applied to the output.
            options = None | "mean" | "sum"
            default = None

        box_form (str): The format of predicted and targets.
            options = "d_ltrb" | "cxcywh"
            default = "d_ltrb"

            d_ltrb = (dl, dt, dr, db) are the distances between the location
            pixel (i, j) and the left, top, right, and bottom boundaries.
            cxcywh = (cx, cy, w, h) corner form boxes
    """

    def __init__(self,
                 iou_type: str = "log_iou",
                 reduction: str = None,
                 box_form: str = "d_ltrb",
                 **kwargs):
        super(IOULoss, self).__init__()

        if not isinstance(iou_type, str):
            raise TypeError("IOULoss: iou_type must be str: "
                            "{}".format(type(iou_type).__name__))
        iou_type = iou_type.lower()
        if iou_type not in ["iou", "log_iou", "giou", "log_giou"]:
            raise ValueError("IOULoss: iou_type must be "
                             "iou/log_iou/giou/log_giou: "
                             "{}".format(iou_type))

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

        self.iou_type = iou_type
        self.is_offsets = box_form == "d_ltrb"
        self.reduction = reduction
        self.tensor_size = 1,

    def forward(self,
                predicted: torch.Tensor,
                targets: torch.Tensor,
                weights: torch.Tensor = None):
        r"""
        predicted: Must have Nx4 shape (N can vary per batch).
            Each box must be in the d_ltrb form; d_ltrb = (dl, dt, dr, db)
            (dl, dt, dr, db) are the distances between the location
            pixel (i, j) and the left, top, right, and bottom boundaries.
            Must be valid boxes not all the possible boxes on an image.
        targets: Same as predicted.
        weights: None or tensor of shape Nx1 or N.
        """

        assert predicted.size(-1) == 4 and targets.size(-1) == 4, \
            "IOULoss: predicted/targets last dimension length != 4"
        predicted, targets = predicted.view(-1, 4), targets.view(-1, 4)
        assert predicted.size(0) == targets.size(0), \
            "IOULoss: predicted.size(0) ~= targets.size(0)"

        if self.is_offsets:
            p_dl, p_dt, p_dr, p_db = [predicted[:, i] for i in range(4)]
            t_dl, t_dt, t_dr, t_db = [targets[:, i] for i in range(4)]
            assert not (p_dl.lt(0).any() or p_dt.lt(0).any() or
                        p_dr.lt(0).any() or p_db.lt(0).any()), \
                "IOULoss: predicted boxes offset must be >= 0 (use relu)"
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

            if self.iou_type in ["giou", "log_giou"]:
                ac = ((torch.max(p_dl, t_dl) + torch.max(p_dr, t_dr)) *
                      (torch.max(p_dt, t_dt) + torch.max(p_db, t_db)))

        else:
            p_cx, p_cy, p_w, p_h = [predicted[:, i] for i in range(4)]
            t_cx, t_cy, t_w, t_h = [targets[:, i] for i in range(4)]
            assert not (p_w.le(0).any() or p_h.le(0).any()), \
                "IOULoss: predicted boxes are not in center form (use relu)"
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

            if self.iou_type in ["giou", "log_giou"]:
                ac = ((torch.max(p_cx + p_w, t_cx + t_w) -
                       torch.min(p_cx - p_w, t_cx - t_w)) *
                      (torch.max(p_cy + p_h, t_cy + t_h) -
                       torch.min(p_cy - p_h, t_cy - t_h)))

        # iou or generalized iou computation
        ious = intersection / union
        if self.iou_type in ["giou", "log_giou"]:
            ious = ious - ((ac - union) / ac.add(1e-8))

        # - ious.log() / 1 - ious
        if self.iou_type in ["iou", "giou"]:
            iou_loss = 1 - ious
        else:
            iou_loss = - torch.log(ious)

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


# predicted = torch.Tensor([[0.1, 0.2, 0.1, 0.3]])
# targets = torch.Tensor([[0.2, 0.1, 0.2, 0.4]])
# IOULoss("log_iou")(predicted, targets)
# IOULoss("iou")(predicted, targets)
# IOULoss("log_giou")(predicted, targets)
# IOULoss("giou")(predicted, targets)
