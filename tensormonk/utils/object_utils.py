""" TensorMONK :: utils """

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ObjectUtils:
    """
    Utils required for object detection that accepts numpy array or torch
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
        pixel_to_norm01 - Normalizes pixel coordinates to 0-1, requires width
            and height of image.
        norm01_to_pixel - Inverse of pixel_to_norm01.
        auto_convert - Does pixel_to_norm01 or norm01_to_pixel based on input
            values.
        ltrb_to_cxcywh - Converts pixel/norm01 ltrb boxes to cxcywh
        cxcywh_to_ltrb - Converts pixel/norm01 cxcywh boxes to ltrb

    compute_iou - computes intersection of union given two sets of boxes
    nms - Non-maximal suppression (requires normalized ltrb_boxes and scores)
    """
    AvailableTypes = ("numpy", "torch")

    @staticmethod
    def to_array(boxes, itype: str = "numpy"):
        """
        Return's torch.Tensor/np.ndarray.
        Accepts list/tuple/np.ndarray/tensor.Tensor
        """

        if isinstance(boxes, list) or isinstance(boxes, tuple):
            if itype == "torch":
                boxes = torch.Tensor(boxes)
            elif itype == "numpy":
                boxes = np.array(boxes)

        if isinstance(boxes, np.ndarray) and itype != "numpy":
            boxes = torch.from_numpy(boxes)
        if isinstance(boxes, torch.Tensor) and itype != "torch":
            boxes = boxes.data.cpu().numpy()

        if isinstance(boxes, np.ndarray):
            if boxes.ndim == 1:
                boxes = boxes.reshape(1, -1)
            boxes = boxes.astype(np.float32)
        if isinstance(boxes, torch.Tensor):
            if boxes.dim() == 1:
                boxes = boxes.view(1, -1)
            boxes = boxes.float()
        return boxes

    @staticmethod
    def pixel_to_norm01(boxes, w: int, h: int, itype: str = "torch"):
        """ Normalizes bounding boxes (ltrb/cxcywh) from pixel coordinates to
        normalized 0-1 form given width (w) and height (h) of an image """

        boxes = ObjectUtils.to_array(boxes, itype)
        # to normalized 0-1
        boxes[:, 0::2] /= w
        boxes[:, 1::2] /= h
        return boxes

    @staticmethod
    def norm01_to_pixel(boxes, w: int, h: int, itype: str = "torch"):
        """ Normalizes bounding boxes (ltrb/cxcywh) from normalized 0-1 form to
        pixel coordinates given width (w) and height (h) of an image """

        boxes = ObjectUtils.to_array(boxes, itype)
        # to pixel coordinates
        boxes[:, 0::2] *= w
        boxes[:, 1::2] *= h
        return boxes

    @staticmethod
    def auto_convert(boxes, w: int, h: int, itype: str = "torch"):
        """ Normalizes bounding boxes (ltrb/cxcywh) from (pixel coordinates to
        normalized 0-1) or (normalized 0-1 form to pixel coordinates) given
        width (w) and height (h) of image """

        boxes = ObjectUtils.to_array(boxes, itype)
        if boxes.max() < 2:
            # to pixel coordinates
            boxes[:, 0::2] *= w
            boxes[:, 1::2] *= h
        else:
            # to normalized 0-1
            boxes[:, 0::2] /= w
            boxes[:, 1::2] /= h
        return boxes

    @staticmethod
    def ltrb_to_cxcywh(boxes, itype: str = "torch"):
        """ Converts bouning boxes from
        (left, top, right, bottom) to (center x, center y, width, height) """

        boxes = ObjectUtils.to_array(boxes, itype)
        is_tensor = isinstance(boxes, torch.Tensor)
        boxes = (boxes[:, 0::2].mean(1), boxes[:, 1::2].mean(1),
                 boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])
        boxes = torch.cat(boxes).view(4, -1).t() if is_tensor else \
            np.vstack(boxes).T
        return boxes

    @staticmethod
    def cxcywh_to_ltrb(boxes, itype: str = "torch"):
        """ Converts bouning boxes from
        (center x, center y, width, height) to (left, top, right, bottom) """

        boxes = ObjectUtils.to_array(boxes, itype)
        is_tensor = isinstance(boxes, torch.Tensor)
        boxes = (boxes[:, 0] - (boxes[:, 2]/2), boxes[:, 1] - (boxes[:, 3]/2),
                 boxes[:, 0] + (boxes[:, 2]/2), boxes[:, 1] + (boxes[:, 3]/2))
        boxes = torch.cat(boxes).view(4, -1).t() if is_tensor else \
            np.vstack(boxes).T
        return boxes

    @staticmethod
    def compute_iou(ltrb_boxes1, ltrb_boxes2):
        """ Computes all combinations of intersection over union for two sets of
        boxes. Accepts np.ndarray or torch.Tensor.
        """
        if isinstance(ltrb_boxes1, np.ndarray):
            # auto convert - cxcywh_to_ltrb
            if any((ltrb_boxes1[:, 2] - ltrb_boxes1[:, 0]) < 0) or \
               any((ltrb_boxes1[:, 3] - ltrb_boxes1[:, 1]) < 0):
                ltrb_boxes1 = ObjectUtils.cxcywh_to_ltrb(ltrb_boxes1, "numpy")
            if any((ltrb_boxes2[:, 2] - ltrb_boxes2[:, 0]) < 0) or \
               any((ltrb_boxes2[:, 3] - ltrb_boxes2[:, 1]) < 0):
                ltrb_boxes2 = ObjectUtils.cxcywh_to_ltrb(ltrb_boxes2, "numpy")

            # intersection
            lt = np.maximum(ltrb_boxes1[:, :2][:, np.newaxis],
                            ltrb_boxes2[:, :2][np.newaxis, :])
            rb = np.minimum(ltrb_boxes1[:, 2:][:, np.newaxis],
                            ltrb_boxes2[:, 2:][np.newaxis, :])
            intersection = np.multiply(*np.split(np.clip(rb - lt, 0, None),
                                       2, 2)).squeeze(2)

            # union
            area_1 = ((ltrb_boxes1[:, 2] - ltrb_boxes1[:, 0]) *
                      (ltrb_boxes1[:, 3] - ltrb_boxes1[:, 1]))
            area_2 = ((ltrb_boxes2[:, 2] - ltrb_boxes2[:, 0]) *
                      (ltrb_boxes2[:, 3] - ltrb_boxes2[:, 1]))
            union = area_1[:, np.newaxis] + area_2[np.newaxis, ] - intersection
            return intersection / union

        """ Torch Implementation """
        # auto convert - cxcywh_to_ltrb
        if (ltrb_boxes1[:, 2] - ltrb_boxes1[:, 0]).lt(0).any() or \
           (ltrb_boxes1[:, 3] - ltrb_boxes1[:, 1]).lt(0).any():
            ltrb_boxes1 = ObjectUtils.cxcywh_to_ltrb(ltrb_boxes1)
        if (ltrb_boxes2[:, 2] - ltrb_boxes2[:, 0]).lt(0).any() or \
           (ltrb_boxes2[:, 3] - ltrb_boxes2[:, 1]).lt(0).any():
            ltrb_boxes2 = ObjectUtils.cxcywh_to_ltrb(ltrb_boxes2)

        # intersection
        lt = torch.max(ltrb_boxes1[:, :2].unsqueeze(1),
                       ltrb_boxes2[:, :2].unsqueeze(0))
        rb = torch.min(ltrb_boxes1[:, 2:].unsqueeze(1),
                       ltrb_boxes2[:, 2:].unsqueeze(0))
        intersection = torch.mul(*(rb-lt).clamp(0).split(1, dim=2))
        intersection.squeeze_(2)
        # union
        area_1 = ((ltrb_boxes1[:, 2] - ltrb_boxes1[:, 0]) *
                  (ltrb_boxes1[:, 3] - ltrb_boxes1[:, 1]))
        area_2 = ((ltrb_boxes2[:, 2] - ltrb_boxes2[:, 0]) *
                  (ltrb_boxes2[:, 3] - ltrb_boxes2[:, 1]))
        union = area_1.unsqueeze(1) + area_2.unsqueeze(0) - intersection
        return intersection / union

    @staticmethod
    def nms(ltrb_boxes, scores, iou_threshold: float = 0.5,
            n_objects: int = -1):
        """ Non-maximal suppression - requires normalized ltrb_boxes and scores.

        Args:
            ltrb_boxes (torch.Tensor): normalized (left, top, right, bottom)
            scores (torch.Tensor): probabilities
            iou_threshold (float, optional): intersection over union threshold
                to eliminate overlaps, default=0.5
            n_objects (float, optional): When > 0, quits after n_objects are
                found (if available)

        Returns:
             torch.Tensor of retained indices
        """
        if isinstance(ltrb_boxes, np.ndarray):
            if not ltrb_boxes.size:
                return np.array([]).astype(np.int32)

            retain = []
            scores = scores.squeeze()
            l, t, r, b = [ltrb_boxes[:, x] for x in range(4)]
            area = ((ltrb_boxes[:, 2] - ltrb_boxes[:, 0]) *
                    (ltrb_boxes[:, 3] - ltrb_boxes[:, 1]))
            idx = np.argsort(scores)
            while idx.numel():
                # add best scored
                retain.append(idx[-1])
                if idx.size == 1 or len(retain) == n_objects:
                    break
                # compute iou
                intersection = ((np.minimum(r[idx[:-1]], r[idx[-1]]) -
                                 np.maximum(l[idx[:-1]], l[idx[-1]])) *
                                (np.minimum(b[idx[:-1]], b[idx[-1]]) -
                                 np.maximum(t[idx[:-1]], t[idx[-1]])))
                iou = intersection / (area[idx[:-1]] + area[idx[-1]] -
                                      intersection)
                # remove boxes with iou above iou_threshold
                idx = idx[:-1][iou < iou_threshold]
            return np.array(retain).astype(np.int32)

        """ Torch Implementation """
        if not ltrb_boxes.numel():
            return torch.Tensor([]).long()
        retain = []
        scores.squeeze_()
        l, t, r, b = [ltrb_boxes[:, x] for x in range(4)]
        area = ((ltrb_boxes[:, 2] - ltrb_boxes[:, 0]) *
                (ltrb_boxes[:, 3] - ltrb_boxes[:, 1]))
        idx = torch.sort(scores)[1]

        while idx.numel():
            # add best scored
            retain.append(idx[-1].item())
            if idx.numel() == 1 or len(retain) == n_objects:
                break
            # compute iou
            intersection = ((torch.min(r[idx[:-1]], r[idx[-1]]) -
                             torch.max(l[idx[:-1]], l[idx[-1]])) *
                            (torch.min(b[idx[:-1]], b[idx[-1]]) -
                             torch.max(t[idx[:-1]], t[idx[-1]])))
            iou = intersection / (area[idx[:-1]]+area[idx[-1]]-intersection)
            # remove boxes with iou above iou_threshold
            idx = idx[:-1][iou < iou_threshold]
        return torch.Tensor(retain).long().to(ltrb_boxes.device)


class Translator(nn.Module):
    """
    Translator that converts the normalized ltrb_boxes to gcxcywh_boxes along
    with labels (boxes and labels are provided) and vice-versa (when boxes are
    provided). When boxes and predictions are provided, does decoding and nms
    to deliver the detection objects, and their respective scores and labels.


    Args:
        layer_infos ({list, tuple}): Used to generate priors, an example is
            provided in ../../examples/mobile_ssd320.py
        gcxcywh_var1 (float): Used to convert ltrb_boxes to gcxcywh_boxes and
            vice-versa, default=0.1
        gcxcywh_var2 (float): Used to convert ltrb_boxes to gcxcywh_boxes and
            vice-versa, default=0.2
        encode_iou_threshold (float): Higher value creates strict conversion,
            used to generate gcxcywh_boxes and their corresponding labels.
            default=0.5
        detect_iou_threshold (float): IoU Threshold used by nms to select
            pick boxes, lower value detects fewer overlapped boxes.
            default=0.2
        detect_score_threshold (float): Used to detect object above the given
            threshold. However, the detect module picks the best available
            non-background when all non-background scores are below the
            threshold. default=0.2
        detect_top_n (int): Picks top_n objects at a given locaiton (currently,
            disabled), default=1
        detect_n_objects (int): Picks a maximum of n object when more are
            available. default=50

    Return:
        Depends on inputs
    """

    def __init__(self,
                 layer_infos: list,
                 gcxcywh_var1: float = 0.1,
                 gcxcywh_var2: float = 0.2,
                 encode_iou_threshold: float = 0.5,
                 detect_iou_threshold: float = 0.2,
                 detect_score_threshold: float = 0.2,
                 detect_top_n: int = 1,
                 detect_n_objects: int = 50, **kwargs):
        super(Translator, self).__init__()

        # get priors for "SSD300", "SSD320"
        priors = SSDUtils.compute_ssd_priors(layer_infos)
        self.cxcywh_priors = priors
        self.ltrb_priors = ObjectUtils.cxcywh_to_ltrb(priors)
        self.var1, self.var2 = gcxcywh_var1, gcxcywh_var2
        self.encode_iou = encode_iou_threshold
        self.detect_iou = detect_iou_threshold
        self.score = detect_score_threshold
        self.detect_top_n = detect_top_n
        self.n_objects = detect_n_objects

    def forward(self, boxes, labels=None, predictions=None):
        """ The order of inputs determine what has to be done! """

        if labels is None and predictions is None:
            return self.decode(boxes)
        if predictions is None:
            return self.encode(boxes, labels)
        return self.detect(boxes, predictions)

    def encode(self, boxes, labels):
        """
        Input:
            ltrb_boxes (from data loader) - can be a list of one set per image
            labels (from data loader) - labels for ltrb_boxes
        Returns:
            target_gcxcywh_boxes & target_label: the ltrb_boxes and labels are
                mapped to priors that meet the encode_iou_threshold
        """

        if isinstance(boxes, list) or isinstance(boxes, tuple):
            # batch Processing
            _boxes, _labels = [], []
            for x, y in zip(boxes, labels):
                box, label = self.encode(x, y)
                _boxes.append(box)
                _labels.append(label)
            return torch.stack(_boxes, 0), torch.stack(_labels, 0)

        # encode ltrb_boxes to gcxcywh_boxes and labels
        if not (self.cxcywh_priors.device == boxes.device):
            self.cxcywh_priors = self.cxcywh_priors.to(boxes.device)
            self.ltrb_priors = self.ltrb_priors.to(boxes.device)
        # iou of n ltrb_boxes vs 8732 priors (for SSD300)
        iou = ObjectUtils.compute_iou(boxes, self.ltrb_priors)
        # best of each
        prior_per_label, prior_per_label_idx = iou.max(0)
        label_per_prior, label_per_prior_idx = iou.max(1)
        # update actual class with high value or actual class
        prior_per_label[label_per_prior_idx] = 1
        for i in range(label_per_prior_idx.size(0)):
            prior_per_label_idx[label_per_prior_idx[i]] = i

        # get expected SSD output
        boxes_on_image = boxes[prior_per_label_idx]
        target_boxes = SSDUtils.ltrb_to_gcxcywh(
            boxes_on_image, self.cxcywh_priors, self.var1, self.var2)
        target_label = labels[prior_per_label_idx]
        target_label[prior_per_label < self.encode_iou] = 0

        return target_boxes, target_label.long()

    def decode(self, boxes: torch.Tensor):
        """
        Input:
            gcxcywh_boxes (from network) for all priors
        Returns:
            ltrb_boxes for all priors
        """
        if isinstance(boxes, list) or isinstance(boxes, tuple):
            # batch Processing
            return torch.stack([self.decode(x) for x in boxes], 0)
        if not (self.cxcywh_priors.device == boxes.device):
            self.cxcywh_priors = self.cxcywh_priors.to(boxes.device)
        return SSDUtils.gcxcywh_to_ltrb(boxes, self.cxcywh_priors,
                                        self.var1, self.var2)

    def detect(self, boxes: torch.Tensor, predictions: torch.Tensor):
        """
        Input:
            boxes (from network) - 2D or 3D torch.Tensor of gcxcywh_boxes
            predictions (from network) - 2D or 3D torch.Tensor of predictions
                (before softmax)
        Returns:
            list of detected_objects, list of their_labels, list of
                their_scores (your final predictions)
        """
        if boxes.dim() == 3:
            # batch Processing
            detected_objects, their_labels, their_scores = [], [], []
            for i in range(boxes.size(0)):
                _box, _label, _score = self.detect(boxes[i], predictions[i])
                detected_objects.append(_box)
                their_labels.append(_label)
                their_scores.append(_score)
            return detected_objects, their_labels, their_scores

        with torch.no_grad():
            predictions = F.softmax(predictions, dim=1)
            detected_objects, their_labels, their_scores = \
                self._detect_per_image(boxes, predictions)
            return detected_objects, their_labels, their_scores

    def _detect_per_image(self, gcxcywh_boxes, predictions):
        # pick top_n objects
        sorted_scores, sorted_idx = torch.sort(predictions, dim=1)
        sorted_scores = sorted_scores[:, -2:]
        sorted_idx = sorted_idx[:, -2:]
        # pick best non-background per location
        best = sorted_idx[:, 1]
        best_2nd = sorted_idx[:, 0]
        best[best == 0] = best_2nd[best == 0]
        scores = predictions.gather(1, best.view(-1, 1)).view(-1)

        # convert to ltrb
        ltrb_boxes = self.decode(gcxcywh_boxes)
        valid_idx = ObjectUtils.nms(ltrb_boxes, scores,
                                    self.detect_iou, self.n_objects)

        # score_threshold
        if self.score > 0:
            valid_score = (scores[valid_idx] > self.score).view(-1)
            if valid_score.sum() == 0:
                # when no objects pass score threshold, pick best available
                valid_score = scores[valid_idx] == scores[valid_idx].max()
            valid_idx = valid_idx[valid_score]
        boxes = ltrb_boxes[valid_idx].clamp(0, 1)
        labels = best[valid_idx]
        scores = scores[valid_idx]
        return boxes, labels, scores


class SSDUtils:
    """
    Utils required for Single Shot MultiBox Detector (SSD).
    Original paper -- https://arxiv.org/pdf/1512.02325.pdf

    cxcywh_to_gcxcywh - center-form to normalized (eq 2 in paper)
    gcxcywh_to_cxcywh - inverse of cxcywh_to_gcxcywh
    ltrb_to_gcxcywh - corner-form to normalized (extension of eq 2 in paper)
    gcxcywh_to_ltrb - inverse of ltrb_to_gcxcywh
    a_prior - generates priors of a layer given tensor_size (BCHW - HW is all
        it needs), aspect ratios (a list of boxes aspect ratios), scale
        (relative to input size), next scale (a lower scale).
    compute_ssd_priors - generates priors for all the layers.
    SSD300_priors - SSD300 priors used in the paper.
    SSD320_priors - SSD320 priors for input image size of 320x320
    """
    from collections import namedtuple
    LayerInfo = namedtuple("LayerInfo", ["tensor_size", "aspect_ratios",
                                         "scale", "next_scale"])
    Translator = Translator

    @staticmethod
    def cxcywh_to_gcxcywh(boxes: torch.Tensor, priors: torch.Tensor,
                          var1: float = 0.1, var2: float = 0.2):
        """ Normalized cxcywh to gcxcywh (encoded using priors & variance) """
        boxes = torch.cat((
            (boxes[:, :2] - priors[:, :2]) / (var1 * priors[:, 2:]),
            torch.log(boxes[:, 2:] / priors[:, 2:]) / var2), 1)
        return boxes

    @staticmethod
    def gcxcywh_to_cxcywh(boxes: torch.Tensor, priors: torch.Tensor,
                          var1: float = 0.1, var2: float = 0.2):
        """ gcxcywh to normalized cxcywh (encoded using priors & variance) """

        boxes = torch.cat((
            priors[:, :2] + boxes[:, :2] * var1 * priors[:, 2:],
            priors[:, 2:] * torch.exp(boxes[:, 2:] * var2)), 1)
        return boxes

    @staticmethod
    def ltrb_to_gcxcywh(boxes: torch.Tensor, priors: torch.Tensor,
                        var1: float = 0.1, var2: float = 0.2):
        """ Normalized ltrb to gcxcywh (encoded using priors & variance) """

        boxes = ObjectUtils.ltrb_to_cxcywh(boxes)
        return SSDUtils.cxcywh_to_gcxcywh(boxes, priors, var1, var2)

    @staticmethod
    def gcxcywh_to_ltrb(boxes: torch.Tensor, priors: torch.Tensor,
                        var1: float = 0.1, var2: float = 0.2):
        """ gcxcywh to normalized ltrb (encoded using priors & variance) """

        boxes = SSDUtils.gcxcywh_to_cxcywh(boxes, priors, var1, var2)
        return ObjectUtils.cxcywh_to_ltrb(boxes)

    @staticmethod
    def a_prior(layer_info) -> torch.Tensor:
        """ Generates priors (torch.Tensor) for a layer """
        # all possible locations (x, y) - normalized between 0-1
        h, w = layer_info.tensor_size[2:]
        xs = (torch.arange(0, w).view(1, -1).repeat(1, h).float() + .5) / w
        ys = (torch.arange(0, h).view(-1, 1).repeat(1, w).float() + .5) / h
        cxcy_locations = torch.cat((xs.view(-1, 1), ys.view(-1, 1)), 1)

        cxcywh_boxes = []
        for ar in layer_info.aspect_ratios:
            # boxes per aspect ratio
            wh = torch.zeros(*cxcy_locations.shape).float()
            wh.add_(layer_info.scale)
            wh[:, 0] *= ar**0.5
            wh[:, 1] /= ar**0.5
            cxcywh_boxes += [torch.cat((cxcy_locations, wh), 1)]
            if ar == 1:
                # extra default box
                if isinstance(layer_info.next_scale, float):
                    scale = (layer_info.scale * layer_info.next_scale)**0.5
                    wh = torch.zeros(*cxcy_locations.shape).float()
                    wh.add_(scale)
                    wh[:, 0] *= ar**0.5
                    wh[:, 1] /= ar**0.5
                    cxcywh_boxes += [torch.cat((cxcy_locations, wh), 1)]
        return torch.cat(cxcywh_boxes, 1).view(-1, 4)

    @staticmethod
    def compute_ssd_priors(layer_infos) -> torch.Tensor:
        """ Generates priors (torch.Tensor) for a set of layer """
        cxcywh_priors = [SSDUtils.a_prior(x) for x in layer_infos]
        return torch.cat(cxcywh_priors).clamp(0, 1)

    @staticmethod
    def SSD300_priors() -> torch.Tensor:
        """
        Expected architecture for input size of 300x300
          None x 3 x 300 x 300
          None x _ x 150 x 150
          None x _ x  75 x  75
          None x _ x  38 x  38 -> Predictions
          None x _ x  19 x  19 -> Predictions
          None x _ x  10 x  10 -> Predictions
          None x _ x   5 x   5 -> Predictions
          None x _ x   3 x   3 -> Predictions
          None x _ x   1 x   1 -> Predictions
        A total of 8732 priors -- (5776+2166+600+150+36+4) = 8732

        scales = [x/300. for x in [30, 60, 111, 162, 213, 264, 315]]
        """

        ratios1 = (1, 2, 1/2)
        ratios2 = (1, 2, 3, 1/2, 1/3)
        layer_infos = [
            SSDUtils.LayerInfo((None, None, 38, 38), ratios1, 0.10, 0.20),
            SSDUtils.LayerInfo((None, None, 19, 19), ratios2, 0.20, 0.37),
            SSDUtils.LayerInfo((None, None, 10, 10), ratios2, 0.37, 0.54),
            SSDUtils.LayerInfo((None, None,  5,  5), ratios2, 0.54, 0.71),
            SSDUtils.LayerInfo((None, None,  3,  3), ratios1, 0.71, 0.88),
            SSDUtils.LayerInfo((None, None,  1,  1), ratios1, 0.88, 1.05)]
        return SSDUtils.compute_ssd_priors(layer_infos)

    @staticmethod
    def SSD320_priors() -> torch.Tensor:
        """"
        Expected architecture for input size of 320x320
          None x 3 x 320 x 320
          None x _ x 160 x 160
          None x _ x  80 x  80
          None x _ x  40 x  40 -> Predictions
          None x _ x  20 x  20 -> Predictions
          None x _ x  10 x  10 -> Predictions
          None x _ x   5 x   5 -> Predictions
          None x _ x   3 x   3 -> Predictions
          None x _ x   1 x   1 -> Predictions
        A total of 9590 priors -- (6400+2400+600+150+36+4) = 9590
        scales = used same scales as SSD300
        """

        ratios1 = (1, 2, 1/2)
        ratios2 = (1, 2, 3, 1/2, 1/3)
        layer_infos = [
            SSDUtils.LayerInfo((None, None, 40, 40), ratios1, .10, 0.20),
            SSDUtils.LayerInfo((None, None, 20, 20), ratios2, .20, 0.37),
            SSDUtils.LayerInfo((None, None, 10, 10), ratios2, .37, 0.54),
            SSDUtils.LayerInfo((None, None,  5,  5), ratios2, .54, 0.71),
            SSDUtils.LayerInfo((None, None,  3,  3), ratios1, .71, 0.88),
            SSDUtils.LayerInfo((None, None,  1,  1), ratios1, .88, 1.05)]
        return SSDUtils.compute_ssd_priors(layer_infos)
