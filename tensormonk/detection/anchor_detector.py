""" TensorMONK's :: detection :: Detector """


__all__ = ["Classifier", "AnchorDetector", "Responses"]

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torchvision
from typing import Union
from .config import CONFIG
from .nofpn_fpn import BiFPNLayer, FPNLayer, PAFPNLayer, NoFPNLayer
from .utils import ObjectUtils
from ..layers import MBBlock
from .responses import Responses


class Classifier(nn.Module):
    r"""Classifier layer to predict labels, boxes, points, objectness and
    centerness.

    Args:
        config (:class:`~tensormonk.detection.CONFIG`): See
            :class:`tensormonk.detection.CONFIG` for more details.

    :rtype: :class:`tensormonk.detection.Responses`
    """
    def __init__(self, config: CONFIG):
        super(Classifier, self).__init__()
        self.config = config

        ic = config.encoding_depth
        n_anchors_per_layer = [len(x) for x in config.anchors_per_layer]
        oc = int(config.n_label + 4 +
                 ((config.n_point * 2) if config.is_point else 0) +
                 config.is_objectness + config.is_centerness)
        self.oc = oc
        self.oc_per_scale = [oc * n for n in n_anchors_per_layer]
        self.n_anchors_per_layer = n_anchors_per_layer

        if self.config.single_classifier_head:
            # Single classifier for all the layers -- feature scaling is used
            oc = self.oc_per_scale[0]
            self._scales = nn.ParameterList([nn.Parameter(torch.tensor(1.))
                                             for _ in self.oc_per_scale])
            self._classifier = nn.Sequential(nn.Conv2d(ic, ic, 1), nn.PReLU(),
                                             nn.Conv2d(ic, oc, 1))
        else:
            # Uses a classifier head for each level
            self._classifier = nn.ModuleList([
                nn.Sequential(nn.Conv2d(ic, ic, 1), nn.PReLU(),
                              nn.Conv2d(ic, oc, 1))
                for oc in self.oc_per_scale])

    def forward(self, *args):
        if self.config.single_classifier_head:
            responses = []
            for i, x in enumerate(args):
                responses.append(torch.cat([
                    self._classifier(x * self._scales[i * len(args) + j])
                    for j in range(self.n_anchors_per_layer)], 1))
        else:
            assert len(self._classifier) == len(args)
            responses = [self._classifier[i](x) for i, x in enumerate(args)]

        # Organize to have all the outputs aligned with AnchorDetector.centers
        label, boxes, point, centerness, objectness = [], [], [], [], []
        for x in responses:
            x = x.view(x.size(0), -1, self.oc, x.size(2), x.size(3))
            x = x.permute(0, 1, 3, 4, 2).contiguous()
            x = x.view(x.size(0), x.size(1), -1, x.size(-1)).contiguous()
            for i in range(x.size(1)):
                label.append(x[:, i, :, :self.config.n_label])
                boxes.append(
                    x[:, i, :, self.config.n_label:self.config.n_label+4])
                if self.config.is_point:
                    n = self.config.n_label+4
                    point.append(x[:, i, :, n:n + self.config.n_point * 2])
                if self.config.is_objectness:
                    objectness.append(
                        x[:, i, :, -(2 if self.config.is_centerness else 1)])
                if self.config.is_centerness:
                    centerness.append(x[:, i, :, -1])

        label, boxes = torch.cat(label, 1), torch.cat(boxes, 1)
        label = label.sigmoid() if label.size(-1) > 1 else label
        if "iou" in self.config.boxes_loss_kwargs["method"]:
            boxes = F.relu(boxes)
        elif (self.config.boxes_encode_format == "normalized_gcxcywh" and
              self.config.boxes_encode_var1 is None):
            boxes[:, :, :2] = torch.tanh(boxes[:, :, :2])
        return Responses(
            label=label,
            score=None,
            boxes=boxes,
            point=torch.cat(point, 1) if self.config.is_point else None,
            objectness=(torch.cat(objectness, 1).sigmoid() if
                        self.config.is_objectness else None),
            centerness=(torch.cat(centerness, 1).sigmoid() if
                        self.config.is_centerness else None))


class AnchorDetector(nn.Module):
    r""" A common detection module on top of base network with NoFPN,
    BiFPN, FPN, and PAFPN.

    .. code-block:: none

        Base is the backbone network (a pretrained or a custom one)

            Ex: ResNet-18
            1x3x224x224     1x64x56x56   1x128x28x28   1x256x14x14   1x512x7x7
               input    ->      o     ->      o     ->      o     ->      o
                                x1            x2            x3            x4
            Lets call x1, x2, x3, x4 as levels.

        Base2Body has one 1x1 convolutional layer per level to convert the
        depth of (x1, x2, x3, x4) to a constant depth (config.encoding_depth)

        Ex: config.encoding_depth = 60
        Base2Body((x1, x2, x3, x4))[0].shape == [1, 60, 56, 56]
        Base2Body((x1, x2, x3, x4))[1].shape == [1, 60, 28, 28]
        Base2Body((x1, x2, x3, x4))[2].shape == [1, 60, 14, 14]
        Base2Body((x1, x2, x3, x4))[3].shape == [1, 60,  7,  7]

        Body can have stacks of NoFPN/FPN/BiFPN/PAFPN layers. Essentially,
        these act as context layers that are interconnected across levels
        (exception is NoFPN layer).
    """
    def __init__(self, config: CONFIG):
        super(AnchorDetector, self).__init__()

        self.config = config
        self.t_size = config.t_size
        # ------------------------------------------------------------------- #
        # Base
        # ------------------------------------------------------------------- #
        if isinstance(config.base_network, str):
            if config.base_network in ("mnas_050", "mnas_100"):
                from ..architectures import MNAS as Base
            elif config.base_network == "mobilev2":
                from ..architectures import MobileNetV2 as Base
            else:
                raise ValueError("AnchorDetector: config.base_network must be "
                                 "mnas_050/mnas_100 when str")
        elif (hasattr(config.base_network, "__base__") and
              config.base_network.__base__ == nn.Module):
            Base = config.base_network
        else:
            raise TypeError("AnchorDetector: config.base_network must be str "
                            "or nn.Module")
        self.base = Base(config=config)
        # Find output tensor sizes at each level of the Base given input size
        self.c_sizes = [x.shape for x in self.base(torch.rand(*config.t_size))]
        assert len(self.c_sizes) == len(config.anchors_per_layer)

        # currently disabled
        if config.base_extension:
            modules = []
            for x in range(config.base_extension):
                modules += [MBBlock(self.c_sizes[-1], 3, self.c_sizes[1], 2)]
                tensor = torch.rand(*self.c_sizes[-1])
                self.c_sizes += [modules[-1](tensor).shape]
            self.base_extension = nn.ModuleList(modules)

        # ------------------------------------------------------------------- #
        # Base2Body
        # ------------------------------------------------------------------- #
        self.base_2_body = nn.ModuleList(
            [nn.Conv2d(sz[1], config.encoding_depth, 1)
             for sz in self.c_sizes])

        # ------------------------------------------------------------------- #
        # Body
        # ------------------------------------------------------------------- #
        if "_nofpn" in config.body_network:
            Body = NoFPNLayer
        elif "_bifpn" in config.body_network:
            Body = BiFPNLayer
        elif "_fpn" in config.body_network:
            Body = FPNLayer
        elif "_pafpn" in config.body_network:
            Body = PAFPNLayer
        else:
            raise NotImplementedError
        self.body = nn.ModuleList([
            Body(config) for _ in range(config.body_network_depth)])

        # ------------------------------------------------------------------- #
        # Classifier
        # ------------------------------------------------------------------- #
        self.classifier = Classifier(config)

        # ------------------------------------------------------------------- #
        # Loss functions
        # ------------------------------------------------------------------- #
        self.label_loss = config.label_loss_fn(**config.label_loss_kwargs)
        self.boxes_loss = config.boxes_loss_fn(**config.boxes_loss_kwargs)
        self.point_loss = config.point_loss_fn(**config.point_loss_kwargs)

        self.compute_anchors()
        self.register_buffer("_counter", torch.tensor(0))

    def forward(self, tensor: Tensor):
        responses = self.base(tensor)

        if self.config.t_size[2:] != tensor.shape[2:]:
            # update for input size changes during prediction
            self.t_size = tensor.shape
            self.c_sizes = [x.shape for x in responses]

        if self.config.ignore_base > self._counter:
            # ignore's backpropagation to base network for config.ignore_base
            # iterations -- used for pretrained networks
            self._counter += 1
            responses = [x.detach() for x in responses]

        if self.config.base_extension:
            for cnn in self.base_extension:
                responses.append(cnn(responses[-1]))

        responses = [cnn(o) for cnn, o in zip(self.base_2_body, responses)]
        for cnn in self.body:
            responses = cnn(*responses)
        return self.classifier(*responses), responses

    def predict(self, tensor: Tensor):
        r"""Calls AnchorDetector.batch_detect with no grads.

        Args:
            tensor (torch.Tensor): input tensor in BCHW

        :rtype: :class:`tensormonk.detection.Responses`
        """
        with torch.no_grad():
            responses, _ = self(tensor)
            responses = self.batch_detect(responses.label, responses.boxes,
                                          responses.point)
        return responses

    def compute_loss(self,
                     tensor: Tensor,
                     r_label: tuple,
                     r_boxes: tuple,
                     r_point: tuple):

        with torch.no_grad():
            # encoding raw label/boxes/point to targets for network
            targets = self.batch_encode(r_label, r_boxes, r_point)
            valid = targets.label.view(-1).gt(0)

        responses, body_network_responses = self(tensor)
        losses = {"label": None, "boxes": None, "point": None,
                  "objectness": None, "centerness": None}
        losses["label"] = self.label_loss(predictions=responses.label,
                                          targets=targets.label)
        losses["boxes"] = self.boxes_loss(
            p_boxes=responses.boxes, t_boxes=targets.boxes,
            t_label=targets.label,
            weights=(responses.centerness if self.config.is_centerness else
                     None))
        if self.config.is_point:
            losses["point"] = self.point_loss(p_point=responses.point,
                                              t_point=targets.point,
                                              t_label=targets.label,
                                              anchor_wh=self.anchor_wh)
        if self.config.is_objectness:
            losses["objectness"] = F.binary_cross_entropy(
                responses.objectness.view(-1), targets.objectness.view(-1))
        if self.config.is_centerness:
            losses["centerness"] = F.binary_cross_entropy(
                responses.centerness.view(-1)[valid],
                targets.centerness.view(-1)[valid])
        if self.config.body_network_return_responses:
            losses["body_network_responses"] = body_network_responses
        return losses

    def batch_encode(self,
                     r_label: Union[list, tuple],
                     r_boxes: Union[list, tuple],
                     r_point: Union[list, tuple]):
        r"""Encode's raw labels, boxes and points of a batch of images.

        Args:
            r_label (list/tuple): list/tuple of tensor's to encode.
                See encode for more information
            r_boxes (list/tuple): list/tuple of tensor's to encode.
                See encode for more information
            r_point (list/tuple): list/tuple of tensor's to encode.
                See encode for more information

        :rtype: :class:`tensormonk.detection.Responses`
        """

        # batch encode
        assert isinstance(r_label, (list, tuple))
        assert isinstance(r_boxes, (list, tuple))
        assert isinstance(r_point, (list, tuple)) or r_point is None

        t_label, t_boxes, t_point = [], [], []
        t_objectness, t_centerness = [], []
        for i in range(len(r_label)):
            targets = self.encode(r_label[i].clone(), r_boxes[i].clone(),
                                  None if r_point is None else
                                  r_point[i].clone())
            t_label.append(targets.label)
            t_boxes.append(targets.boxes)
            t_point.append(targets.point)
            t_objectness.append(targets.objectness)
            t_centerness.append(targets.centerness)
        return Responses(
            label=torch.stack(t_label),
            score=None,
            boxes=torch.stack(t_boxes),
            point=None if r_point is None else torch.stack(t_point),
            objectness=torch.stack(t_objectness),
            centerness=torch.stack(t_centerness))

    def encode(self,
               r_label: Tensor,
               r_boxes: Tensor,
               r_point: Tensor):
        r"""Encode's raw labels, boxes and points of a single image.

        Args:
            r_label (Tensor): label for each object (0 is background)
            r_boxes (Tensor): ltrb boxes of each object (pixel coordinates
                without any normalization)
            r_point (Tensor): x, y, x, y, ... for each object (pixel
                coordinates without any normalization), nan's are avoided in
                loss computation.

        :rtype: :class:`tensormonk.detection.Responses`
        """

        assert isinstance(r_label, Tensor) and isinstance(r_boxes, Tensor)
        assert isinstance(r_point, Tensor) or r_point is None
        device = self.centers.device
        r_label, r_boxes = r_label.to(device), r_boxes.to(device)

        # compute ious
        ious = ObjectUtils.compute_iou(
            torch.cat((self.centers - self.anchor_wh / 2,
                       self.centers + self.anchor_wh / 2), 1), r_boxes)
        boxes2centers_mapping = ious.max(1)[1].view(-1)

        # compute objectness -- intersection over foreground
        objectness = ObjectUtils.compute_objectness(
            self.centers, self.pix2pix_delta, r_boxes)

        # compute centerness
        centerness = ObjectUtils.compute_centerness(
            self.centers, r_boxes, boxes2centers_mapping)

        # Filter 1: targets based on encode_iou
        t_label = r_label[boxes2centers_mapping]
        t_label[ious.max(1)[0] < self.config.encode_iou] = 0

        # Filter 2: check if center lies within -1 to 1 pixel as tanh is used.
        # However, this creates an issue if anchor w & h are way higher than
        # strides
        valid = t_label.nonzero().view(-1)
        if valid.numel() != 0:
            idx = ious[valid].max(1)[1].view(-1)
            x_delta = self.centers[valid, 0] - r_boxes[idx, 0::2].mean(1)
            y_delta = self.centers[valid, 1] - r_boxes[idx, 1::2].mean(1)
            if self.config.hard_encode:
                valid_centers = (
                    (x_delta.abs() < self.pix2pix_delta[valid, 0]) *
                    (y_delta.abs() < self.pix2pix_delta[valid, 1]))
            else:
                valid_centers = (
                    (x_delta.abs() < self.anchor_wh[valid, 0]) *
                    (y_delta.abs() < self.anchor_wh[valid, 1]))
            if (~ valid_centers).all():
                t_label[idx[~ valid_centers]] = 0

        # encode boxes
        valid = t_label.nonzero().view(-1)
        t_boxes = torch.zeros(self.centers.size(0), 4).to(device)
        if valid.numel() != 0:
            t_boxes[valid] = ObjectUtils.encode_boxes(
                self.config.boxes_encode_format,
                self.centers, self.pix2pix_delta, self.anchor_wh,
                r_boxes, boxes2centers_mapping,
                self.config.boxes_encode_var1,
                self.config.boxes_encode_var2)[valid]

        # encode points
        t_point = None
        if r_point is not None:
            t_point = ObjectUtils.encode_point(
                self.config.point_encode_format, self.centers,
                self.pix2pix_delta, self.anchor_wh,
                r_point, boxes2centers_mapping,
                self.config.point_encode_var)
            t_point[t_label.eq(0)] = 0.

        return Responses(label=t_label,
                         score=None,
                         boxes=t_boxes,
                         point=t_point,
                         objectness=objectness,
                         centerness=centerness)

    def batch_detect(self, p_label: Tensor, p_boxes: Tensor, p_point: Tensor):
        r"""A list of Responses from detect.

        Args:
            p_label (Tensor): label predictions at each pixel for all levels
            p_boxes (Tensor): boxes predictions at each pixel for all levels
            p_point (Tensor): boxes predictions at each pixel for all levels

            p_label.size(0) == p_boxes.size(0) == p_point.size(0) ==
                self.centers.size(0)

        :rtype: [:class:`tensormonk.detection.Responses`,
            :class:`tensormonk.detection.Responses`, ...]
        """
        # batch detect
        assert isinstance(p_label, Tensor) and isinstance(p_boxes, Tensor)
        assert isinstance(p_point, Tensor) or p_point is None
        assert p_label.size(1) == p_boxes.size(1)

        detections = []
        for i in range(p_label.size(0)):
            detections.append(self.detect(
                p_label[i], p_boxes[i],
                None if p_point is None else p_point[i]))
        return detections

    def detect(self, p_label: Tensor, p_boxes: Tensor, p_point: Tensor):
        r"""Detects labels, boxes and points of a single image.

        Args:
            p_label (Tensor): label predictions at each pixel for all levels
            p_boxes (Tensor): boxes predictions at each pixel for all levels
            p_point (Tensor): boxes predictions at each pixel for all levels

            p_label.size(0) == p_boxes.size(0) == p_point.size(0) ==
                self.centers.size(0)

        :rtype: :class:`tensormonk.detection.Responses`
        """
        if self.t_size[1:] != self.config.t_size[1:]:
            centers, pix2pix_delta, anchor_wh = self.compute_anchors()
        else:
            centers, pix2pix_delta, anchor_wh = self.centers, \
                self.pix2pix_delta, self.anchor_wh

        assert isinstance(p_label, Tensor) and isinstance(p_boxes, Tensor)
        assert p_label.ndim == 1 or p_label.ndim == 2
        assert p_boxes.ndim == 2 and p_boxes.size(-1) == 4
        assert p_label.size(0) == p_boxes.size(0) == centers.size(0)
        assert isinstance(p_point, Tensor) or p_point is None

        if p_label.ndim == 2 and p_label.size(1) == 1:
            p_label = p_label.view(-1)

        if p_label.ndim == 2:
            # pick top_n objects
            sorted_scores, sorted_idx = torch.sort(p_label, dim=1)
            sorted_scores = sorted_scores[:, -2:]
            sorted_idx = sorted_idx[:, -2:]
            # pick best non-background per location
            label = sorted_idx[:, 1]
            label[label == 0] = sorted_idx[:, 0][label == 0]
            score = p_label.gather(1, label.view(-1, 1)).view(-1)
        else:
            score = p_label
            if not (score.max() <= 1 and score.min() >= 0):
                score = torch.sigmoid(score)
            label = p_label.mul(0).add(1).long()

        # decode boxes
        boxes = ObjectUtils.decode_boxes(
            self.config.boxes_encode_format,
            centers, pix2pix_delta, anchor_wh, p_boxes,
            self.config.boxes_encode_var1,
            self.config.boxes_encode_var2)
        # nms
        retain = torchvision.ops.nms(boxes, score, self.config.detect_iou)

        # score thresholding
        if self.config.score_threshold > 0 and retain.numel() > 0:
            valid_score = (score[retain] > self.config.score_threshold)
            valid_score = valid_score.view(-1)
            if valid_score.sum() == 0:
                # when no objects pass score threshold, pick best available
                valid_score = score[retain] == score[retain].max()
            retain = retain[valid_score]

        if p_point is not None:
            point = ObjectUtils.decode_point(
                self.config.point_encode_format,
                centers, pix2pix_delta, anchor_wh, p_point,
                self.config.point_encode_var)

        if retain.numel() == 0:
            return Responses(label=None, score=None, boxes=None, point=None,
                             objectness=None, centerness=None)

        return Responses(label=label[retain],
                         score=score[retain],
                         boxes=boxes[retain],
                         point=point[retain] if p_point is not None else None,
                         objectness=None,
                         centerness=None)

    def compute_anchors(self):
        assert len(self.c_sizes) == len(self.config.anchors_per_layer)
        centers, pix2pix_delta, anchor_wh = [], [], []

        for c_size, anchors in zip(self.c_sizes,
                                   self.config.anchors_per_layer):
            cs = ObjectUtils.centers_per_layer(self.t_size, c_size,
                                               self.config.is_pad)
            for an_anchor in anchors:
                zeros = torch.zeros(cs.size(0))
                centers.append(cs)
                # x and y limits at a pixel
                pix2pix_delta.append(torch.stack((
                    zeros + (cs[1, 0] - cs[0, 0]),
                    zeros + (cs[c_size[3], 1] - cs[0, 1])), 1))
                # anchor width and height for normalization
                anchor_wh.append(
                    torch.stack((zeros + an_anchor.w, zeros + an_anchor.h), 1))

        if hasattr(self, "centers"):
            # For on the fly computation when input size changes
            device = self.centers.device
            return (torch.cat(centers).to(device),
                    torch.cat(pix2pix_delta).to(device),
                    torch.cat(anchor_wh).to(device))
        self.register_buffer("centers", torch.cat(centers))
        self.register_buffer("pix2pix_delta", torch.cat(pix2pix_delta))
        self.register_buffer("anchor_wh", torch.cat(anchor_wh))
