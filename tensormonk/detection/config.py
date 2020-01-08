""" TensorMONK's :: detection :: CONFIG """

__all__ = ["CONFIG"]

import torch
from collections import namedtuple
from .. import layers, loss


class CONFIG:

    def __init__(self, name: str):
        self.name = name

        # ------------------------------------------------------------------- #
        # network options
        self._base_network_pretrained = None
        self._base_network = None
        self._base_network_options = ("mnas_050",
                                      "mnas_100",
                                      "mobilev2")
        self._base_network_forced_stride = False
        self._base_extension = 0  # not enabled
        self._body_network = None
        self._body_network_options = ("anchor_nofpn",
                                      "anchor_fpn",
                                      "anchor_pafpn",
                                      "anchor_bifpn")
        self._body_network_depth = 2
        # Refer tensormonk.layers.FeatureFusion.METHODS for fusion options
        self._body_fpn_fusion = "softmax"
        self._anchors_per_layer = None
        self._body_network_return_responses = False

        # ------------------------------------------------------------------- #
        # input size and feature encoding size
        self._t_size = None  # BCHW - Ex: (1, 3, 256, 256)
        self._encoding_depth = None

        # ------------------------------------------------------------------- #
        self._single_classifier_head = False

        # ------------------------------------------------------------------- #
        # info on labels
        self._n_label = None
        # loss function - refer tensormonk.loss.LabelLoss
        self._label_loss_fn = loss.LabelLoss
        self._label_loss_kwargs = None

        # ------------------------------------------------------------------- #
        # info on boxes
        self._is_boxes = None
        # loss function - refer tensormonk.loss.BoxesLoss
        self._boxes_loss_fn = loss.BoxesLoss
        self._boxes_loss_kwargs = None
        # target boxes transformation format and prediction boxes format
        self._boxes_encode_format = None
        self._boxes_encode_format_options = (
            "normalized_offset",
            "normalized_gcxcywh")

        # ------------------------------------------------------------------- #
        self._is_point = None
        self._n_point = None
        # loss function - refer tensormonk.loss.PointLoss
        self._point_loss_fn = loss.PointLoss
        self._point_loss_kwargs = None
        # target points transformation format and prediction points format
        self._point_encode_format = "normalized_xy_offsets"
        self._point_encode_format_options = (
            "normalized_xy_offsets")

        # ------------------------------------------------------------------- #
        # FCOS
        self._is_centerness = False
        # YoloV3 - Intersection (of pixel and any box) over area of the pixel
        self._is_objectness = False
        self._hard_encode = False
        self._encode_iou = 0.5
        self._detect_iou = 0.2
        self._score_threshold = 0.1

        # ------------------------------------------------------------------- #
        self._boxes_encode_var1 = 0.1
        self._boxes_encode_var2 = 0.2

        self._anchors_per_layer = None
        self._an_anchor = namedtuple("anchor", ("w", "h"))
        self._ignore_base = 0

    @property
    def base_network(self):
        return self._base_network

    @base_network.setter
    def base_network(self, value):
        assert value in self._base_network_options or \
            value.__base__ == torch.nn.Module
        value = value.lower() if isinstance(value, str) else value
        self._base_network = value

    @property
    def base_network_pretrained(self):
        return self._base_network_pretrained

    @base_network_pretrained.setter
    def base_network_pretrained(self, value):
        assert isinstance(value, bool)
        self._base_network_pretrained = value

    @property
    def base_network_forced_stride(self):
        return self._base_network_forced_stride

    @base_network_forced_stride.setter
    def base_network_forced_stride(self, value):
        assert isinstance(value, bool)
        self._base_network_forced_stride = value

    @property
    def base_extension(self):
        return self._base_extension

    @base_extension.setter
    def base_extension(self, value):
        assert isinstance(value, int) and value >= 0
        self._base_extension = value

    @property
    def body_network(self):
        return self._body_network

    @body_network.setter
    def body_network(self, value):
        assert value in self._body_network_options
        self._body_network = value.lower()

    @property
    def body_network_depth(self):
        return self._body_network_depth

    @body_network_depth.setter
    def body_network_depth(self, value):
        assert isinstance(self._body_network_depth, int)
        assert self._body_network_depth >= 1
        self._body_network_depth = value

    @property
    def body_fpn_fusion(self):
        return self._body_fpn_fusion

    @body_fpn_fusion.setter
    def body_fpn_fusion(self, value):
        assert isinstance(self._body_fpn_fusion, str)
        assert self._body_fpn_fusion in layers.FeatureFusion.METHODS
        self._body_fpn_fusion = value

    @property
    def body_network_return_responses(self):
        return self._body_network_return_responses

    @body_network_return_responses.setter
    def body_network_return_responses(self, value):
        assert isinstance(self._body_network_return_responses, bool)
        self._body_network_return_responses = value

    @property
    def t_size(self):
        return self._t_size

    @t_size.setter
    def t_size(self, value):
        assert isinstance(value, (list, tuple)) and len(value) == 4
        value = list(value)
        value[0] = 1
        self._t_size = tuple(value)

    @property
    def encoding_depth(self):
        return self._encoding_depth

    @encoding_depth.setter
    def encoding_depth(self, value):
        assert isinstance(value, int) and value >= 8
        self._encoding_depth = value

    @property
    def single_classifier_head(self):
        return self._single_classifier_head

    @single_classifier_head.setter
    def single_classifier_head(self, value):
        assert isinstance(value, bool)
        self._single_classifier_head = value

    @property
    def n_label(self):
        return self._n_label

    @n_label.setter
    def n_label(self, value):
        assert isinstance(value, int) and value > 1
        self._n_label = value

    @property
    def label_loss_fn(self):
        return self._label_loss_fn

    @label_loss_fn.setter
    def label_loss_fn(self, value):
        assert value.__base__ == torch.nn.Module
        self._label_loss_fn = value

    @property
    def label_loss_kwargs(self):
        return self._label_loss_kwargs

    @label_loss_kwargs.setter
    def label_loss_kwargs(self, value):
        assert isinstance(value, dict)
        assert all([x in loss.LabelLoss.KWARGS for x in value.keys()])
        assert value["method"] in loss.LabelLoss.METHODS
        self._label_loss_kwargs = value

    @property
    def is_boxes(self):
        return self._is_boxes

    @is_boxes.setter
    def is_boxes(self, value):
        assert isinstance(value, bool)
        self._is_boxes = value

    @property
    def boxes_loss_fn(self):
        return self._boxes_loss_fn

    @boxes_loss_fn.setter
    def boxes_loss_fn(self, value):
        assert value.__base__ == torch.nn.Module
        self._boxes_loss_fn = value

    @property
    def boxes_loss_kwargs(self):
        return self._boxes_loss_kwargs

    @boxes_loss_kwargs.setter
    def boxes_loss_kwargs(self, value):
        assert isinstance(value, dict)
        assert all([x in loss.BoxesLoss.KWARGS for x in value.keys()])
        assert value["method"] in loss.BoxesLoss.METHODS
        self._boxes_loss_kwargs = value

    @property
    def boxes_encode_format(self):
        return self._boxes_encode_format

    @boxes_encode_format.setter
    def boxes_encode_format(self, value):
        assert value in self._boxes_encode_format_options
        assert self.boxes_loss_kwargs is not None, \
            "boxes_loss_kwargs must be set before boxes_encode_format"
        if "iou" in self.boxes_loss_kwargs["method"]:
            assert "normalized_offset" in value, \
                "iou based loss requires boxes_encode_format=normalized_offset"
        self._boxes_encode_format = value

    @property
    def is_point(self):
        return self._is_point

    @is_point.setter
    def is_point(self, value):
        assert isinstance(value, bool)
        self._is_point = value

    @property
    def n_point(self):
        return self._n_point

    @n_point.setter
    def n_point(self, value):
        assert isinstance(value, int)
        self._n_point = value

    @property
    def point_loss_fn(self):
        return self._point_loss_fn

    @point_loss_fn.setter
    def point_loss_fn(self, value):
        assert value.__base__ == torch.nn.Module
        self._point_loss_fn = value

    @property
    def point_loss_kwargs(self):
        return self._point_loss_kwargs

    @point_loss_kwargs.setter
    def point_loss_kwargs(self, value):
        assert isinstance(value, dict)
        assert all([x in loss.PointLoss.KWARGS for x in value.keys()])
        assert value["method"] in loss.PointLoss.METHODS
        self._point_loss_kwargs = value

    @property
    def point_encode_format(self):
        return self._point_encode_format

    @point_encode_format.setter
    def point_encode_format(self, value):
        assert value in self._point_encode_format_options
        self._point_encode_format = value

    @property
    def is_centerness(self):
        return self._is_centerness

    @is_centerness.setter
    def is_centerness(self, value):
        assert isinstance(value, bool)
        self._is_centerness = value

    @property
    def is_objectness(self):
        return self._is_objectness

    @is_objectness.setter
    def is_objectness(self, value):
        assert isinstance(value, bool)
        self._is_objectness = value

    @property
    def hard_encode(self):
        return self._hard_encode

    @hard_encode.setter
    def hard_encode(self, value):
        assert isinstance(value, bool)
        self._hard_encode = value

    @property
    def encode_iou(self):
        return self._encode_iou

    @encode_iou.setter
    def encode_iou(self, value):
        assert isinstance(value, float)
        self._encode_iou = value

    @property
    def detect_iou(self):
        return self._detect_iou

    @detect_iou.setter
    def detect_iou(self, value):
        assert isinstance(value, float)
        self._detect_iou = value

    @property
    def score_threshold(self):
        return self._score_threshold

    @score_threshold.setter
    def score_threshold(self, value):
        assert isinstance(value, float)
        self._score_threshold = value

    @property
    def boxes_encode_var1(self):
        return self._boxes_encode_var1

    @boxes_encode_var1.setter
    def boxes_encode_var1(self, value):
        assert isinstance(value, float)
        self._boxes_encode_var1 = value

    @property
    def boxes_encode_var2(self):
        return self._boxes_encode_var2

    @boxes_encode_var2.setter
    def boxes_encode_var2(self, value):
        assert isinstance(value, float)
        self._boxes_encode_var2 = value

    @property
    def ignore_base(self):
        return self._ignore_base

    @ignore_base.setter
    def ignore_base(self, value):
        assert isinstance(value, int)
        self._ignore_base = value

    @property
    def anchors_per_layer(self):
        return self._anchors_per_layer

    @anchors_per_layer.setter
    def anchors_per_layer(self, value):
        assert isinstance(value, (list, tuple))
        for x in value:
            assert isinstance(x, (list, tuple))
            for y in x:
                assert isinstance(y, self._an_anchor)
        self._anchors_per_layer = value

    def an_anchor(self, w, h):
        return self._an_anchor(w, h)

    def __repr__(self):
        msg = ["CONFIG :: {}".format(self.name)]
        msg += ["Base = {}".format(self.base_network)]
        msg += ["Body = {}".format(self.body_network)]
        msg += ["t_size = {}".format("x".join(map(str, self.t_size)))]
        msg += ["n_label = {}".format(self.n_label)]
        msg += ["LabelLoss = {}".format(self.label_loss_kwargs["method"])]
        if self.is_boxes:
            msg += ["BoxesLoss = {}".format(self.boxes_loss_kwargs["method"])]
        if self.is_point:
            msg += ["PointLoss = {}".format(self.boxes_loss_kwargs["method"])]
        msg += ["Objectness is " + (" ON" if self.is_objectness else "OFF") +
                " & Centerness is " + (" ON" if self.is_centerness else "OFF")]
        return "\n\t".join(msg)
