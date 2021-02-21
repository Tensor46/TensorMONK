""" TensorMONK's :: detection :: CONFIG """

__all__ = ["CONFIG"]

import torch
from collections import namedtuple
from .. import layers, loss


class CONFIG:
    r"""CONFIG is used to configure all the options for object detection tasks.

    Example: Assume an object detection model that is trained on 320x320 images
    to detect dogs and cats.

    .. code-block:: python

        import tensormonk

        config = tensormonk.detection.CONFIG("mnas_bifpn_dogs_cats")

        # Define input size
        config.t_size = (1, 3, 320, 320)

        # Use pretrained MNAS model as base network
        config.base_network = "mnas_100"
        config.base_network_pretrained = True
        # Given the above config and input size of (4, 3, 320, 320), base
        # network will return a tuple of tensor's of shape
        # ((4, 24, 80, 80), (4, 40, 40, 40), (4, 96, 20, 20), (4, 320, 10, 10))
        # By using base_network_forced_stride, base network will return a tuple
        # of tensor's of shape
        # ((4, 24, 40, 40), (4, 40, 20, 20), (4, 96, 10, 10), (4, 320, 5, 5)).
        config.base_network_forced_stride = True

        # All the ouputs from base network are encoded to have constant depth
        # (96) using a 1x1 convolution per level.
        # Essentially, the base network output with tensor shapes
        # ((4, 24, 40, 40), (4, 40, 20, 20), (4, 96, 10, 10) and (4, 320,5, 5))
        # is converted to
        # ((4, 96, 40, 40), (4, 96, 20, 20), (4, 96, 10, 10) and (4, 96, 5, 5))
        config.encoding_depth = 96

        # Define a body network with 4 "bifpn" layers.
        config.body_network = "bifpn"
        config.body_network_depth = 4

        # Define number of labels (labels to detect + background)
        config.n_label = 2 + 1
        config.label_loss_fn = tensormonk.loss.LabelLoss
        config.label_loss_kwargs = {
            "method": "ce_with_negative_mining",
            "pos_to_neg_ratio": 1 / 3.,
            "reduction": "mean"}

        # Define loss function and encoding for bounding box
        config.is_boxes = True
        config.boxes_loss_fn = tensormonk.loss.BoxesLoss
        config.boxes_loss_kwargs = {
            "method": "smooth_l1", "reduction": "mean"}
        config.boxes_encode_format = "normalized_offset"

        # Enable objectness and disable centerness
        config.is_point = False
        config.is_objectness = True
        config.is_centerness = False

        # Define encode_iou
        # minimum iou required for a prior to set a location as non background
        config.encode_iou = 0.5
        # Define detect_iou - iou_threshold for non-maximal suppression
        config.detect_iou = 0.2
        # Define score_threshold - minimum score required to label an anchor as
        # non background during inference.
        config.score_threshold = 0.46
        # Define ignore_base - As a pretrained base network is used in this
        # example, disable the gradients to reach base_network for 5000
        # iterations.
        config.ignore_base = 5000

        # Define anchors
        config.anchors_per_layer = (
            # anchors at 40x40
            (config.an_anchor(32,   32), config.an_anchor(46,   46)),
            # anchors at 20x20
            (config.an_anchor(64,   64), config.an_anchor(90,   90)),
            # anchors at 10x10
            (config.an_anchor(128, 128), config.an_anchor(180, 180)),
            # anchors at 5x5
            (config.an_anchor(256, 256), config.an_anchor(320, 320))
        print(config)

    """

    def __init__(self, name: str):
        self.name = name

        # ------------------------------------------------------------------- #
        # network options
        self._base_network_pretrained = False
        self._base_network = "mnas_050"
        self._base_network_options = ("mnas_050",
                                      "mnas_100",
                                      "mobilev2")
        self._base_network_forced_stride = False
        self._base_extension = 0  # not enabled
        self._body_network = "bifpn"
        self._body_network_options = ("nofpn", "fpn", "pafpn", "bifpn")
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
        self._label_loss_kwargs = {}

        # ------------------------------------------------------------------- #
        # info on boxes
        self._is_boxes = True
        # loss function - refer tensormonk.loss.BoxesLoss
        self._boxes_loss_fn = loss.BoxesLoss
        self._boxes_loss_kwargs = {}
        # target boxes transformation format and prediction boxes format
        self._boxes_encode_format = "normalized_gcxcywh"
        self._boxes_encode_format_options = (
            "normalized_offset",
            "normalized_gcxcywh")

        # ------------------------------------------------------------------- #
        self._is_point = False
        self._n_point = None
        # loss function - refer tensormonk.loss.PointLoss
        self._point_loss_fn = loss.PointLoss
        self._point_loss_kwargs = {}
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
        self._encode_iou_max_background = self.encode_iou - 0.1
        self._detect_iou = 0.2
        self._score_threshold = 0.1

        # ------------------------------------------------------------------- #
        self._boxes_encode_var1 = 0.1
        self._boxes_encode_var2 = 0.2
        self._point_encode_var = 0.5
        self._is_pad = True

        self._anchors_per_layer = None
        self._an_anchor = namedtuple("anchor", ("w", "h", "offset"))
        self._ignore_base = 0

    @property
    def base_network(self):
        r"""Base network for anchor detector (str/nn.Module).

        Args:
            value (str, optional): Current options are :obj:`"mnas_050"`,
                :obj:`"mnas_100"`, and :obj:`"mobilev2"`.
                See :class:`tensormonk.architectures.MNAS`, and
                :class:`tensormonk.architectures.MobileNetV2` for more
                information. Also accept a custom network.
                default = :obj:`"mnas_050"`.

        Example custom network:

        .. code-block:: python

            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            import tensormonk


            class Tiny(torch.nn.Module):
                def __init__(self, **kwargs):
                    super(Tiny, self).__init__()
                    self._layer_0 = torch.nn.Sequential(
                        nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.PReLU(),
                        nn.Conv2d(16, 16, 3, stride=2, padding=1), nn.PReLU())
                    self._layer_1 = torch.nn.Sequential(
                        nn.Conv2d(16, 24, 3, stride=2, padding=1), nn.PReLU(),
                        nn.Conv2d(24, 24, 3, stride=1, padding=1), nn.PReLU())
                    self._layer_2 = torch.nn.Sequential(
                        nn.Conv2d(24, 32, 3, stride=2, padding=1), nn.PReLU(),
                        nn.Conv2d(32, 32, 3, stride=1, padding=1), nn.PReLU())
                    self._layer_3 = torch.nn.Sequential(
                        nn.Conv2d(32, 48, 3, stride=2, padding=1), nn.PReLU(),
                        nn.Conv2d(48, 48, 3, stride=1, padding=1), nn.PReLU())

                def forward(self, tensor: torch.Tensor):
                    x0 = self._layer_0(tensor)
                    x1 = self._layer_1(x0)
                    x2 = self._layer_2(x1)
                    x3 = self._layer_3(x2)
                    return (x1, x2, x3)


            config = tensormonk.detection.CONFIG("tiny")
            config.base_network = Tiny
        """
        return self._base_network

    @base_network.setter
    def base_network(self, value):
        assert value in self._base_network_options or \
            value.__base__ == torch.nn.Module
        value = value.lower() if isinstance(value, str) else value
        self._base_network = value

    @property
    def base_network_pretrained(self):
        r"""Used when base_network is :obj:`"mnas_050"`, :obj:`"mnas_100"`, or
        :obj:`"mobilev2"` to load pretrained weights.

        Args:
            value (bool, optional): default = :obj:`True`
        """
        return self._base_network_pretrained

    @base_network_pretrained.setter
    def base_network_pretrained(self, value):
        assert isinstance(value, bool)
        self._base_network_pretrained = value

    @property
    def base_network_forced_stride(self):
        r"""Used when base_network is :obj:`"mnas_050"`, :obj:`"mnas_100"`, or
        :obj:`"mobilev2"` to add an additional stride in the second or third
        convolution layer.

        Args:
            value (bool, optional): default = :obj:`False`
        """
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
        r"""Body network options are

        Args:
            value (str, optional): :obj:`"bifpn"`, :obj:`"fpn"`,
                :obj:`"nofpn"`, and :obj:`"pafpn"`. default = :obj:`"bifpn"`.

        :obj:`"bifpn"` = :class:`tensormonk.detection.BiFPNLayer`

        :obj:`"fpn"`   = :class:`tensormonk.detection.FPNLayer`

        :obj:`"nofpn"` = :class:`tensormonk.detection.NoFPNLayer`

        :obj:`"pafpn"` = :class:`tensormonk.detection.PAFPNLayer`
        """
        return self._body_network

    @body_network.setter
    def body_network(self, value):
        if isinstance(value, str) and value.startswith("anchor_"):
            value = value.split("_")[-1]
        assert value in self._body_network_options
        self._body_network = value.lower()

    @property
    def body_network_depth(self):
        r"""Number of FPN or NoFPN layers to stack. Below is an example
        config of body network that has 6 :obj:`"bifpn"` layers:

        .. code-block:: python

            import tensormonk
            config = tensormonk.detection.CONFIG("mnas_bifpn")
            config.base_network = "mnas_050"
            config.encoding_depth = 96
            config.body_network = "bifpn"
            config.body_network_depth = 6

        Args:
            value (int, optional): default = :obj:`2`.
        """
        return self._body_network_depth

    @body_network_depth.setter
    def body_network_depth(self, value):
        assert isinstance(self._body_network_depth, int)
        assert self._body_network_depth >= 1
        self._body_network_depth = value

    @property
    def body_fpn_fusion(self):
        r"""Fusion scheme used by FPN and NoFPN.
        See :class:`tensormonk.layers.FeatureFusion` and
        :class:`tensormonk.detection.Block` for more information.

        Args:
            value (str, optional): default = :obj:`"softmax"`. See
                :class:`tensormonk.layers.FeatureFusion` for all available
                options.
        """
        return self._body_fpn_fusion

    @body_fpn_fusion.setter
    def body_fpn_fusion(self, value):
        assert isinstance(self._body_fpn_fusion, str)
        assert self._body_fpn_fusion in layers.FeatureFusion.METHODS
        self._body_fpn_fusion = value

    @property
    def body_network_return_responses(self):
        r"""When True, compute_loss in
        :class:`tensormonk.detection.AnchorDetector` also return's the
        responses from body network.

        Args:
            value (bool, optional): default = :obj:`False`.
        """
        return self._body_network_return_responses

    @body_network_return_responses.setter
    def body_network_return_responses(self, value):
        assert isinstance(self._body_network_return_responses, bool)
        self._body_network_return_responses = value

    @property
    def t_size(self):
        r"""Input tensor size in BCHW. Also, used to precompute centers,
        anchor_wh and pix2pix_delta.

        Args:
            value (tuple, required): Input tensor shape in BCHW
                (None/any integer >0, channels, height, width).
        """
        return self._t_size

    @t_size.setter
    def t_size(self, value):
        assert isinstance(value, (list, tuple)) and len(value) == 4
        value = list(value)
        value[0] = 1
        self._t_size = tuple(value)

    @property
    def encoding_depth(self):
        r"""Encoding depth to convert all the base network outputs to a
        constant depth in order to enable FPN and NoFPN layers.

        Args:
            value (int, required): See the example in
                :class:`tensormonk.detection.AnchorDetector` for more
                information.
        """
        return self._encoding_depth

    @encoding_depth.setter
    def encoding_depth(self, value):
        assert isinstance(value, int) and value >= 8
        self._encoding_depth = value

    @property
    def single_classifier_head(self):
        r"""Flag to enable single classifier head in
        :class:`tensormonk.detection.Classifier`.

        Args:
            value (bool, optional): default = :obj:`False`. See
                :class:`tensormonk.detection.Classifier` for more
                information.
        """
        return self._single_classifier_head

    @single_classifier_head.setter
    def single_classifier_head(self, value):
        assert isinstance(value, bool)
        self._single_classifier_head = value

    @property
    def n_label(self):
        r"""Number of labels (including background) to predict.

        Args:
            value (int, required): Must be >= 2.
        """
        return self._n_label

    @n_label.setter
    def n_label(self, value):
        assert isinstance(value, int) and value > 1
        self._n_label = value

    @property
    def label_loss_fn(self):
        r"""Loss function to compute loss given p_label and t_label. This
        function is initialized in
        :class:`tensormonk.detection.AnchorDetector`. A custom loss function
        can be initialized as long as it is a nn.Module and all the
        label_loss_kwargs are set.

        Args:
            value (nn.Module, optional): default =
                :class:`tensormonk.loss.LabelLoss`
        """
        return self._label_loss_fn

    @label_loss_fn.setter
    def label_loss_fn(self, value):
        assert value.__base__ == torch.nn.Module
        self._label_loss_fn = value

    @property
    def label_loss_kwargs(self):
        r"""Dictonary of parameters required to initialize config.label_loss_fn
        function.

        Args:
            value (dict, required): See :class:`tensormonk.loss.LabelLoss` for
                more information if config.label_loss_fn is
                :class:`tensormonk.loss.LabelLoss`.
        """
        return self._label_loss_kwargs

    @label_loss_kwargs.setter
    def label_loss_kwargs(self, value):
        assert isinstance(value, dict)
        if self.label_loss_fn == loss.LabelLoss:
            assert all([x in loss.LabelLoss.KWARGS for x in value.keys()])
            assert value["method"] in loss.LabelLoss.METHODS
        self._label_loss_kwargs = value

    @property
    def is_boxes(self):
        r"""Flag to enable bounding box detection. Not used in current
        implementation (default = :obj:`"True"`), will get updated with
        inclusion of segmentation task.

        Args:
            value (bool, optional): default = :obj:`"True"`
        """
        return self._is_boxes

    @is_boxes.setter
    def is_boxes(self, value):
        assert isinstance(value, bool)
        self._is_boxes = value

    @property
    def boxes_loss_fn(self):
        r"""Loss function to compute loss given p_boxes and t_boxes. This
        function is initialized in
        :class:`tensormonk.detection.AnchorDetector`. A custom loss function
        can be initialized as long as it is a nn.Module and all the
        boxes_loss_kwargs are set.

        Args:
            value (nn.Module, optional): default =
                :class:`tensormonk.loss.BoxesLoss`
        """
        return self._boxes_loss_fn

    @boxes_loss_fn.setter
    def boxes_loss_fn(self, value):
        assert value.__base__ == torch.nn.Module
        self._boxes_loss_fn = value

    @property
    def boxes_loss_kwargs(self):
        r"""Dictonary of parameters required to initialize config.boxes_loss_fn
        function.

        Args:
            value (dict, required): See :class:`tensormonk.loss.BoxesLoss` for
                more information if config.boxes_loss_fn is
                :class:`tensormonk.loss.BoxesLoss`.
        """
        return self._boxes_loss_kwargs

    @boxes_loss_kwargs.setter
    def boxes_loss_kwargs(self, value):
        assert isinstance(value, dict)
        assert all([x in loss.BoxesLoss.KWARGS for x in value.keys()])
        assert value["method"] in loss.BoxesLoss.METHODS
        self._boxes_loss_kwargs = value

    @property
    def boxes_encode_format(self):
        r"""Boxes encoding format. See
        :class:`tensormonk.detection.ObjectUtils` for more options.

        Note: IOU based loss functions require "normalized_offset"

        Args:
            value (str, optional): Options "normalized_gcxcywh" or
                "normalized_offset". default = :obj:`"normalized_gcxcywh"`.
        """
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
        r"""Flag to enable point localization within a bounding box.

        Args:
            value (bool, optional): default = :obj:`"False"`
        """
        return self._is_point

    @is_point.setter
    def is_point(self, value):
        assert isinstance(value, bool)
        self._is_point = value

    @property
    def n_point(self):
        r"""Number of points to detect in an object. This is relavent to tasks
        like identifying body parts/joints in person detection, facial
        landmarks in face detection, etc.

        Args:
            value (int, optional): Must be >= 1 and set when config.is_point is
                True.
        """
        return self._n_point

    @n_point.setter
    def n_point(self, value):
        assert isinstance(value, int)
        self._n_point = value

    @property
    def point_loss_fn(self):
        r"""Loss function to compute loss given p_point and t_point. This
        function is initialized in
        :class:`tensormonk.detection.AnchorDetector`. A custom loss function
        can be initialized as long as it is a nn.Module and all the
        point_loss_kwargs are set.

        Args:
            value (nn.Module, optional): default =
                :class:`tensormonk.loss.PointLoss`
        """
        return self._point_loss_fn

    @point_loss_fn.setter
    def point_loss_fn(self, value):
        assert value.__base__ == torch.nn.Module
        self._point_loss_fn = value

    @property
    def point_loss_kwargs(self):
        r"""Dictonary of parameters required to initialize config.point_loss_fn
        function.

        Args:
            value (dict, required): See :class:`tensormonk.loss.PointLoss` for
                more information if config.point_loss_fn is
                :class:`tensormonk.loss.PointLoss`.
        """
        return self._point_loss_kwargs

    @point_loss_kwargs.setter
    def point_loss_kwargs(self, value):
        assert isinstance(value, dict)
        assert all([x in loss.PointLoss.KWARGS for x in value.keys()])
        assert value["method"] in loss.PointLoss.METHODS
        self._point_loss_kwargs = value

    @property
    def point_encode_format(self):
        r"""Point encoding format. See
        :class:`tensormonk.detection.ObjectUtils` for more information.

        Args:
            value (str, optional): default = :obj:`"normalized_xy_offsets"`.
        """
        return self._point_encode_format

    @point_encode_format.setter
    def point_encode_format(self, value):
        assert value in self._point_encode_format_options
        self._point_encode_format = value

    @property
    def is_centerness(self):
        r"""Enables centerness as defined in `FCOS: Fully Convolutional
        One-Stage Object Detection <https://arxiv.org/pdf/1904.01355.pdf>`_

        Args:
            value (bool, optional): default = :obj:`False`.
        """
        return self._is_centerness

    @is_centerness.setter
    def is_centerness(self, value):
        assert isinstance(value, bool)
        self._is_centerness = value

    @property
    def is_objectness(self):
        r"""Enables centerness as defined in `YOLOv3: An Incremental
        Improvement <https://pjreddie.com/media/files/papers/YOLOv3.pdf>`_.

        Args:
            value (bool, optional): default = :obj:`False`.
        """
        return self._is_objectness

    @is_objectness.setter
    def is_objectness(self, value):
        assert isinstance(value, bool)
        self._is_objectness = value

    @property
    def hard_encode(self):
        r"""Eliminates boxes with centers that are not within pix2pix_delta.

        Args:
            value (bool, optional): default = :obj:`False`.
        """
        return self._hard_encode

    @hard_encode.setter
    def hard_encode(self, value):
        assert isinstance(value, bool)
        self._hard_encode = value

    @property
    def encode_iou(self):
        r"""IOU required by a box to map it to an anchor.

        Args:
            value (float, optional): default = :obj:`0.5`.
        """
        return self._encode_iou

    @encode_iou.setter
    def encode_iou(self, value):
        assert isinstance(value, float)
        self._encode_iou = value

    @property
    def encode_iou_max_background(self):
        r"""IOU below which is considered as background.

        Args:
            value (float, optional): default = :obj:`0.5`.
        """
        return self._encode_iou_max_background

    @encode_iou_max_background.setter
    def encode_iou_max_background(self, value):
        assert isinstance(value, float)
        self._encode_iou_max_background = value

    @property
    def detect_iou(self):
        r"""IOU used to filter boxes during detection.

        Args:
            value (float, optional): default = :obj:`0.5`.
        """
        return self._detect_iou

    @detect_iou.setter
    def detect_iou(self, value):
        assert isinstance(value, float)
        self._detect_iou = value

    @property
    def score_threshold(self):
        r"""Score threshold used to filter boxes during detection.

        Args:
            value (float, optional): default = :obj:`0.5`.
        """
        return self._score_threshold

    @score_threshold.setter
    def score_threshold(self, value):
        assert isinstance(value, float)
        self._score_threshold = value

    @property
    def boxes_encode_var1(self):
        r"""Variance used to encode boxes - `SSD: Single Shot MultiBox
        Detector <https://arxiv.org/pdf/1512.02325.pdf>`_.

        Args:
            value (float, optional): default = :obj:`0.1`.
        """
        return self._boxes_encode_var1

    @boxes_encode_var1.setter
    def boxes_encode_var1(self, value):
        assert isinstance(value, float)
        self._boxes_encode_var1 = value

    @property
    def boxes_encode_var2(self):
        r"""Variance used to encode boxes - `SSD: Single Shot MultiBox
        Detector <https://arxiv.org/pdf/1512.02325.pdf>`_.

        Args:
            value (float, optional): default = :obj:`0.2`.
        """
        return self._boxes_encode_var2

    @boxes_encode_var2.setter
    def boxes_encode_var2(self, value):
        assert isinstance(value, float)
        self._boxes_encode_var2 = value

    @property
    def point_encode_var(self):
        r"""SSD normalization variance is used for points.

        Args:
            value (float, optional): default = :obj:`0.5`.
        """
        return self._point_encode_var

    @point_encode_var.setter
    def point_encode_var(self, value):
        assert isinstance(value, float)
        self._point_encode_var = value

    @property
    def is_pad(self):
        r"""Used for computing centers.

        Args:
            value (bool, optional): default = :obj:`True`.
        """
        return self._is_pad

    @is_pad.setter
    def is_pad(self, value):
        assert isinstance(value, bool)
        self._is_pad = value

    @property
    def ignore_base(self):
        r"""Gradients are not propagated to base network for ignore_base
        iterations. Used when a pretrained network is used to tune parameters.

        Args:
            value (int, optional): default = :obj:`0`.
        """
        return self._ignore_base

    @ignore_base.setter
    def ignore_base(self, value):
        assert isinstance(value, int)
        self._ignore_base = value

    @property
    def anchors_per_layer(self):
        r"""All anchors per layer. A list/tuple of list/tuple of
        config.an_anchor's.

        Args:
            value (int, optional): default = :obj:`0`.
        """
        return self._anchors_per_layer

    @anchors_per_layer.setter
    def anchors_per_layer(self, value):
        assert isinstance(value, (list, tuple))
        for x in value:
            assert isinstance(x, (list, tuple))
            for y in x:
                assert isinstance(y, self._an_anchor)
        self._anchors_per_layer = value

    def an_anchor(self, w: int, h: int, offset: int = 0):
        r"""A namedtuple with w and h of anchor."""
        return self._an_anchor(w, h, offset)

    def __repr__(self):
        msg = ["CONFIG :: {}".format(self.name)]
        msg += ["Base = {}".format(
            self.base_network if isinstance(self.base_network, str) else
            self.base_network.__name__)]
        msg += ["Body = {}".format(self.body_network)]
        msg += ["t_size = {}".format("x".join(map(str, self.t_size)))]
        msg += ["n_label = {}".format(self.n_label)]
        msg += ["LabelLoss = {}".format(self.label_loss_kwargs["method"])]
        if self.is_boxes:
            msg += ["BoxesLoss = {}".format(self.boxes_loss_kwargs["method"])]
        if self.is_point:
            msg += ["PointLoss = {}".format(self.point_loss_kwargs["method"])]
        msg += ["Objectness is " + (" ON" if self.is_objectness else "OFF") +
                " & Centerness is " + (" ON" if self.is_centerness else "OFF")]
        return "\n\t".join(msg)
