""" TensorMONK's :: detection :: NoFPN & FPN layers

* Implementation may vary when compared to what is refered as the intension was
not to replicate but to have the flexibility to utilize concepts across several
papers.

# TODO: More options for Block
"""

__all__ = ["BiFPNLayer", "FPNLayer", "PAFPNLayer", "NoFPNLayer"]

import torch
import torch.nn as nn
from ..layers import FeatureFusion
from .config import CONFIG


class Block(nn.Module):
    r"""DepthWiseSeparable + FeatureFusion or FeatureFusion +
    DepthWiseSeparable.
    (`EfficientDet: Scalable and Efficient Object Detection
    <https://arxiv.org/pdf/1911.09070.pdf>`_)

    Args:
        encoding_depth (int, required): depth of all the input tensor's
        n_features (int, required): #Features to fuse. When n_features = 1,
            FeatureFusion is performed with input and the ouput of
            DepthWiseSeparable layer. Otherwise, FeatureFusion is performed on
            all the inputs, followed by DepthWiseSeparable layers.
        fusion (str, optional): fusion logic after resizing all the tensor's
            to match the first tensor in the list/tuple/args using bilinear
            interpolation. Options - :obj:`"sum"`,
            :obj:`"fast-normalize"`, :obj:`"softmax"`.
            (default = :obj:`"softmax"`)

    # TODO: More options for Block
    """

    def __init__(self,
                 encoding_depth: int,
                 n_features: int,
                 fusion: str = "softmax"):

        super(Block, self).__init__()
        assert n_features >= 1 and isinstance(n_features, int)

        self.n_features = n_features
        self.depthwise = nn.Sequential(
            nn.Conv2d(encoding_depth, encoding_depth, 3, 1, 1, bias=False,
                      groups=encoding_depth),
            nn.BatchNorm2d(encoding_depth, momentum=0.003, eps=0.5e-3))
        self.pointwise = nn.Sequential(
            nn.Conv2d(encoding_depth, encoding_depth, 1, bias=False),
            nn.BatchNorm2d(encoding_depth, momentum=0.003, eps=0.5e-3))
        self.fusion = FeatureFusion(n_features if n_features > 1 else 2,
                                    fusion)

    def forward(self, *args) -> torch.Tensor:
        assert len(args) == self.n_features
        tensor = args[0] if len(args) == 1 else self.fusion(*args)

        o = self.depthwise(tensor)
        o = o * o.sigmoid()
        o = self.pointwise(o)
        o = o * o.sigmoid()
        return self.fusion(tensor, o) if len(args) == 1 else o


class NoFPNLayer(nn.Module):
    r"""Residual DepthWiseSeparable is used as base block.

    Args:
        config (:class:`~tensormonk.detection.CONFIG`, required): See
            :class:`~tensormonk.detection.CONFIG` for more details.

    .. code-block:: none

        n_scales = 3
        ------------
        Ex: Base with single FPN layer

        Pretrained | Detection Layers
        Ex: ResNet | with anchors
        -----------|-----------------
            o      |   -> o
            ^      |
            |      |
            o      |   -> o
            ^      |
            |      |
            o      |   -> o
            ^      |
            |      |
            o      |
            ^      |
            |      |
                   |
          input    |
    """

    def __init__(self, config: CONFIG):
        super(NoFPNLayer, self).__init__()

        self.n_scales = len(config.anchors_per_layer)
        self.encoding_depth = config.encoding_depth
        self.context = nn.ModuleList([Block(config.encoding_depth, 1, "sum")
                                      for _ in range(self.n_scales)])

    def forward(self, *args) -> tuple:
        assert any(isinstance(o, torch.Tensor) for o in args)
        assert any(o.size(1) == self.encoding_depth for o in args)
        assert len(args) == self.n_scales
        return [cnn(o) for cnn, o in zip(self.context, args)]


class FPNLayer(nn.Module):
    r"""A modified version of FPN compatible with
    :class:`~tensormonk.detection.CONFIG`.
    Upscale/downscale is done with bilinear interpolation.
    (`Feature Pyramid Networks for Object Detection
    <https://arxiv.org/pdf/1612.03144.pdf>`_).

    Args:
        config (:class:`~tensormonk.detection.CONFIG`, required): See
            :class:`~tensormonk.detection.CONFIG` for more details.

    .. code-block:: none

        n_scales = 3           Ex: Base with single FPN layer
        ------------           ------------------------------
            -> o ->            o -> o -> low-resolution
               |               ^    |
               v               |    v
            -> o ->            o -> o ->
               |               ^    |
               v               |    v
            -> o ->            o -> o -> high-resolution
                               ^
                               |
                               o
                               ^
                               |
                             input
    """

    def __init__(self, config: CONFIG):
        super(FPNLayer, self).__init__()

        self.n_scales = len(config.anchors_per_layer)
        self.encoding_depth = config.encoding_depth
        self.fusion = config.body_fpn_fusion
        assert len(config.anchors_per_layer) > 1, \
            "FPNLayer: Must have more than 1 prediction layers to use FPN's"

        self.down_2_up = nn.ModuleList(
            [Block(self.encoding_depth, 1 if i == 0 else 2,
                   fusion="sum" if i == 0 else self.fusion)
             for i in range(self.n_scales)])

    def forward(self, *args) -> tuple:
        assert any(isinstance(o, torch.Tensor) for o in args)
        assert any(o.size(1) == self.encoding_depth for o in args)
        assert len(args) == self.n_scales

        # args are higher to lower resolution --> so flipped
        args = args[::-1]

        responses = []
        for i, cnn in zip(range(self.n_scales), self.down_2_up):
            if i == 0:
                # Residual DepthWiseSeparable
                responses.append(cnn(args[i]))
                continue
            # Weighted DepthWiseSeparable
            responses.append(cnn(args[i], responses[-1]))
        return responses[::-1]  # flip to output higher to lower resolution


class PAFPNLayer(nn.Module):
    r"""A modified version of PAFPN compatible with
    :class:`~tensormonk.detection.CONFIG`. Upscale/downscale is done with
    bilinear interpolation.
    (`Path aggregation network for instance segmentation
    <https://arxiv.org/pdf/1803.01534.pdf>`_).

    Args:
        config (:class:`~tensormonk.detection.CONFIG`, required): See
            :class:`~tensormonk.detection.CONFIG` for more details.

    .. code-block:: none

        Logic:  n_scales = 3
        --------------------
        low-resolution    -> o -> o ->
                             |    ^
                             v    |
                          -> o -> o ->
                             |    ^
                             v    |
        high-resolution   -> o -> o ->
    """

    def __init__(self, config: CONFIG):
        super(PAFPNLayer, self).__init__()

        self.n_scales = len(config.anchors_per_layer)
        self.encoding_depth = config.encoding_depth
        self.fusion = config.body_fpn_fusion
        assert len(config.anchors_per_layer) > 1, \
            "PAFPNLayer: Must have more than 1 prediction layers to use FPN's"

        self.down_2_up = nn.ModuleList(
            [Block(self.encoding_depth, 1 if i == 0 else 2,
                   fusion="sum" if i == 0 else self.fusion)
             for i in range(self.n_scales)])

        self.up_2_down = nn.ModuleList(
            [Block(self.encoding_depth, 1 if i == 0 else 2,
                   fusion="sum" if i == 0 else self.fusion)
             for i in range(self.n_scales)])

    def forward(self, *args) -> tuple:
        assert any(isinstance(o, torch.Tensor) for o in args)
        assert any(o.size(1) == self.encoding_depth for o in args)
        assert len(args) == self.n_scales

        # args are higher to lower resolution --> so flipped
        args = args[::-1]
        # down to up
        intermediate = []
        for i, cnn in zip(range(self.n_scales), self.down_2_up):
            if i == 0:
                # Residual DepthWiseSeparable
                intermediate.append(cnn(args[i]))
                continue
            # Weighted DepthWiseSeparable
            intermediate.append(cnn(args[i], intermediate[-1]))

        # flip for higher to lower resolution
        intermediate = intermediate[::-1]

        # up to down
        responses = []
        for i, cnn in zip(range(self.n_scales), self.up_2_down):
            if i == 0:
                # Residual DepthWiseSeparable
                responses.append(cnn(intermediate[i]))
                continue
            # Weighted DepthWiseSeparable
            responses.append(cnn(intermediate[i], responses[-1]))
        return responses


class BiFPNLayer(nn.Module):
    r"""A modified version of BiFPNLayer compatible with
    :class:`~tensormonk.detection.CONFIG`. Upscale/downscale is done with
    bilinear interpolation. (`EfficientDet: Scalable and Efficient Object
    Detection <https://arxiv.org/pdf/1911.09070.pdf>`_).

    Args:
        config (:class:`~tensormonk.detection.CONFIG`, required): See
            :class:`~tensormonk.detection.CONFIG` for more details.

    .. code-block:: none

        Logic: n_scales = 4
        -------------------
        low-resolution    o ------> o ->
                           _\_____  ^
                          |  \    \ |
                          o -> o -> o ->
                           ___ | _  ^
                          |    v  \ |
                          o -> o -> o ->
                                 \  ^
                                  \ |
        high-resolution   o ------> o ->
    """

    def __init__(self, config: CONFIG):
        super(BiFPNLayer, self).__init__()

        self.n_scales = len(config.anchors_per_layer)
        self.encoding_depth = config.encoding_depth
        self.fusion = config.body_fpn_fusion
        assert len(config.anchors_per_layer) > 2, \
            "BiFPNLayer: Must have more than 2 prediction layers to use FPN's"

        self.down_2_up = nn.ModuleList(
            [Block(self.encoding_depth, 2, fusion=self.fusion)
             for _ in range(self.n_scales - 2)])

        self.up_2_down = nn.ModuleList(
            [Block(self.encoding_depth, 3 if i % (self.n_scales - 1) else 2,
             fusion=self.fusion) for i in range(self.n_scales)])

    def forward(self, *args) -> tuple:
        assert any(isinstance(o, torch.Tensor) for o in args)
        assert any(o.size(1) == self.encoding_depth for o in args)
        assert len(args) == self.n_scales

        # args are higher to lower resolution --> so flipped
        args = args[::-1]
        # down to up
        intermediate = []
        for i, cnn in zip(range(1, self.n_scales-1), self.down_2_up):
            if i == 1:
                intermediate.append(cnn(args[i], args[i-1]))
                continue
            intermediate.append(cnn(args[i], intermediate[-1]))

        # flip for higher to lower resolution
        intermediate = intermediate[::-1]
        args = args[::-1]

        # up to down
        responses = []
        for i, cnn in zip(range(self.n_scales), self.up_2_down):
            if i == 0:
                responses.append(cnn(args[i], intermediate[i]))
                continue
            if i + 1 == self.n_scales:
                responses.append(cnn(args[i], responses[-1]))
                continue
            responses.append(cnn(args[i], intermediate[i-1], responses[-1]))
        return responses
