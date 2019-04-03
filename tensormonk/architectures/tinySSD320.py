""" TensorMONK :: architectures """

import torch
import torch.nn as nn
from ..layers import Convolution as CONV
# =========================================================================== #


class Tiny(nn.Module):
    def __init__(self, tensor_size, filter_size, out_channels, strides=1,
                 groups=4):
        super(Tiny, self).__init__()
        assert (tensor_size[1] / 2. / groups) == int(tensor_size[1]/2./groups)

        t_size = (None, tensor_size[1]//2, tensor_size[2], tensor_size[3])
        self.Block1 = CONV(t_size, filter_size, tensor_size[1]//2, strides,
                           True, "relu", 0., "batch", groups=groups)
        self.Block2 = CONV(self.Block1.tensor_size, 1, out_channels, 1,
                           True, None)
        self.tensor_size = self.Block2.tensor_size

    def forward(self, tensor):
        maxo_tensor = torch.max(*tensor.split(tensor.size(1)//2, 1))
        o = self.Block2(self.Block1(maxo_tensor))
        if o.size(1) == tensor.size(1) and o.size(2) == tensor.size(2):
            return tensor + o
        return o


class TinySSD320(nn.Module):
    r""" A Tiny SSD320 based architecture! """
    def __init__(self, tensor_size: list, n_labels: int,
                 boxes_per_layer: list,
                 translator: nn.Module, **kwargs):
        super(TinySSD320, self).__init__()

        self._tensor_size = tensor_size
        self.translator = translator
        self.n_labels = n_labels

        # transforms for detection (contains priors)
        self.translator = translator

        # Spatial features
        self.to40 = nn.Sequential(
            CONV(tensor_size, 3, 48, 2, True, "relu", 0., "batch"),
            Tiny((1, 48, 160, 160), 3, 48, 1, 4),
            Tiny((1, 48, 160, 160), 3, 64, 2, 2),
            Tiny((1, 64,  80,  80), 3, 64, 1, 4),
            Tiny((1, 64,  80,  80), 3, 64, 2, 2),
            Tiny((1, 64,  40,  40), 3, 64, 1, 4),
            Tiny((1, 64,  40,  40), 3, 64, 1, 2),
            Tiny((1, 64,  40,  40), 3, 64, 1, 4))
        print("to40", self.to40[-1].tensor_size)

        self.to20 = nn.Sequential(
            Tiny((1,  64, 40, 40), 3, 128, 2, 4),
            Tiny((1, 128, 20, 20), 3, 128, 1, 4),
            Tiny((1, 128, 20, 20), 3, 128, 1, 4),
            Tiny((1, 128, 20, 20), 3, 128, 1, 4))
        print("to20", self.to20[-1].tensor_size)

        self.to10 = nn.Sequential(
            Tiny((1, 128, 20, 20), 3, 256, 2, 4),
            Tiny((1, 256, 10, 10), 3, 256, 1, 4),
            Tiny((1, 256, 10, 10), 3, 256, 1, 4),
            Tiny((1, 256, 10, 10), 3, 256, 1, 4),
            Tiny((1, 256, 10, 10), 3, 256, 1, 4))
        print("to10", self.to10[-1].tensor_size)

        self.to05 = Tiny((1, 256, 10, 10), 3, 128, 2, 2)
        print("to05", self.to05.tensor_size)

        self.to03 = Tiny((1, 128,  5,  5), 3, 128, 2, 2)
        print("to03", self.to03.tensor_size)

        self.to01 = nn.Sequential(
            CONV((1, 128, 3, 3), 3, 128, 1, False, "relu"),
            CONV((1, 128, 1, 1), 1, 128, 1, False, None))
        print("to01", self.to01[-1].tensor_size)

        # boxes & predictions
        nb = [x*4 for x in boxes_per_layer]
        nl = [x*n_labels for x in boxes_per_layer]
        self.box40 = CONV(self.to40[-1].tensor_size, 1, nb[0], 1, True, None)
        self.pre40 = CONV(self.to40[-1].tensor_size, 1, nl[0], 1, True, None)
        print("40's --", self.box40.tensor_size, "&", self.pre40.tensor_size)
        self.box20 = CONV(self.to20[-1].tensor_size, 1, nb[1], 1, True, None)
        self.pre20 = CONV(self.to20[-1].tensor_size, 1, nl[1], 1, True, None)
        print("20's --", self.box20.tensor_size, "&", self.pre20.tensor_size)
        self.box10 = CONV(self.to10[-1].tensor_size, 1, nb[2], 1, True, None)
        self.pre10 = CONV(self.to10[-1].tensor_size, 1, nl[2], 1, True, None)
        print("10's --", self.box10.tensor_size, "&", self.pre10.tensor_size)
        self.box05 = CONV(self.to05.tensor_size, 1, nb[3], 1, True, None)
        self.pre05 = CONV(self.to05.tensor_size, 1, nl[3], 1, True, None)
        print("05's --", self.box05.tensor_size, "&", self.pre05.tensor_size)
        self.box03 = CONV(self.to03.tensor_size, 1, nb[4], 1, True, None)
        self.pre03 = CONV(self.to03.tensor_size, 1, nl[4], 1, True, None)
        print("03's --", self.box03.tensor_size, "&", self.pre03.tensor_size)
        self.box01 = CONV(self.to01[-1].tensor_size, 1, nb[5], 1, True, None)
        self.pre01 = CONV(self.to01[-1].tensor_size, 1, nl[5], 1, True, None)
        print("01's --", self.box01.tensor_size, "&", self.pre01.tensor_size)
        self.tensor_size, self.t_size = (None, 9590, n_labels), (None, 9590, 4)

        self.mean = torch.Tensor([0.4078, 0.4588, 0.4824]).view(1, 3, 1, 1)

    def feat_boxs_pres(self, tensor: torch.Tensor, who: str = "40"):
        # spatial features -- ex: to40 320x320 to 40x40
        toXX = getattr(self, "to" + who)(tensor)
        # Ex: 4 boxes per pixel on a tensor of size _x_x40x40 -> 6400x4 boxes
        boxXX = getattr(self, "box" + who)(toXX).permute(0, 2, 3, 1)
        boxXX = boxXX.contiguous().view(tensor.size(0), -1, 4)
        # Ex: boxes_per_layer*n_labels per pixel on a tensor of size
        #   _xboxes_per_layer*n_labelsx40x40 -> 6400xn_labels scores
        preXX = getattr(self, "pre" + who)(toXX).permute(0, 2, 3, 1)
        preXX = preXX.contiguous().view(tensor.size(0), -1, self.n_labels)
        return toXX, boxXX, preXX

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        t40, box40, pre40 = self.feat_boxs_pres(tensor, "40")
        t20, box20, pre20 = self.feat_boxs_pres(t40,    "20")
        t10, box10, pre10 = self.feat_boxs_pres(t20,    "10")
        t05, box05, pre05 = self.feat_boxs_pres(t10,    "05")
        t03, box03, pre03 = self.feat_boxs_pres(t05,    "03")
        t01, box01, pre01 = self.feat_boxs_pres(t03,    "01")

        # (6400+2400+600+150+36+4) = 9590 boxes and predictions
        gcxcywh_boxes = torch.cat((box40, box20, box10,
                                   box05, box03, box01), 1)
        predictions = torch.cat((pre40, pre20, pre10, pre05, pre03, pre01), 1)
        return gcxcywh_boxes, predictions

    def detect(self, tensor: torch.Tensor):
        r""" Detect all the objects in the images after score > threshold and
        non-maximal suppression """

        if self.translator is None:
            return

        with torch.no_grad():
            gcxcywh_boxes, predictions = self(tensor)

        detected_objects, their_labels, their_scores = \
            self.translator(gcxcywh_boxes, None, predictions)
        return detected_objects, their_labels, their_scores


# from tensormonk.layers import Convolution as CONV
# from tensormonk.utils import SSDUtils
# # the configuration for SSD 320 - works for MobileNetV2SSD320 & TinySSD320
# ratios1, ratios2 = (1, 2, 1/2), (1, 2, 3, 1/2, 1/3)
# layer_infos = [
#     SSDUtils.LayerInfo((None, None, 40, 40), ratios1, .10, 0.20),
#     SSDUtils.LayerInfo((None, None, 20, 20), ratios2, .20, 0.37),
#     SSDUtils.LayerInfo((None, None, 10, 10), ratios2, .37, 0.54),
#     SSDUtils.LayerInfo((None, None,  5,  5), ratios2, .54, 0.71),
#     SSDUtils.LayerInfo((None, None,  3,  3), ratios1, .71, 0.88),
#     SSDUtils.LayerInfo((None, None,  1,  1), ratios1, .88, 1.05)]
#
# CONFIG_SSD320 = {"model": "SSD320",
#                  "tensor_size": (1, 3, 320, 320),
#                  "n_labels": 21,
#                  "layer_infos": layer_infos,
#                  "boxes_per_layer": [4, 6, 6, 6, 4, 4],
#                  "gcxcywh_var1": 0.1,
#                  "gcxcywh_var2": 0.2,
#                  "encode_iou_threshold": 0.5,
#                  "detect_iou_threshold": 0.2,
#                  "detect_score_threshold": 0.2,
#                  "detect_top_n": 1,
#                  "detect_n_objects": 50}
# tensor = torch.rand(*CONFIG_SSD320["tensor_size"])
# translator = SSDUtils.Translator(**CONFIG_SSD320)
# test = TinySSD320(**{"translator": translator, **CONFIG_SSD320})
# test(tensor)[0].size()
# test(tensor)[1].size()
# %timeit test(tensor)[1].size()
# %timeit test.detect(tensor)
