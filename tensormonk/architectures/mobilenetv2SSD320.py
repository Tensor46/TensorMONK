""" TensorMONK :: architectures """

import torch
import torch.nn as nn
from ..layers import Convolution as CONV
from ..layers import ResidualInverted as ResI
from ..layers import SeparableConvolution as SepC
# =========================================================================== #


class MobileNetV2SSD320(nn.Module):
    r""" SSD320 based on MobileNetV2 architecture! """
    def __init__(self, tensor_size: list, n_labels: int,
                 boxes_per_layer: list,
                 translator: nn.Module, **kwargs):
        super(MobileNetV2SSD320, self).__init__()

        self._tensor_size = tensor_size
        self.translator = translator
        self.n_labels = n_labels

        # transforms for detection (contains priors)
        self.translator = translator

        # Spatial features
        kwgs = {"activation": "relu6", "normalization": "batch"}
        self.to40 = nn.Sequential(
            CONV(tensor_size, 3, 16, 2, True, "relu", 0., "batch"),
            ResI((1, 16, 160, 160), 3, 16, 1, t=1, **kwgs),
            ResI((1, 16, 160, 160), 3, 24, 2, t=6, **kwgs),
            ResI((1, 24,  80,  80), 3, 24, 1, t=6, **kwgs),
            ResI((1, 24,  80,  80), 3, 32, 2, t=6, **kwgs),
            ResI((1, 32,  40,  40), 3, 32, 1, t=6, **kwgs),
            ResI((1, 32,  40,  40), 3, 32, 1, t=6, **kwgs))
        print("to40", self.to40[-1].tensor_size)

        self.to20 = nn.Sequential(
            ResI((1, 32, 40, 40), 3, 64, 2, t=6, **kwgs),
            ResI((1, 64, 20, 20), 3, 64, 1, t=6, **kwgs),
            ResI((1, 64, 20, 20), 3, 64, 1, t=6, **kwgs),
            ResI((1, 64, 20, 20), 3, 64, 1, t=6, **kwgs),
            ResI((1, 64, 20, 20), 3, 96, 1, t=6, **kwgs),
            ResI((1, 96, 20, 20), 3, 96, 1, t=6, **kwgs),
            ResI((1, 96, 20, 20), 3, 96, 1, t=6, **kwgs))
        print("to20", self.to20[-1].tensor_size)

        self.to10 = nn.Sequential(
            ResI((1,  96, 20, 20), 3, 160, 2, t=6, **kwgs),
            ResI((1, 160, 10, 10), 3, 160, 1, t=6, **kwgs),
            ResI((1, 160, 10, 10), 3, 160, 1, t=6, **kwgs),
            ResI((1, 160, 10, 10), 3, 320, 1, t=6, **kwgs),
            CONV((1, 320, 10, 10), 1, 1280, 1, True, "relu6", 0., "batch"))
        print("to10", self.to10[-1].tensor_size)

        self.to05 = ResI((1, 1280, 10, 10), 3, 512, 2, t=0.2, **kwgs)
        print("to05", self.to05.tensor_size)

        self.to03 = ResI((1, 512, 5, 5), 3, 256, 2, t=0.25, **kwgs)
        print("to03", self.to03.tensor_size)

        self.to01 = nn.Sequential(
            CONV((1, 256, 3, 3), 3, 256, 1, False, "relu6"),
            CONV((1, 256, 1, 1), 1,  64, 1, False, None))
        print("to01", self.to01[-1].tensor_size)

        # boxes & predictions
        nb = [x*4 for x in boxes_per_layer]
        nl = [x*n_labels for x in boxes_per_layer]
        self.box40 = SepC(self.to40[-1].tensor_size, 3, nb[0], 1, True, None)
        self.pre40 = SepC(self.to40[-1].tensor_size, 3, nl[0], 1, True, None)
        print("40's --", self.box40.tensor_size, "&", self.pre40.tensor_size)
        self.box20 = SepC(self.to20[-1].tensor_size, 3, nb[1], 1, True, None)
        self.pre20 = SepC(self.to20[-1].tensor_size, 3, nl[1], 1, True, None)
        print("20's --", self.box20.tensor_size, "&", self.pre20.tensor_size)
        self.box10 = SepC(self.to10[-1].tensor_size, 3, nb[2], 1, True, None)
        self.pre10 = SepC(self.to10[-1].tensor_size, 3, nl[2], 1, True, None)
        print("10's --", self.box10.tensor_size, "&", self.pre10.tensor_size)
        self.box05 = SepC(self.to05.tensor_size, 3, nb[3], 1, True, None)
        self.pre05 = SepC(self.to05.tensor_size, 3, nl[3], 1, True, None)
        print("05's --", self.box05.tensor_size, "&", self.pre05.tensor_size)
        self.box03 = SepC(self.to03.tensor_size, 3, nb[4], 1, True, None)
        self.pre03 = SepC(self.to03.tensor_size, 3, nl[4], 1, True, None)
        print("03's --", self.box03.tensor_size, "&", self.pre03.tensor_size)
        self.box01 = SepC(self.to01[-1].tensor_size, 1, nb[5], 1, True, None)
        self.pre01 = SepC(self.to01[-1].tensor_size, 1, nl[5], 1, True, None)
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
        tensor = tensor - self.mean

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
# from tensormonk.layers import ResidualInverted as ResI
# from tensormonk.layers import SeparableConvolution as SepC
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
# test = MobileNetV2SSD320(**{"translator": translator, **CONFIG_SSD320})
# test(tensor)[0].size()
# test(tensor)[1].size()
# %timeit test(tensor)[1].size()
# %timeit test.detect(tensor)
