""" TensorMONK's :: NeuralArchitectures                                     """
import torch.nn as nn
from ..NeuralLayers import Convolution
from ..NeuralLayers import ContextNet_Bottleneck


class ContextNet(nn.Module):
    r"""ContextNet: Exploring Context and Detail for Semantic Seg in Real-time
        Implemented  from https://arxiv.org/pdf/1805.04554.pdf

        Naming Convention for layers in the module:
        cnv::Convolution:: regular convolution block
        bn::bottleneck:: bottlenect residual block
        dw::Convilution:: depth-wise seperable convolution block
        dn:: deep netwrk for Context
        sn:: shallow network for Context
        Args:
            tensor_size: shape of tensor in BCHW
    """
    def __init__(self, tensor_size=(1, 3, 1024, 2048), *args, **kwargs):
        super(ContextNet, self).__init__()
        normalization, strides = "batch", [2, 1, 1]
        bottleneck = ContextNet_Bottleneck

        self.DeepNET = nn.Sequential()
        self.ShallowNET = nn.Sequential()

        # 1, 1, 256, 512
        self.DeepNET.add_module("avgpl", nn.AvgPool2d((5, 5), (4, 4), 2))
        # 1, 1, 128, 256
        self.DeepNET.add_module("dn_cnv1", Convolution(tensor_size, 3, 32, 2,
                                True, "relu", 0., normalization, False, 1))
        self.DeepNET.add_module("dn_bn10",
                                bottleneck(self.DeepNET[-1].tensor_size, 3, 32,
                                           1, expansion=1))
        self.DeepNET.add_module("dn_bn20",
                                bottleneck(self.DeepNET[-1].tensor_size, 3, 32,
                                           1, expansion=6))  # 1, 1, 128, 256
        for i in range(3):
            self.DeepNET.add_module("dn_bn3"+str(i),
                                    bottleneck(self.DeepNET[-1].tensor_size, 3,
                                               48, strides[i], expansion=6))
        # 1, 1, 64, 128
        for i in range(3):
            self.DeepNET.add_module("dn_bn4"+str(i),
                                    bottleneck(self.DeepNET[-1].tensor_size, 3,
                                               64, strides[i], expansion=6))
        # 1, 1, 32, 64
        for i in range(2):
            self.DeepNET.add_module("dn_bn5"+str(i),
                                    bottleneck(self.DeepNET[-1].tensor_size, 3,
                                               96, 1, expansion=6))
        for i in range(2):
            self.DeepNET.add_module("DN_BN6"+str(i),
                                    bottleneck(self.DeepNET[-1].tensor_size, 3,
                                               128, 1, expansion=6))
        # 1, 1, 32, 64
        self.DeepNET.add_module("dn_cnv2",
                                Convolution(self.DeepNET[-1].tensor_size, 3,
                                            128, 1, True, "relu", 0.,
                                            normalization, False, 1))
        # 1, 1, 32, 64
        self.DeepNET.add_module("upsample", nn.Upsample(scale_factor=4,
                                                        mode='bilinear'))
        # 1, 1, 128, 256
        _tensor_size = (1, 128, self.DeepNET[-2].tensor_size[2]*4,
                        self.DeepNET[-2].tensor_size[3]*4)
        self.DeepNET.add_module("dn_dw11",
                                Convolution(_tensor_size, 3, _tensor_size[1],
                                            1, True, "relu", 0., None, False,
                                            groups=_tensor_size[1],
                                            dilation=4))
        self.DeepNET.add_module("dn_dw12",
                                Convolution(self.DeepNET[-1].tensor_size, 1,
                                            128, 1, True, "relu", 0.,
                                            normalization, False, 1))
        self.DeepNET.add_module("dn_cnv3",
                                Convolution(self.DeepNET[-1].tensor_size, 1,
                                            128, 1, True, "relu", 0.,
                                            normalization, False, 1))
        # 128, 256
        activation, pre_nm, groups = "relu", False, 1
        self.ShallowNET.add_module("sm_cnv1",
                                   Convolution(tensor_size, 3, 32, 2, True,
                                               "relu", 0., True, False, 1))
        # 512 x 1024
        self.ShallowNET.add_module("sm_dw11",
                                   Convolution(self.ShallowNET[-1].tensor_size,
                                               3, 32, 2, True, activation, 0.,
                                               None, pre_nm,
                                               groups=tensor_size[1]))
        # 256, 512
        self.ShallowNET.add_module("sm_dw12",
                                   Convolution(self.ShallowNET[-1].tensor_size,
                                               1, 64, 1, True, activation, 0.,
                                               normalization, pre_nm, groups))
        self.ShallowNET.add_module("sm_dw21",
                                   Convolution(self.ShallowNET[-1].tensor_size,
                                               3, 64, 2, True, activation, 0.,
                                               None, pre_nm,
                                               groups=tensor_size[1]))
        self.ShallowNET.add_module("sm_dw22",
                                   Convolution(self.ShallowNET[-1].tensor_size,
                                               1, 128, 1, True, activation, 0.,
                                               normalization, pre_nm, groups))
        self.ShallowNET.add_module("sm_dw31",
                                   Convolution(self.ShallowNET[-1].tensor_size,
                                               3, 128, 1, True, activation, 0.,
                                               None, pre_nm,
                                               groups=tensor_size[1]))
        self.ShallowNET.add_module("sm_dw32",
                                   Convolution(self.ShallowNET[-1].tensor_size,
                                               1, 128, 1, True, activation, 0.,
                                               normalization, pre_nm, groups))
        self.ShallowNET.add_module("sm_cnv2",
                                   Convolution(self.ShallowNET[-1].tensor_size,
                                               1, 128, 1, True, activation, 0.,
                                               normalization, pre_nm, groups))
        # 128, 256
        self.tensor_size = self.ShallowNET[-1].tensor_size

        self.FuseNET = Convolution(self.ShallowNET[-1].tensor_size, 1,
                                   self.ShallowNET[-1].tensor_size[1], 1, True)

    def forward(self, tensor):
        return self.FuseNET(self.DeepNET(tensor)+self.ShallowNET(tensor))

# from core.NeuralLayers import *
# tensor_size = (1, 1, 1024, 2048)
# test = ContextNet(tensor_size)
