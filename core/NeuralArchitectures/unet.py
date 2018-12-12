import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from core.NeuralLayers import *

class BaseConv(nn.Module):
    '''
    https://arxiv.org/pdf/1505.04597.pdf --> base UNet {(convolution+ReLU+batch_norm)*2}
    chop: squeeze the number of channels by the ratio in the first convolution opertion
    '''
    def __init__(self, tensor_size, out_channels, pad=False, chop=2, *args, **kwargs):
        super(BaseConv, self).__init__()
        norm = "batch"
        self.baseconv = nn.Sequential()
        self.baseconv.add_module("cnv_1", Convolution(tensor_size, 3, out_channels//chop, pad=pad, normalization=norm))
        self.baseconv.add_module("cnv_2", Convolution(self.baseconv[-1].tensor_size, 3, out_channels, pad=pad, normalization=norm))
        self.tensor_size = self.baseconv[-1].tensor_size

    def forward(self, tensor):
        return self.baseconv(tensor)


class BaseResSE(nn.Module):
    '''
    https://arxiv.org/pdf/1808.05238.pdf --> AnatomyNet Base Block {(residual+squeeze_excitation+leakyReLU)*2}
    '''
    def __init__(self, tensor_size, out_channels, pad=True, *args, **kwargs):
        super(BaseResSE, self).__init__()
        norm, activation = "batch", "lklu"
        self.baseresSE = nn.Sequential()
        self.baseresSE.add_module("resSE_1", SEResidualComplex(tensor_size, 3, out_channels, pad=pad, activation=activation, normalization=norm, r=4))
        self.baseresSE.add_module("resSE_2", SEResidualComplex(self.baseresSE[-1].tensor_size, 3, out_channels, pad=pad, activation=activation, normalization=norm, r=4))
        self.tensor_size = self.baseresSE[-1].tensor_size

    def forward(self, tensor):
        return self.baseresSE(tensor)


class Down(nn.Module):
    '''
    change: uses convolution with strides instead of pooling to downsample
    '''
    def __init__(self, tensor_size, in_channels, out_channels, strides=(1,1), pad=False, dropout=0.0, nettype='anatomynet', *args, **kwargs):
        super(Down, self).__init__()
        assert nettype.lower() in ["unet", "anatomynet", "none"], "network sould be unet or anatomynet or none"
        self.down = nn.Sequential()
        self.down.add_module("down_conv_1", Convolution(tensor_size, 2, in_channels, (2,2), pad=False))
        if nettype is 'unet':
            self.down.add_module("down_conv_2", BaseConv(self.down[-1].tensor_size, out_channels, pad=pad))
        elif nettype is 'anatomynet':
            self.down.add_module("down_conv_2", BaseResSE(self.down[-1].tensor_size, out_channels, pad=pad))
        else:
            pass
        self.tensor_size = self.down[-1].tensor_size

    def forward(self, tensor):
        return self.down(tensor)


class Up(nn.Module):
    '''
    upsampling the feature maps
    '''
    def __init__(self, tensor_size, out_shape, strides=(1,1), pad=False, dropout=0.0, nettype='anatomynet', *args, **kwargs):
        super(Up, self).__init__()
        assert nettype.lower() in ["unet", "anatomynet", "none"], "network sould be unet or anatomynet or none"
        self.up = ConvolutionTranspose(tensor_size, 2, tensor_size[1]//2, 2, False)
        _ts = self.up.tensor_size
        _tensor_size = (_ts[0], _ts[1]*2)+out_shape[2:]
        if nettype is 'unet':
            self.up_base =  BaseConv(_tensor_size, out_shape[1], pad=pad)
            self.tensor_size = self.up_base.tensor_size
        elif nettype is 'anatomynet':
            self.up_base = BaseResSE(_tensor_size, out_shape[1], pad=pad)
            self.tensor_size = self.up_base.tensor_size
        else:
            self.tensor_size = _tensor_size
            pass

    def forward(self, tensor1,tensor2, nettype="anatomynet"):
        tensor1 = self.up(tensor1, tensor2.shape)
        if nettype is not "none":
            tensor = torch.cat([tensor2, tensor1], dim=1)
            return self.up_base(tensor)
        else:
            return tensor1

class UNet(nn.Module):
    '''
    UNet: https://arxiv.org/pdf/1505.04597.pdf
    4 down blocks and 4 up blocks
    '''
    def __init__(self, tensor_size, out_channels, n_classes = 10, *args, **kwargs):
        super(UNet, self).__init__()
        self.downnet1   = BaseConv(tensor_size, out_channels, pad=True)
        self.downnet2   = Down(self.downnet1.tensor_size, out_channels*1, out_channels*2, pad=True)

        self.downnet3   = Down(self.downnet2.tensor_size, out_channels*2, out_channels*4, pad=True)
        self.downnet4   = Down(self.downnet3.tensor_size, out_channels*4, out_channels*8, pad=True)
        self.downnet5   = Down(self.downnet4.tensor_size, out_channels*8, out_channels*16, pad=True)
        self.upnet1     = Up(self.downnet5.tensor_size, self.downnet4.tensor_size)
        self.upnet2     = Up(self.upnet1.tensor_size, self.downnet3.tensor_size)
        self.upnet3     = Up(self.upnet2.tensor_size, self.downnet2.tensor_size)
        self.upnet4     = Up(self.upnet3.tensor_size, self.downnet1.tensor_size)
        self.final_layer= Convolution(self.upnet4.tensor_size, 1, n_classes)
        self.tensor_size = self.final_layer.tensor_size

    def forward(self, tensor):
        d1 = self.downnet1(tensor)
        d2 = self.downnet2(d1)
        d3 = self.downnet3(d2)
        d4 = self.downnet4(d3)
        d5 = self.downnet5(d4)
        u1 = self.upnet1(d5, d4)
        u2 = self.upnet2(u1, d3)
        u3 = self.upnet3(u2, d2)
        u4 = self.upnet4(u3, d1)
        return self.final_layer(u4)

class UNetMini(nn.Module):
    '''
    AnatomyNet architecture -- https://arxiv.org/pdf/1808.05238.pdf
    '''
    def __init__(self, tensor_size, out_channels, n_classes = 2, *args, **kwargs):
        super(UNetMini, self).__init__()
        self.downnet1   = Down(tensor_size, tensor_size[1], out_channels, pad=True)
        self.basenet1   = BaseResSE(self.downnet1.tensor_size, int(out_channels*1.25))
        self.basenet2   = BaseResSE(self.basenet1.tensor_size, int(out_channels*1.50))
        self.basenet3   = BaseResSE(self.basenet2.tensor_size, int(out_channels*1.75))
        self.basenet4   = BaseResSE(self.basenet3.tensor_size, int(out_channels*1.75))
        _tensor_size    = self.basenet4.tensor_size
        _tensor_size    = (_tensor_size[0], self.basenet4.tensor_size[1]+self.basenet2.tensor_size[1], _tensor_size[2], _tensor_size[3])
        self.concat1    = BaseResSE(_tensor_size, int(out_channels*1.50))
        _tensor_size    = self.concat1.tensor_size
        _tensor_size    = (_tensor_size[0], self.concat1.tensor_size[1]+self.basenet1.tensor_size[1], _tensor_size[2], _tensor_size[3])
        self.concat2    = BaseResSE(_tensor_size, int(out_channels*1.25))
        _tensor_size    = self.concat2.tensor_size
        _tensor_size    = (_tensor_size[0], self.concat2.tensor_size[1]+self.downnet1.tensor_size[1], _tensor_size[2], _tensor_size[3])
        self.concat3    = BaseResSE(_tensor_size, int(out_channels))
        self.upnet      = Up(self.concat3.tensor_size, tensor_size, nettype="none")
        _tensor_size    = (_tensor_size[0], self.upnet.tensor_size[1]//2+tensor_size[1], _tensor_size[2], _tensor_size[3])
        self.final1     = Convolution(_tensor_size, 3, int(out_channels*0.5), pad=True, normalization="batch")
        self.final2     = Convolution(self.final1.tensor_size, 3, n_classes, pad=True, normalization="batch")

    def forward(self, tensor):
        d1 = self.downnet1(tensor)
        b1 = self.basenet1(d1)
        b2 = self.basenet2(b1)
        b3 = self.basenet3(b2)
        b4 = self.basenet4(b3)
        c1 = self.concat1(torch.cat([b4, b2],dim=1))
        c2 = self.concat2(torch.cat([c1, b1],dim=1))
        c3 = self.concat3(torch.cat([c2, d1],dim=1))
        up = self.upnet(c3, tensor, "none")
        f1 = self.final1(torch.cat([up,tensor],dim=1))
        f2 = self.final2(f1)
        return f2

# tsize = (1,1,572,572)
# unet = UNet(tsize, 64)
# unet_mini = UNetMini(tsize, 64)
# unet(torch.rand(tsize)).shape
# unet_mini(torch.rand(tsize)).shape
