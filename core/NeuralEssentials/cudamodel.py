""" TensorMONK's :: NeuralEssentials                                         """

import torch
import torch.nn as nn
import visdom
#==============================================================================#


class CudaModel(nn.Module):
    """ Works on both CPU & GPU """
    def __init__(self, is_cuda, gpus, net, net_kwargs):
        super(CudaModel, self).__init__()

        self.gpus = gpus
        self.is_cuda = is_cuda
        self.NET46 = net( **net_kwargs )
        self.tensor_size = self.NET46.tensor_size

    def forward(self, inputs):
        if type(inputs) in [list,tuple]:
            if self.is_cuda:
                inputs = [x.cuda() if hasattr(x, "is_cuda") else x
                          for x in inputs]
            return self.NET46(*inputs)
            if self.is_cuda:
                inputs = [x.cuda() for x in inputs]
            return self.NET46(*inputs)
        else:
            if self.is_cuda:
                inputs = inputs.cuda()
            if self.is_cuda and self.gpus>1:
                return nn.parallel.data_parallel(self.NET46, inputs, range(self.gpus))
            else:
                return self.NET46(inputs)

    def regularize_weights(self, clip=0., only_convs=False):
        if self.training:
            self.clip_weights(clip)
            for p in self.NET46.parameters():
                if p.data.ndimension() == 4:
                    # convolution
                    if p.data.size(2)*p.data.size(3) > 1:
                        # ignore 1x1's
                        l2 = p.data.pow(2).sum(3).sum(2).pow(.5).add(1e-8)
                        p.data.div_(l2.unsqueeze(2).unsqueeze(3)).div_(p.size(1)**0.5)
                    else:
                        p.data.div_(p.data.abs().max(1, True)[0]).div_(p.size(1)**0.5)
                elif p.data.ndimension() == 3 and not only_convs:
                    # routing capsule
                    # pass
                    l2 = p.data.pow(2).sum(2).sum(1).pow(.5).add(1e-8)
                    p.data.div_(l2.unsqueeze(1).unsqueeze(2))
                elif p.data.ndimension() == 2 and not only_convs:
                    # fully-connected and lossfunctions
                    l2 = p.data.pow(2).sum(1).pow(.5).add(1e-8)
                    p.data.div_(l2.unsqueeze(1))
                else:
                    # bias, gamma, beta are excluded
                    pass

    def clip_weights(self, clip):
        if self.training:
            if not isinstance(clip, float):
                clip = 0.
            if clip > 0.:
                for p in self.NET46.parameters():
                    if p.data.ndimension() in [2, 3, 4]:
                        p.data.clamp_(-clip, clip)

    def show_weights(self, visplots=None):
        for p in self.NET46.state_dict().keys():
            if isinstance(visplots, visdom.Visdom) and "weight" in p and \
               "weight_g" not in p and "Normalization" not in p and \
               "bias" not in p:

                # ignore normalization weights (gamma's & beta's) and bias
                newid = p.replace("NET46.", "").replace("network.", "")
                ws = self.NET46.state_dict()[p].data.cpu()
                sz = ws.size()
                if ws.ndimension() == 4:
                    if sz[2] > 3 and sz[3] > 3:
                        if not (sz[1] == 1 or sz[1] == 3):
                            ws = ws.view(-1, 1, sz[2], sz[3])
                            sz = ws.size()
                        if sz[0] <= 2*10:
                            min_ws = ws.min(2, True)[0].min(3, True)[0]
                            max_ws = ws.max(2, True)[0].max(3, True)[0]
                            visplots.images((ws-min_ws) / (max_ws-min_ws),
                                opts={"title": "Ws-"+newid}, win="Ws-"+newid)
                ws = ws.view(-1)
                visplots.histogram(X=ws,
                    opts={"numbins": 20, "title":newid}, win=newid)
