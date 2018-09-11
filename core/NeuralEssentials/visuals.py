""" TensorMONK's :: NeuralEssentials                                         """

import sys
import types
import visdom
if sys.version_info.major == 3:
    from functools import reduce
import torch
import torch.nn.functional as F
import torchvision.utils as tutils
import imageio
# ============================================================================ #


def MakeGIF(image_list, gif_name):
    if not gif_name.endswith(".gif"):
        gif_name += ".gif"
    imageio.mimsave(gif_name, [imageio.imread(x) for x in image_list])
# ============================================================================ #


class VisPlots(object):
    def __init__(self, env="main", server=None):

        if server is None:
            self.visplots = visdom.Visdom(env=env)
        else:
            self.visplots = visdom.Visdom(env=env, server=server)

    def histograms(self, data, name="hist"):
        if isinstance(data, dict):
            # parameter generator (essentially, model.state_dict())
            for p in data.keys():
                if "weight" in p and "weight_g" not in p and \
                   "Normalization" not in p and "bias" not in p:

                    # ignore normalization weights (gamma's & beta's) and bias
                    newid = p.replace("NET46.", "").replace("Net46.", "").replace("network.", "")
                    self.visplots.histogram(X=data[p].data.cpu().view(-1),
                        opts={"numbins": 46, "title":newid}, win=newid)
        elif isinstance(data, torch.nn.parameter.Parameter) or \
             isinstance(data, torch.Tensor):
            # pytorch tensor or parameter
            self.visplots.histogram(X=data.data.cpu().view(-1),
                opts={"numbins": 46, "title":name}, win=name)
        else:
            raise NotImplementedError

    def show_images(self, data, vis_name="images", png_name=None,
               normalize=False, height=None, max_samples=512):

        if isinstance(data, torch.Tensor):
            if data.dim() != 4:
                return None
            # pytorch tensor
            data = data.data.cpu()
            # adjust range to 0-1
            if normalize:
                _min = data.min(2, True)[0].min(3, True)[0]
                _max = data.max(2, True)[0].max(3, True)[0]
                data.add_(-_min).div(_max-_min)
            # adjust 4d tensor and reduce samples when too many
            sz = data.size()
            multiplier = 1
            if sz[1] not in [1, 3]:
                data = data.view(-1, 1, *sz[2:])
                multiplier = sz[1]
            if sz[0]*multiplier > max_samples:
                samples = reduce(lambda x, y: max(x, y), [x*multiplier for x \
                    in range(sz[0]) if x*multiplier <= max_samples])
                data = data[:samples]
            # resize image when height is not None
            if height is not None:
                data = F.interpolate(data, size=(height, int(float(height)*sz[2]/sz[3])))

            self.visplots.images(data, nrow=max(4, int(data.size(0)**0.5)),
                opts={"title": vis_name}, win=vis_name)
            # save a png if png_name is defined
            if png_name is not None:
                tutils.save_image(data, png_name)

    def show_weights(self, data, vis_name="weights", png_name=None, min_width=3):
        # all histograms
        self.histograms(data, vis_name)
        # only convolution weights when kernel size > 3
        n = 0
        if isinstance(data, dict):
            # parameter generator (essentially, model.state_dict())
            for p in data.keys():
                newid = p.replace("NET46.", "").replace("Net46.", "").replace("network.", "")
                ws = data[p].data.cpu()
                sz = ws.size()
                if ws.dim() == 4 and sz[2] > min_width and sz[3] > min_width:
                    if sz[1] not in [1, 3]:
                        ws = ws.view(-1, 1, sz[2], sz[3])
                        sz = ws.size()
                    if sz[0] <= 2**10:
                        _min = ws.min(2, True)[0].min(3, True)[0]
                        _max = ws.max(2, True)[0].max(3, True)[0]
                        self.visplots.images((ws-_min)/(_max-_min),
                            nrow=max(4, int(sz[0]**0.5)),
                            opts={"title": "Ws-"+newid}, win="Ws-"+newid)
                    if png_name is not None:
                        tutils.save_image(ws, png_name.rstrip(".png") +
                            "-ws{}".format(n) + ".png")
                        n += 1
        elif isinstance(data, torch.nn.parameter.Parameter):
            # pytorch tensor or parameter
            data = data.data.cpu()
            sz = data.size()
            if len(sz) == 4 and sz[2] > min_width and sz[3] > min_width:
                if sz[1] not in [1, 3]:
                    data = data.view(-1, 1, sz[2], sz[3])
                    sz = data.size()
                if sz[0] <= 2**10:
                    _min = data.min(2, True)[0].min(3, True)[0]
                    _max = data.max(2, True)[0].max(3, True)[0]
                    self.visplots.images((data-_min)/(_max-_min),
                        nrow=max(4, int(sz[0]**0.5)),
                        opts={"title": "Ws-"+vis_name}, win="Ws-"+vis_name)
                if png_name is not None:
                    tutils.save_image(ws, png_name.rstrip(".png")+"-ws.png")
        else:
            raise NotImplementedError


# visplots = VisPlots()
# hasattr(visplots, "visplots")
# visplots.show_images(torch.rand(10, 10, 200, 200), height=32)
# visplots.show_weights(torch.nn.Parameter(torch.rand(10, 10, 7, 7)))
