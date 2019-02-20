""" TensorMONK :: plots """

import torch
import torch.nn.functional as F
import torchvision.utils as tutils
import visdom
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib
from PIL import Image as ImPIL
from torchvision import transforms
matplotlib.use('Agg')
_totensor = transforms.ToTensor()


class VisPlots(object):
    r"""Visdom plots to monitor weights (histograms and 2D kernels larger than
    3x3), and responses.

    Args:
        env: name of your environment, default = main
        server: server address, default = None
    """
    def __init__(self, env: str = "main", server: str = None):
        if server is None:
            self.visplots = visdom.Visdom(env=env)
        else:
            self.visplots = visdom.Visdom(env=env, server=server)

    def histograms(self, data, vis_name: str = "hist"):
        r""" Plots histograms of weights. For Model.state_dict(), parameter
        names are used to name the plots.

        Args:
            data: Accepts nn.Parameter, torch.Tensor and Model.state_dict()
            vis_name: required for nn.Parameter, and torch.Tensor,
                default = "hist"
        """
        if isinstance(data, dict):
            # parameter generator (essentially, model.state_dict())
            for p in data.keys():
                if "weight" in p and "weight_g" not in p and \
                   "Normalization" not in p and "bias" not in p:

                    # ignore normalization weights (gamma's & beta's) and bias
                    newid = self._trim_name(p)
                    self.visplots.histogram(X=data[p].data.cpu().view(-1),
                                            opts={"numbins": 46,
                                                  "title": newid},
                                            win=newid)
        elif (isinstance(data, torch.nn.parameter.Parameter) or
              isinstance(data, torch.Tensor)):
            # pytorch tensor or parameter
            self.visplots.histogram(X=data.data.cpu().view(-1),
                                    opts={"numbins": 46, "title": vis_name},
                                    win=vis_name)
        else:
            raise NotImplementedError

    def show_images(self, data, vis_name: str = "images", png_name: str = None,
                    normalize: bool = False, height: int = None,
                    max_samples: int = 512, attention: bool = False):
        r""" Plots responses in RGB (C=3) and grey (C=1), requires BCHW
        torch.Tensor. When C != 1/3, reorganizes the BxCxHxW to BCx1xHxC if
        attention is False, else Bx1xHxC.

        Args:
            data: 4D torch.Tensor
            vis_name: name for visdom plots, default = "images"
            png_name: used to save png images, default = None
            normalize: normalized the range to 0-1
            height: max height of image, retains aspect ratio. default = None
            max_samples: limited to speed ploting, default = 512
            attention: computes attention BxCxHxW to Bx1xHxC using l2,
                normalize is applied default = False
        """
        if isinstance(data, torch.Tensor):
            if data.dim() != 4:
                return None
            # pytorch tensor
            data = data.data.cpu()
            if attention:
                data = data.pow(2).sum(1, True).pow(.5)
            if normalize or attention:  # adjust range to 0-1
                data = self._normalize_01(data)
            # adjust 4d tensor and reduce samples when too many
            sz = data.size()
            multiplier = 1
            if sz[1] not in [1, 3]:  # BxCxHxW to BCx1xHxC
                data = data.view(-1, 1, *sz[2:])
                multiplier = sz[1]
            if sz[0]*multiplier > max_samples:
                samples = max([x*multiplier for x in range(sz[0]) if
                               x*multiplier <= max_samples][-1], multiplier)
                data = data[:samples]
            # resize image when height is not None
            if height is not None:
                sz = (height, int(float(height)*sz[3]/sz[2]))
                data = F.interpolate(data, size=sz)
            self.visplots.images(data, nrow=max(4, int(data.size(0)**0.5)),
                                 opts={"title": vis_name}, win=vis_name)
            # save a png if png_name is defined
            if png_name is not None:
                tutils.save_image(data, png_name)

    def show_weights(self, data, vis_name: str = "weights",
                     png_name: str = None, min_width: int = 3,
                     max_samples: int = 512):
        r""" Plots weights (histograms and images of 2D kernels larger than
        min_width). 2D kernels are normalized between 0-1 for visualization.
        Requires a minimum of 4 kernels to plot images.

        Args:
            data: Accepts nn.Parameter, torch.Tensor and Model.state_dict()
            vis_name: name for visdom plots, default = "weights"
            png_name: used to save png images, default = None
            min_width: only plots images if the kernel size width and height is
                above min_width
            max_samples: limited to speed ploting, default = 512
        """
        # all histograms
        self.histograms(data, vis_name)
        # only convolution weights when kernel size > 3
        n = 0
        if isinstance(data, dict):
            # parameter generator (essentially, model.state_dict())
            for p in data.keys():
                if data[p].dim() == 4 and data[p].size(2) > min_width and \
                   data[p].size(3) > min_width:
                    newid = self._trim_name(p)
                    ws = data[p].data.cpu()
                    sz = ws.size()
                    if sz[1] not in [1, 3]:
                        ws = ws.view(-1, 1, sz[2], sz[3])
                        sz = ws.size()
                    if 4 < sz[0] <= max_samples:
                        ws = self._normalize_01(ws)
                        self.visplots.images(ws, nrow=max(4, int(sz[0]**0.5)),
                                             opts={"title": "Ws-"+newid},
                                             win="Ws-"+newid)
                    if png_name is not None:
                        tutils.save_image(ws, png_name.rstrip(".png") +
                                          "-ws{}".format(n) + ".png")
                        n += 1
        elif isinstance(data, torch.nn.parameter.Parameter):
            # pytorch parameter
            if data.dim() == 4 and data.size(2) > min_width and \
               data.size(3) > min_width:
                data = data.data.cpu()
                sz = data.size()
                if sz[1] not in [1, 3]:
                    data = data.view(-1, 1, sz[2], sz[3])
                    sz = data.size()
                if sz[0] <= max_samples:
                    data = self._normalize_01(data)
                    self.visplots.images(data, nrow=max(4, int(sz[0]**0.5)),
                                         opts={"title": "Ws-"+vis_name},
                                         win="Ws-"+vis_name)
                if png_name is not None:
                    tutils.save_image(data, png_name.rstrip(".png") + "-ws-" +
                                      vis_name + ".png")
        else:
            raise NotImplementedError

    def _normalize_01(self, tensor: torch.Tensor):
        _min = tensor.min(2, True)[0].min(3, True)[0]
        _max = tensor.max(2, True)[0].max(3, True)[0]
        return tensor.add(-_min).div(_max - _min + 1e-6)

    @staticmethod
    def _trim_name(name: str):
        return name.replace("NET46.", "").replace("Net46.",
                                                  "") .replace("network.", "")

    def show_gradients(self,
                       data: list,
                       png_name: str,
                       vis_name: str = "grads",
                       width: int = 946,
                       fast: bool = True,
                       weights: bool = False):
        r""" Violin plots of gradients that need to be called after .backward()

        Args:
            data: Accepts list(named_parameters())
            vis_name: name for visdom plots, default = "grads"
            fast: a faster way using boxplots but induces interpolation
                artifacts
        """
        if isinstance(data, list):
            gs, ws, ns = [], [], []
            for n, p in data:
                if "weight" in n and "weight_g" not in n and \
                   "Normalization" not in n and "bias" not in n and \
                   p.requires_grad:
                    _n = n.replace("NET46.", "")
                    _n = _n.replace(".weight", "-ws")
                    _n = _n.replace(".Convolution", ".c")
                    if fast:
                        # sort and interpolate to width for visdom boxplot
                        _p = torch.sort(p.grad.detach().view(-1))[0]
                        _p = F.interpolate(_p.view(1, 1, -1, 1),
                                           size=(width, 1))
                        gs.append(_p.view(-1, 1))
                        ns.append(_n)
                        if weights:
                            _p = torch.sort(p.detach().view(-1))[0]
                            _p = F.interpolate(_p.view(1, 1, -1, 1),
                                               size=(width, 1))
                            ws.append(_p.view(-1, 1))
                        continue
                    gs += p.grad.abs().cpu().view(-1).numpy().tolist()
                    ns += [_n] * (len(gs) - len(ns))

            if fast:
                gs = torch.cat(gs, 1)
                self.visplots.boxplot(gs, win="grads",
                                      opts={"legend": ns, "title": "grads"})
                if weights:
                    ws = torch.cat(ws, 1)
                    self.visplots.boxplot(ws, win="weights",
                                          opts={"legend": ns,
                                                "title": "weights"})
            else:
                dFrame = pd.DataFrame({"param": ns, "grads": gs})
                f, ax = plt.subplots()
                f.canvas.draw()
                ax = sns.violinplot(x="param", y="grads", data=dFrame,
                                    palette="Set3")
                ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
                plt.tight_layout()
                plt.savefig(png_name, dpi=200)
                plt.close()
                # TODO: figure out visplot.matplot issue
                image = ImPIL.open(png_name).convert("RGB")
                sz = (width, int(image.size[1] * float(width) / image.size[0]))
                image = image.resize(sz, ImPIL.BILINEAR)
                self.visplots.image(_totensor(image), win="grads",
                                    opts={"title": "grads"},)
        else:
            raise NotImplementedError
