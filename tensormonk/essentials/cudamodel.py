""" TensorMONK's :: NeuralEssentials                                        """

import torch
import torch.nn as nn


class CudaModel(torch.nn.Module):
    """ Works on both CPU & GPU """
    def __init__(self, is_cuda, gpus, net, net_kwargs):
        super(CudaModel, self).__init__()

        self.gpus = gpus
        self.is_cuda = is_cuda
        self.NET46 = net(**net_kwargs)
        self.tensor_size = self.NET46.tensor_size

    def forward(self, inputs):
        inputs = self.check_precision_device(inputs)
        if type(inputs) in [list, tuple]:
            return self.NET46(*inputs)
        else:
            if self.is_cuda and self.gpus > 1:
                return nn.parallel.data_parallel(self.NET46, inputs,
                                                 range(self.gpus))
            else:
                return self.NET46(inputs)

    def check_precision_device(self, inputs):
        r"""Converts the inputs to float or half using parameter precision and
        to cuda if is_cuda.
        """
        if not hasattr(self, "precision"):
            for p in self.parameters():
                break
            self.precision = p.dtype if "p" in locals() else torch.float32
        if type(inputs) in [list, tuple]:
            inputs = [x if x.dtype == torch.long else x.type(self.precision)
                      for x in inputs]
            if self.is_cuda:
                inputs = [x.cuda() for x in inputs]
            return inputs
        else:
            if not (inputs.dtype == torch.long):
                inputs = inputs.type(self.precision)
            if self.is_cuda:
                inputs = inputs.cuda()
        return inputs

    def regularize_weights(self, clip=0., only_convs=False, l2_factor=0.):
        r"""Does several weight regulaizations. All the weights are renormalized
        using l2 (exceptions - bias, gamma, beta)
        For convolutions with kernel height*width > 1 -- l2-norm is done
        on the dim-2 and dim-3 and divided by sqrt(number of input channels).
        For convolutions with kernel height*width = 1 -- bounds are normalized
        between -1 to 1 without sign change and divided by sqrt(number of input
        channels).
        For routing capsule weights which are 3D (N, I, O) i.e, we have N IxO
        linear layers. l2-norm is done on dim-1.
        For linear layer weights which are 2D (out_channels, in_channels) --
        l2-norm is done on dim-1.

        Args
            clip: when > 0., does parameters.clip(-clip, clip) before l2-norm
            only_convs: True/False l2-norm is restricted to convolutional
            layers l2_factor: a factor of l2-norm added to weights
        """
        if self.training:
            self.clip_weights(clip)
            for name, p in self.NET46.named_parameters():

                if p.data.ndimension() == 4:
                    # convolution
                    if p.data.size(2)*p.data.size(3) > 1:
                        # ignore 1x1's -- does l2 and considers input channels
                        l2 = p.data.norm(2, 2, True).norm(2, 3, True)
                        p.data.div_(l2.add(1e-8)).div_(p.size(1)**0.5)
                        if l2_factor != 0.:
                            p.data.add_(l2.mul(l2_factor))
                    else:
                        bounds = p.data.abs().max(1, True)[0]
                        p.data.div_(bounds).div_(p.size(1)**0.5)
                        if l2_factor != 0.:
                            l2 = p.data.norm(2, 1, True)
                            p.data.add_(l2.mul(l2_factor))

                elif p.data.ndimension() == 3 and not only_convs:
                    # routing capsule - has K MxN linears - normalize M
                    l2 = p.data.norm(2, 1, True)
                    p.data.div_(l2.add(1e-8))
                    if l2_factor != 0.:
                        p.data.add_(l2.mul(l2_factor))

                elif p.data.ndimension() == 2 and not only_convs:
                    if name.endswith("centers"):  # avoid centers
                        continue
                    # fully-connected and lossfunctions
                    l2 = p.data.norm(2, 1, True)
                    p.data.div_(l2.add(1e-8))
                    if l2_factor != 0.:
                        p.data.add_(l2.mul(l2_factor))

    def clip_weights(self, clip):
        if self.training:
            if not isinstance(clip, float):
                clip = 0.
            if clip > 0.:
                for p in self.NET46.parameters():
                    if p.data.ndimension() in [2, 3, 4]:
                        p.data.clamp_(-clip, clip)
