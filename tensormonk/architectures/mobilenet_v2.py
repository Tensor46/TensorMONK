""" TensorMONK's :: architectures :: MobileNetV2 """


import torch
import torch.nn as nn
from PIL import Image as ImPIL
import torchvision
from ..layers import Convolution, Linear, MBBlock
from ..detection import CONFIG
to_tensor = torchvision.transforms.ToTensor()


class MobileNetV2(torch.nn.Module):
    r"""MobileNetV2 implemented from https://arxiv.org/pdf/1801.04381.pdf
    Designed for input size of (1, 1/3, 224, 224), works for
    min(height, width) >= 32.
        if min(height, width) <= 128: 2nd MBBlock stride is changed from 2 to 1
        if min(height, width) <=  64: 1st Conv2D  stride is changed from 2 to 1
            the above two conditions are disabled when config is not None
        if min(height, width) >  320: 1st MBBlock stride is changed from 1 to 2


    When config is not None, last few layers and classifier are ignored to
    make MobileNetV2 available for detection. For detection tasks, the features
    at strides 4, 8, 16 and 32 are returned (for max(tensor_size[2:]) > 320,
    responses at strides 8, 16, 32 and 64 are returned).

    *Pretrained weights are from PyTorch.
    https://pytorch.org/docs/stable/torchvision/
        models.html?highlight=vision#torchvision.models.mobilenet_v2

    Args:
        tensor_size (tuple): shape of tensor in BCHW
            (None/any integer >0, channels, height, width)

            default = (1, 3, 224, 224)

        activation (str): Refer tensormonk.activations.Activations

            default = "relu6"

        dropout (float): dropout probability

            default = 0.

        normalization (str): Refer tensormonk.normalizations.Normalizations

            default = "batch"

        pre_nm (bool): if True, normalization -> activation -> convolution else
            convolution -> normalization -> activation

            default = False

        weight_nm (bool): https://arxiv.org/pdf/1602.07868.pdf

            default = False

        equalized (bool): https://arxiv.org/pdf/1710.10196.pdf

            default = False

        n_embedding (int): when not None and > 0, adds a linear layer to the
            network and returns a torch.Tensor of shape (None, n_embedding).
            Only works when config is None.

            default = None

        config (CONFIG): This is a detection config. When config is not None,
            all the above arguments are overwritten by config parameters.

            default: None

        pretrained (bool): When True, loads pretrained weights provided by
            pytorch. Activation, dropout, normalization, pre_nm, weight_nm and
            equalized are set to defaults. BatchNorm2d is replace with
            FrozenBatch2D.

            default: True

        predict_imagenet (bool): When True (+ pretrained=True), adds linear
            layer for imagenet labels predictions (automatically, ignores
            n_embedding).

            options: True | False
            default: False

    Return:
        if n_embedding = None, imagenet_labels = False, config = None
            embedding of length 1280, a torch.Tensor

        if n_embedding = 600,  imagenet_labels = False, config = None
            embedding of length  600, a torch.Tensor

        if n_embedding = None, imagenet_labels =  True, config = None
            predictions of length 1000, a torch.Tensor

        if n_embedding = None, imagenet_labels =  False, config = CONFIG
            tuple of tensors

    """

    URL = "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth"

    def __init__(self,
                 tensor_size: tuple = (1, 3, 224, 224),
                 activation: str = "relu6",
                 dropout: float = 0.1,
                 normalization: str = "batch",
                 pre_nm: bool = False,
                 weight_nm: bool = False,
                 equalized: bool = False,
                 n_embedding: int = None,
                 config: CONFIG = None,
                 pretrained: bool = True,
                 predict_imagenet: bool = False,
                 **kwargs):
        super(MobileNetV2, self).__init__()

        self.detection = False
        self.t_size = tensor_size
        if config is not None:
            if not isinstance(config, CONFIG):
                raise TypeError("MobileNetV2: config != "
                                "tensormonk.detection.CONFIG")
            self.detection = True
            self.t_size = config.t_size
            self.architecture = config.base_network.lower()

        stride1 = 1 if min(self.t_size[2:]) <= 64 else 2
        stride2 = 2 if min(self.t_size[2:]) > 320 else 1
        stride3 = 1 if min(self.t_size[2:]) <= 128 else 2
        if self.detection:
            stride1, stride3 = 2, 2
            stride2 = 2 if max(self.t_size[2:]) > 320 or \
                config.base_network_forced_stride else 1

        if pretrained:
            # overwritten to defaults
            activation, dropout, normalization = "relu6", 0., "frozenbatch"
            pre_nm, weight_nm, equalized = False, False, False

        kwargs["pad"] = True
        kwargs["activation"] = activation
        kwargs["normalization"] = normalization
        kwargs["pre_nm"] = False
        kwargs["weight_nm"] = weight_nm
        kwargs["equalized"] = equalized

        block_params = [(16, stride2, 1), (24, stride3, 6), (24, 1, 6),
                        (32, 2, 6), (32, 1, 6), (32, 1, 6),
                        (64, 2, 6), (64, 1, 6), (64, 1, 6), (64, 1, 6),
                        (96, 1, 6), (96, 1, 6), (96, 1, 6),
                        (160, 2, 6), (160, 1, 6), (160, 1, 6),
                        (320, 1, 6)]

        net = [Convolution(tensor_size, 3, 32, stride1, **kwargs)]
        kwargs["pre_nm"] = pre_nm
        for i, (oc, s, t) in enumerate(block_params):
            kwargs["expansion"] = t
            net += [MBBlock(net[-1].tensor_size, 3, oc, s, **kwargs)]

        if not self.detection:
            net += [Convolution(net[-1].tensor_size, 1, 1280, 1,
                                dropout=dropout, **kwargs)]
            net += [nn.AdaptiveAvgPool2d(1)]
            if predict_imagenet and pretrained:
                net += [Linear((1, 1280), 1000)]

        self._layer_1 = nn.Sequential(*net[:4])
        self._layer_2 = nn.Sequential(*net[4:7])
        self._layer_3 = nn.Sequential(*net[7:14])
        self._layer_4 = nn.Sequential(*net[14:18])
        if not self.detection:
            self._layer_5 = nn.Sequential(*net[18:])

        if pretrained:
            self.load_pretrained()
            self.register_buffer(
                "mean", torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer(
                "std",  torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, tensor: torch.Tensor):
        if hasattr(self, "mean"):
            tensor = (tensor - self.mean) / (self.std + 1e-8)

        x1 = self._layer_1(tensor)
        x2 = self._layer_2(x1)
        x3 = self._layer_3(x2)
        x4 = self._layer_4(x3)
        if self.detection:
            return (x1, x2, x3, x4)
        return self._layer_5(x4).view(tensor.size(0), -1)

    def load_pretrained(self):
        from torchvision.models.utils import load_state_dict_from_url
        ws_a = self.state_dict()
        ws_b = load_state_dict_from_url(MobileNetV2.URL, progress=True)
        for a, b in zip(ws_a.keys(), ws_b.keys()):
            if ws_a[a].shape == ws_b[b].shape:
                ws_a[a].data = ws_b[b].to(ws_a[a].data.device)
            else:
                print(a, b)
                print(ws_a[a].shape, ws_b[b].shape)
        self.load_state_dict(ws_a)

    def preprocess(self, image: str):
        if isinstance(image, str):
            image = ImPIL.open(image)
        image = image.convert("RGB")
        image = image.resize(self.t_size[2:][::-1], ImPIL.BILINEAR)
        return to_tensor(image).unsqueeze(0)
