""" TensorMONK's :: architectures :: MNAS """

__all__ = ["MNAS"]

import torch
import torch.nn as nn
from ..layers import Convolution, Linear, MBBlock
from ..detection import CONFIG
from PIL import Image as ImPIL
import torchvision
to_tensor = torchvision.transforms.ToTensor()


class MNAS(torch.nn.Module):
    r"""MNAS based on https://arxiv.org/pdf/1807.11626.pdf, varies from
    MnasNet-A1 to use the pretrained weights from pytorch (however, se-block
    can be enabled for all the blocks by passing an additional argument
    seblock=True).

    Designed for size (1, 3, 224, 224), adds additional stride at second
    convolution if max(height, width) > 320.
    When config is not None, last few layers and classifier are ignored to
    make MNAS available for detection tasks. For detection tasks, the features
    at strides 4, 8, 16 and 32 are returned (for max(tensor_size[2:]) > 320,
    responses at strides 8, 16, 32 and 64 are returned).

    *All the pretrained weights are from PyTorch.
    https://pytorch.org/docs/stable/torchvision/
        models.html?highlight=vision#torchvision.models.mnasnet0_5
    https://pytorch.org/docs/stable/torchvision/
        models.html?highlight=vision#torchvision.models.mnasnet1_0

    Args:
        tensor_size (tuple): shape of tensor in BCHW
            (None/any integer >0, channels, height, width)

            default = (1, 3, 224, 224)

        architecture (str): MNAS architectures

            options = "mnas_050" | "mnas_100"
            default = "mnas_100"

        activation (str): Refer tensormonk.activations.Activations

            default = "relu"

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

    URLS = {"mnas_050": ("https://download.pytorch.org/models"
                         "/mnasnet0.5_top1_67.592-7c6cb539b9.pth"),
            "mnas_100": ("https://download.pytorch.org/models"
                         "/mnasnet1.0_top1_73.512-f206786ef8.pth")}

    def __init__(self,
                 tensor_size: tuple = (1, 3, 224, 224),
                 architecture: str = "mnas_100",
                 activation: str = "relu",
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
        super(MNAS, self).__init__()

        self.detection = False
        self.t_size = tensor_size
        self.architecture = architecture.lower()
        additional_stride = 2 if max(self.t_size[2:]) > 320 else 1
        if config is not None:
            if not isinstance(config, CONFIG):
                raise TypeError("MNAS: config != tensormonk.detection.CONFIG")
            self.detection = True
            self.t_size = config.t_size
            self.architecture = config.base_network.lower()
            additional_stride = 2 if max(self.t_size[2:]) > 320 or \
                config.base_network_forced_stride else 1
            pretrained = config.base_network_pretrained

        if pretrained:
            # overwritten to defaults
            activation, dropout, normalization = "relu", 0., "frozenbatch"
            pre_nm, weight_nm, equalized = False, False, False

        kwargs["pad"] = True
        kwargs["activation"] = activation
        kwargs["normalization"] = normalization
        kwargs["pre_nm"] = pre_nm
        kwargs["weight_nm"] = weight_nm
        kwargs["equalized"] = equalized

        if self.architecture == "mnas_050":
            nc = (16, 24, 40, 48, 96, 160)
        elif self.architecture == "mnas_075":
            nc = (24, 32, 64, 72, 144, 240)
        elif self.architecture == "mnas_100":
            nc = (24, 40, 80, 96, 192, 320)
        elif self.architecture == "mnas_130":
            nc = (32, 56, 104, 128, 248, 416)
        else:
            raise ValueError("MNAS: architecture is not valid!")

        net = [Convolution(tensor_size, 3, 32, 2, **kwargs)]
        kwargs["pre_nm"] = pre_nm
        net += [Convolution(net[-1].tensor_size, 3, 32, additional_stride,
                            groups=32, **kwargs)]
        kwargs["activation"] = None
        net += [Convolution(net[-1].tensor_size, 1, 16, **kwargs)]
        kwargs["activation"] = activation
        kwargs["expansion"] = 3
        net += [MBBlock(net[-1].tensor_size, 3, nc[0], 2, **kwargs)]
        net += [MBBlock(net[-1].tensor_size, 3, nc[0], 1, **kwargs)]
        net += [MBBlock(net[-1].tensor_size, 3, nc[0], 1, **kwargs)]
        net += [MBBlock(net[-1].tensor_size, 5, nc[1], 2, **kwargs)]
        net += [MBBlock(net[-1].tensor_size, 5, nc[1], 1, **kwargs)]
        net += [MBBlock(net[-1].tensor_size, 5, nc[1], 1, **kwargs)]
        kwargs["expansion"] = 6
        net += [MBBlock(net[-1].tensor_size, 5, nc[2], 2, **kwargs)]
        net += [MBBlock(net[-1].tensor_size, 5, nc[2], 1, **kwargs)]
        net += [MBBlock(net[-1].tensor_size, 5, nc[2], 1, **kwargs)]
        net += [MBBlock(net[-1].tensor_size, 3, nc[3], 1, **kwargs)]
        net += [MBBlock(net[-1].tensor_size, 3, nc[3], 1, **kwargs)]
        net += [MBBlock(net[-1].tensor_size, 5, nc[4], 2, **kwargs)]
        net += [MBBlock(net[-1].tensor_size, 5, nc[4], 1, **kwargs)]
        net += [MBBlock(net[-1].tensor_size, 5, nc[4], 1, **kwargs)]
        net += [MBBlock(net[-1].tensor_size, 5, nc[4], 1, **kwargs)]
        net += [MBBlock(net[-1].tensor_size, 3, nc[5], 1, **kwargs)]

        if not self.detection:
            net += [Convolution(net[-1].tensor_size, 1, 1280, 1,
                                dropout=dropout, **kwargs)]
            net += [nn.AdaptiveAvgPool2d(1)]
            if predict_imagenet and pretrained:
                net += [Linear((1, 1280), 1000)]

        self._layer_1 = nn.Sequential(*net[:6])
        self._layer_2 = nn.Sequential(*net[6:9])
        self._layer_3 = nn.Sequential(*net[9:14])
        self._layer_4 = nn.Sequential(*net[14:19])
        if not self.detection:
            self._layer_5 = nn.Sequential(*net[19:])

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
        if self.architecture not in ("mnas_050", "mnas_100"):
            print("No weight file")
            return

        from torchvision.models.utils import load_state_dict_from_url
        url = MNAS.URLS[self.architecture]
        ws_a = self.state_dict()
        ws_b = load_state_dict_from_url(url, progress=True)
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
