""" TensorMONK's :: NeuralArchitectures                                      """

__all__ = ["CapsuleNet", "SimpleNet", "ShuffleNet", "MobileNetV1", "MobileNetV2",
           "ResidualNet", "InceptionV4", "LinearVAE", "ConvolutionalVAE", "Models"]

from .capsulenet import CapsuleNet
from .simplenet import SimpleNet
from .shufflenet import ShuffleNet
from .mobilenetv1 import MobileNetV1
from .mobilenetv2 import MobileNetV2
from .residualnet import ResidualNet
from .inceptionv4 import InceptionV4
from .linearvae import LinearVAE
from .convolutionalvae import ConvolutionalVAE
from .pggan import PGGAN

del pggan
del capsulenet
del simplenet
del shufflenet
del mobilenetv1
del mobilenetv2
del residualnet
del inceptionv4
del linearvae
del convolutionalvae

def Models(Architecture):
    Architecture = Architecture.lower()
    if Architecture == "residual18":
        embedding_net = ResidualNet
        embedding_net_kwargs = {"type": "r18"}
    elif Architecture == "residual34":
        embedding_net = ResidualNet
        embedding_net_kwargs = {"type": "r34"}
    elif Architecture == "residual50":
        embedding_net = ResidualNet
        embedding_net_kwargs = {"type": "r50"}
    elif Architecture == "residual101":
        embedding_net = ResidualNet
        embedding_net_kwargs = {"type": "r101"}
    elif Architecture == "residual152":
        embedding_net = ResidualNet
        embedding_net_kwargs = {"type": "r152"}
    elif Architecture == "resnext50":
        embedding_net = ResidualNet
        embedding_net_kwargs = {"type": "rn50"}
    elif Architecture == "resnext101":
        embedding_net = ResidualNet
        embedding_net_kwargs = {"type": "rn101"}
    elif Architecture == "resnext152":
        embedding_net = ResidualNet
        embedding_net_kwargs = {"type": "rn152"}
    elif Architecture == "seresidual50":
        embedding_net = ResidualNet
        embedding_net_kwargs = {"type": "ser50"}
    elif Architecture == "seresidual101":
        embedding_net = ResidualNet
        embedding_net_kwargs = {"type": "ser101"}
    elif Architecture == "seresidual152":
        embedding_net = ResidualNet
        embedding_net_kwargs = {"type": "ser152"}
    elif Architecture == "inceptionv4":
        embedding_net = InceptionV4
        embedding_net_kwargs = {}
        tensor_size = (1, 3, 299, 299)
    elif Architecture == "mobilev1":
        embedding_net = MobileNetV1
        embedding_net_kwargs = {}
    elif Architecture == "mobilev2":
        embedding_net = MobileNetV2
        embedding_net_kwargs = {}
    elif Architecture == "shuffle1":
        embedding_net = ShuffleNet
        embedding_net_kwargs = {"type": "g1"}
    elif Architecture == "shuffle2":
        embedding_net = ShuffleNet
        embedding_net_kwargs = {"type": "g2"}
    elif Architecture == "shuffle3":
        embedding_net = ShuffleNet
        embedding_net_kwargs = {"type": "g3"}
    elif Architecture == "shuffle4":
        embedding_net = ShuffleNet
        embedding_net_kwargs = {"type": "g4"}
    elif Architecture == "shuffle8":
        embedding_net = ShuffleNet
        embedding_net_kwargs = {"type": "g8"}
    else:
        raise NotImplementedError
    return embedding_net, embedding_net_kwargs
