""" TensorMONK :: architectures """

import torch
from ..layers import Convolution, ResidualShuffle, Linear


class ShuffleNet(torch.nn.Sequential):
    r"""Versions of ShuffleNet. With the ability to adjust the strides of
    initial convolution and remove max pool the models works for all the
    min(height, width) >= 32. To replicate the paper, use default parameters
    (and select type). Implemented from https://arxiv.org/pdf/1707.01083.pdf

    Args:
        tensor_size: shape of tensor in BCHW
            (None/any integer >0, channels, height, width)
        type (string): model type (g1/g2/g3/g4/g8), default = g4
        activation: None/relu/relu6/lklu/elu/prelu/tanh/sigm/maxo/rmxo/swish,
            default = relu
        dropout: 0. - 1., default = 0.1 with dropblock=True
        normalization: None/batch/group/instance/layer/pixelwise,
            default = batch
        pre_nm: if True, normalization -> activation -> convolution else
            convolution -> normalization -> activation
            default = True
        weight_nm: True/False, default = False
        shift: True/False, default = False
            Shift replaces 3x3 convolution with pointwise convs after shifting.
            Requires tensor_size[1] >= 9 and filter_size = 3
        n_embedding: when not None and > 0, adds a linear layer to the network
            and returns a torch.Tensor of shape (None, n_embedding)
        pretrained: downloads and updates the weights with pretrained weights

    Return:
        embedding (a torch.Tensor)
    """
    def __init__(self,
                 tensor_size=(6, 3, 224, 224),
                 type: str = "g4",
                 activation: str = "relu",
                 dropout: float = 0.1,
                 normalization: str = "batch",
                 pre_nm: bool = True,
                 weight_nm: bool = False,
                 equalized: bool = False,
                 shift: bool = False,
                 n_embedding: int = None,
                 *args, **kwargs):
        super(ShuffleNet, self).__init__()

        assert type.lower() in ("g1", "g2", "g3", "g4", "g8"), \
            "ShuffleNet -- type must be g1/g2/g3/g4/g8"

        if type.lower() == "g1":
            groups = 1
            block_params = [(144, 2)] + [(144, 1)]*3 + \
                           [(288, 2)] + [(288, 1)]*7 + \
                           [(576, 2)] + [(576, 1)]*3
        elif type.lower() == "g2":
            groups = 2
            block_params = [(200, 2)] + [(200, 1)]*3 + \
                           [(400, 2)] + [(400, 1)]*7 + \
                           [(800, 2)] + [(800, 1)]*3
        elif type.lower() == "g3":
            groups = 3
            block_params = [(240, 2)] + [(240, 1)]*3 + \
                           [(480, 2)] + [(480, 1)]*7 + \
                           [(960, 2)] + [(960, 1)]*3
        elif type.lower() == "g4":
            groups = 4
            block_params = [(272, 2)] + [(272, 1)]*3 + \
                           [(544, 2)] + [(544, 1)]*7 + \
                           [(1088, 2)] + [(1088, 1)]*3
        elif type.lower() == "g8":
            groups = 8
            block_params = [(384, 2)] + [(384, 1)]*3 + \
                           [(768, 2)] + [(768, 1)]*7 + \
                           [(1536, 2)] + [(1536, 1)]*3
        else:
            raise NotImplementedError

        print("Input", tensor_size)
        s = 2
        if min(tensor_size[2], tensor_size[3]) < 64:
            # Addon -- To make it flexible for other tensor_size's
            s = 1
            print("ShuffleNet: InitialConvolution strides changed from 2 to 1,"
                  + " as min(tensor_size[2], tensor_size[3]) <  64")

        kwargs = {"activation": activation, "normalization": normalization,
                  "weight_nm": weight_nm, "equalized": equalized,
                  "shift": shift, "pad": True, "groups": 1}
        self.add_module("InitialConvolution",
                        Convolution(tensor_size, 3, 24, s,
                                    pre_nm=False, **kwargs))
        t_size = self.InitialConvolution.tensor_size
        print("InitialConvolution", t_size)

        if min(tensor_size[2], tensor_size[3]) > 128:
            self.add_module("MaxPool", torch.nn.MaxPool2d(3, 2, padding=1))
            h, w = t_size[2:]
            t_size = (1, 24, h//2 + (1 if h % 2 == 1 else 0),
                      w//2 + (1 if w % 2 == 1 else 0))
            print("MaxPool", t_size)
        else:
            # Addon -- To make it flexible for other tensor_size's
            print("ShuffleNet: MaxPool is ignored if min(h, w) <=  128")

        kwargs["groups"] = groups
        kwargs["dropout"] = dropout
        for i, (oc, s) in enumerate(block_params):
            self.add_module("Shuffle-"+str(i),
                            ResidualShuffle(t_size, 3, oc, s, **kwargs))
            t_size = getattr(self, "Shuffle-"+str(i)).tensor_size
            print("Shuffle-"+str(i), t_size)

        self.add_module("AveragePool", torch.nn.AvgPool2d(t_size[2:]))
        print("AveragePool", (1, oc, 1, 1))
        self.tensor_size = (1, oc)

        if n_embedding is not None and n_embedding > 0:
            self.add_module("Embedding", Linear(self.tensor_size, n_embedding,
                                                "", 0., False))
            self.tensor_size = (1, n_embedding)
            print("Linear", (1, n_embedding))


# from tensormonk.layers import Convolution, ResidualShuffle, Linear
# tensor_size = (1, 3, 224, 224)
# tensor = torch.rand(*tensor_size)
# test = ShuffleNet(tensor_size, "g8")
# test(tensor).size()
