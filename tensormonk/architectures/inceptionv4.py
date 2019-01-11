""" TensorMONK :: architectures """


import torch
from ..layers import Stem2, InceptionA, ReductionA, InceptionB, \
    ReductionB, InceptionC, Linear


class InceptionV4(torch.nn.Sequential):
    r"""Inception-V4 implemented from https://arxiv.org/pdf/1602.07261.pdf
    Requires input size of (1, 1/3, 299, 299)

    Args:
        tensor_size: shape of tensor in BCHW
            (None/any integer >0, channels, height, width)
        activation: None/relu/relu6/lklu/elu/prelu/tanh/sigm/maxo/rmxo/swish
        dropout: 0. - 1., default = 0.1 with dropblock=True
        normalization: None/batch/group/instance/layer/pixelwise
        pre_nm: if True, normalization -> activation -> convolution else
            convolution -> normalization -> activation
        groups: grouped convolution, value must be divisble by tensor_size[1]
            and out_channels, default = 1
        weight_nm: True/False, default = False
        equalized: True/False, default = False
        shift: True/False, default = False
            Shift replaces 3x3 convolution with pointwise convs after shifting.
            Requires tensor_size[1] >= 9 and filter_size = 3
        n_embedding: when not None and > 0, adds a linear layer to the network
            and returns a torch.Tensor of shape (None, n_embedding)

    Return:
        embedding, a torch.Tensor
    """

    def __init__(self,
                 tensor_size=(6, 3, 299, 299),
                 activation: str = "relu",
                 dropout: float = 0.1,
                 normalization: str = "batch",
                 pre_nm: bool = False,
                 groups: int = 1,
                 weight_nm: bool = False,
                 equalized: bool = False,
                 shift: bool = False,
                 n_embedding: int = None,
                 *args, **kwargs):
        super(InceptionV4, self).__init__()

        print("Input", tensor_size)
        kwargs = {"activation": activation, "normalization": normalization,
                  "pre_nm": pre_nm, "groups": groups, "weight_nm": weight_nm,
                  "equalized": equalized, "shift": shift, "dropout": dropout}
        self.add_module("Stem", Stem2(tensor_size, **kwargs))
        print("Stem", self.Stem.tensor_size)

        t_size = self.Stem.tensor_size
        for i in range(4):
            self.add_module("InceptionA"+str(i), InceptionA(t_size, **kwargs))
            t_size = getattr(self, "InceptionA"+str(i)).tensor_size
            print("InceptionA", t_size)

        self.add_module("ReductionA", ReductionA(t_size, **kwargs))
        t_size = self.ReductionA.tensor_size
        print("ReductionA", t_size)

        for i in range(7):
            self.add_module("InceptionB"+str(i), InceptionB(t_size, **kwargs))
            t_size = getattr(self, "InceptionB"+str(i)).tensor_size
            print("InceptionB", t_size)

        self.add_module("ReductionB", ReductionB(t_size, **kwargs))
        t_size = self.ReductionB.tensor_size
        print("ReductionB", t_size)

        for i in range(3):
            self.add_module("InceptionC"+str(i), InceptionC(t_size, **kwargs))
            t_size = getattr(self, "InceptionC"+str(i)).tensor_size
            print("InceptionC", t_size)

        self.add_module("AveragePool", torch.nn.AvgPool2d(t_size[2:]))
        print("AveragePool", (1, t_size[1], 1, 1))
        self.tensor_size = (6, t_size[1], 1, 1)

        if n_embedding is not None and n_embedding > 0:
            self.add_module("Embedding", Linear(self.tensor_size, n_embedding,
                                                "", 0., False))
            self.tensor_size = (6, n_embedding)
            print("Linear", (1, n_embedding))


# from tensormonk.layers import Stem2, InceptionA, ReductionA, InceptionB, \
#     ReductionB, InceptionC, Linear
# tensor_size = (1, 3, 299, 299)
# tensor = torch.rand(*tensor_size)
# test = InceptionV4(tensor_size)
# %timeit test(tensor).size()
