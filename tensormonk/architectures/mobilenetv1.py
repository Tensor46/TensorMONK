""" TensorMONK :: architectures """

import torch
from ..layers import Convolution, Linear
from ..layers.utils import compute_flops


class MobileNetV1(torch.nn.Sequential):
    r"""MobileNetV1 implemented from https://arxiv.org/pdf/1704.04861.pdf
    Designed for input size of (1, 1/3, 224, 224), works for
    min(height, width) >= 32

    Args:
        tensor_size: shape of tensor in BCHW
            (None/any integer >0, channels, height, width)
        activation: None/relu/relu6/lklu/elu/prelu/tanh/sigm/maxo/rmxo/swish
        dropout: 0. - 1., default = 0.1 with dropblock=True
        normalization: None/batch/group/instance/layer/pixelwise
        pre_nm: if True, normalization -> activation -> convolution else
            convolution -> normalization -> activation
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
                 tensor_size: tuple = (1, 3, 224, 224),
                 activation: str = "relu",
                 dropout: float = 0.1,
                 normalization: str = "batch",
                 pre_nm: bool = False,
                 weight_nm: bool = False,
                 equalized: bool = False,
                 shift: bool = False,
                 n_embedding: int = None,
                 *args, **kwargs):
        super(MobileNetV1, self).__init__()

        import numpy as np
        block_params = [(3, 32, 2, 1), (3, 32, 1, 32),
                        (1, 64, 1, 1), (3, 64, 2, 64),
                        (1, 128, 1, 1), (3, 128, 1, 128),
                        (1, 128, 1, 1), (3, 128, 2, 128),
                        (1, 256, 1, 1), (3, 256, 1, 256),
                        (1, 256, 1, 1), (3, 256, 2, 256),
                        (1, 512, 1, 1), (3, 512, 1, 512),
                        (1, 512, 1, 1), (3, 512, 1, 512),
                        (1, 512, 1, 1), (3, 512, 1, 512),
                        (1, 512, 1, 1), (3, 512, 1, 512),
                        (1, 512, 1, 1), (3, 512, 1, 512),
                        (1, 512, 1, 1), (3, 512, 2, 512),
                        (1, 1024, 1, 1), (3, 1024, 1, 1024), (1, 1024, 1, 1)]

        if min(tensor_size[2], tensor_size[3]) <= 64:
            block_params[3] = (3, 64, 1, 64)
        if min(tensor_size[2], tensor_size[3]) <= 128:
            block_params[0] = (3, 32, 1, 1)

        kwargs = {"activation": activation, "normalization": normalization,
                  "weight_nm": weight_nm, "equalized": equalized,
                  "shift": shift, "dropout": dropout}

        print("Input", tensor_size)
        t_size = tensor_size
        for i, (k, oc, s, g) in enumerate(block_params):
            self.add_module("Mobile"+str(i),
                            Convolution(t_size, k, oc, s, groups=g,
                                        pre_nm=False if i == 0 else pre_nm,
                                        **kwargs))
            t_size = getattr(self, "Mobile"+str(i)).tensor_size
            print("Mobile"+str(i), t_size)

        self.add_module("AveragePool", torch.nn.AvgPool2d(t_size[2:]))
        print("AveragePool", (1, 1024, 1, 1))
        self.pool_flops = (np.prod(t_size[2:]) * 2 - 1) * t_size[1]
        self.tensor_size = (1, 1024)

        if n_embedding is not None and n_embedding > 0:
            self.add_module("Embedding", Linear(self.tensor_size, n_embedding,
                                                "", 0., False))
            self.tensor_size = (1, n_embedding)
            print("Linear", (1, n_embedding))

    def flops(self):
        # all operations
        return compute_flops(self) + self.pool_flops


# from tensormonk.layers import Convolution, Linear
# from tensormonk.layers.utils import compute_flops
# tensor_size = (1, 3, 224, 224)
# tensor = torch.rand(*tensor_size)
# test = MobileNetV1(tensor_size, n_embedding=0)
# test(tensor).size()
# test.flops() / 1000 / 1000 / 1000
# %timeit test(tensor).size()
