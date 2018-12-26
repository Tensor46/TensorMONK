""" TensorMONK's :: NeuralArchitectures                                     """

import torch
from ..NeuralLayers import Convolution, Linear


class MobileNetV1(torch.nn.Sequential):
    """
        Implemented https://arxiv.org/pdf/1704.04861.pdf

        To replicate the paper, use default parameters
        Works fairly well, for tensor_size of min(height, width) >= 128
    """
    def __init__(self,
                 tensor_size=(6, 3, 224, 224),
                 activation: str = "relu",
                 normalization: str = "batch",
                 pre_nm: bool = False,
                 weight_nm: bool = False,
                 equalized: bool = False,
                 shift: bool = False,
                 n_embedding: int = None,
                 *args, **kwargs):
        super(MobileNetV1, self).__init__()

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

        kwargs = {"activation": activation, "normalization": normalization,
                  "weight_nm": weight_nm, "equalized": equalized,
                  "shift": shift}

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
        self.tensor_size = (6, 1024)

        if n_embedding is not None and n_embedding > 0:
            self.add_module("Embedding", Linear(self.tensor_size, n_embedding,
                                                "", 0., False))
            self.tensor_size = (6, n_embedding)
            print("Linear", (1, n_embedding))


# from core.NeuralLayers import Convolution, Linear
# tensor_size = (1, 3, 224, 224)
# tensor = torch.rand(*tensor_size)
# test = MobileNetV1(tensor_size, n_embedding=64)
# test(tensor).size()
# %timeit test(tensor).size()
