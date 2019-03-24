""" TensorMONK :: layers :: Linear """

__all__ = ["Linear", ]

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..activations import Activations


class Linear(nn.Module):
    r"""Linear layer with built-in dropout and activations. (Moved out of
    nn.Linear and a fix is available in LoadModels to convert old model
    weights).

    Args:
        tensor_size (int/list/tuple): shape of tensor in
            (None/any integer >0, channels, height, width) or
            (None/any integer >0, in_features) or in_features
        out_features (int): output features, tensor.size(1)
        activation: None/relu/relu6/lklu/elu/prelu/tanh/sigm/maxo/rmxo/swish,
            default = None
        dropout (float): 0. - 1., default = 0.
        bias (bool): to enable bias, default = True
        out_shape (tuple): a desired output shape in tuple with out batches

    Return:
        torch.Tensor of shape (B, out_features)

    """
    def __init__(self,
                 tensor_size,
                 out_features,
                 activation: str = None,
                 dropout: float = 0.,
                 bias: bool = True,
                 out_shape: tuple = None,
                 **kwargs):
        super(Linear, self).__init__()

        show_msg = "x".join(["_"]+list(map(str, tensor_size[1:]))) + " -> "
        # Checks
        if not type(tensor_size) in [int, list, tuple]:
            raise TypeError("Linear: tensor_size must be int/tuple/list: " +
                            "{}".format(type(tensor_size).__name__))
        tensor_size = tensor_size if isinstance(tensor_size, int) else \
            np.prod(tensor_size[1:])

        if not isinstance(out_features, int):
            raise TypeError("Linear: out_features must be int: " +
                            "{}".format(type(out_features).__name__))

        if not (activation is None or type(activation) == str):
            raise TypeError("Linear: activation must be None/str: " +
                            "{}".format(type(activation).__name__))
        if isinstance(activation, str):
            activation = activation.lower()
        if not (activation in [None, "", ] + Activations.available()):
            raise ValueError("Linear: Invalid activation: " +
                             "{}".format(activation))
        # compensate channels for maxout
        multiplier = 2 if activation in ("maxo", "rmxo") else 1

        if not type(dropout) is float:
            raise TypeError("Linear: dropout must be float: " +
                            "{}".format(type(dropout).__name__))
        if not 0. <= dropout < 1:
            raise ValueError("Linear: dropout must be >=0 and <1: " +
                             "{}".format(dropout))
        if dropout > 0.:
            self.dropout = nn.Dropout(dropout)
            show_msg += "dropout({}) -> ".format(dropout)

        if not type(bias) == bool:
            raise TypeError("Linear: bias must be boolean: " +
                            "{}".format(type(bias).__name__))

        if out_shape is not None:
            if not np.prod(out_shape) == out_features:
                raise ValueError("Linear: np.prod(out_shape) != out_features" +
                                 ": {}".format(out_shape))
            self.out_shape = out_shape

        # get weight and bias
        w = torch.randn(out_features*multiplier, tensor_size)
        self.weight = nn.Parameter(F.normalize(w, p=2, dim=1))
        self.weight.data.normal_(0., 0.02)
        show_msg += "linear({}x{}) -> ".format(*self.weight.shape)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features*multiplier))
        # get activation function
        if activation is not None:
            if activation in Activations.available():
                self.activation = Activations((None, out_features), activation,
                                              **kwargs)
                show_msg += activation + " -> "
        # out tensor size
        self.tensor_size = (1, out_features)
        if hasattr(self, "out_shape"):
            self.tensor_size = tuple([1, ] + list(out_shape))
        show_msg += "x".join(["_"]+[str(x)for x in self.tensor_size[1:]])
        self.show_msg = show_msg

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() > 2:
            tensor = tensor.view(tensor.size(0), -1)
        if hasattr(self, "dropout"):
            tensor = self.dropout(tensor)
        tensor = tensor.mm(self.weight.t())
        if hasattr(self, "bias"):
            tensor = tensor + self.bias.view(1, -1)
        if hasattr(self, "activation"):
            tensor = self.activation(tensor)
        if hasattr(self, "out_shape"):
            tensor = tensor.view(-1, *self.out_shape)
        return tensor

    def __repr__(self):
        return self.show_msg

    def flops(self):
        flops = np.prod(self.weight.shape) * self.weight.shape[1]
        if hasattr(self, "bias"):
            flops += self.bias.numel()
        if hasattr(self, "activation"):
            flops += self.activation.flops()
        return flops


# from tensormonk.activations import Activations
# tensor_size = (2, 3, 10, 10)
# x = torch.rand(*tensor_size)
# test = Linear(tensor_size, 16, None, 0., True, (1, 4, 4))
# test(x).size()
# test.weight.shape
# test.bias.shape
# test
