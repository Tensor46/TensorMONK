""" TensorMONK's :: NeuralLayers :: Linear                                  """

__all__ = ["Linear", ]

import torch
import torch.nn as nn
import numpy as np
from .activations import Activations


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
        self.t_size = tuple(tensor_size)
        # Checks
        if not type(tensor_size) in [int, list, tuple]:
            raise TypeError("Linear: tensor_size must tuple/list")

        if isinstance(tensor_size, int):
            in_features = tensor_size
        else:
            assert len(tensor_size) >= 2, \
                "Linear: when tuple/list, tensor_size must of length 2 or more"
            in_features = np.prod(tensor_size[1:])

        if not isinstance(out_features, int):
            raise TypeError("Linear:out_features must be int")

        if not isinstance(dropout, float):
            raise TypeError("Linear: dropout must be float")
        if 1. > dropout > 0.:
            self.dropout = nn.Dropout2d(dropout)

        if isinstance(activation, str):
            activation = activation.lower()
        assert activation in [None, "", ] + Activations.available(),\
            "Linear: activation must be None/''/" + \
            "/".join(Activations.available())
        self.act = activation

        if not isinstance(bias, bool):
            raise TypeError("Linear: bias must be bool")

        if out_shape is not None:
            assert np.prod(out_shape) == out_features, \
                "Linear: np.prod(out_shape) != out_features"
            self.out_shape = out_shape

        multiplier = 2 if activation in ("maxo", "rmxo") else 1
        # get weight and bias
        self.weight = nn.Parameter(torch.rand(out_features*multiplier,
                                              in_features))
        self.weight.data.add_(- 0.5)
        self.weight.data.div_(self.weight.data.pow(2).sum(1, True).pow(0.5))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features*multiplier))
        # get activation function
        if activation is not None:
            if activation in Activations.available():
                self.activation = Activations(activation)
        # out tensor size
        self.tensor_size = (1, out_features)
        if hasattr(self, "out_shape"):
            self.tensor_size = tuple([1, ] + list(out_shape))

    def forward(self, tensor):
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
        msg = "x".join(["_"]+[str(x)for x in self.t_size[1:]]) + " -> "
        if hasattr(self, "dropout"):
            msg += "dropout -> "
        msg += "{}({})".format("linear", "x".join([str(x) for x in
                                                   self.weight.shape]))+" -> "
        if hasattr(self, "activation"):
            msg += self.act + " -> "
        msg += "x".join(["_"]+[str(x)for x in self.tensor_size[1:]])
        return msg

# from core.NeuralLayers import Activations
# tensor_size = (2, 3, 10, 10)
# x = torch.rand(*tensor_size)
# test = Linear(tensor_size, 16, "", 0., True, (1, 4, 4))
# test(x).size()
# test.weight.shape
# test.bias.shape
