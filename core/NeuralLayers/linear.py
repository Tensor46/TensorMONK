""" TensorMONK's :: NeuralLayers :: Linear                                   """

__all__ = ["Linear", ]

import torch
import torch.nn as nn
import numpy as np
from .activations import Activations
# ============================================================================ #


class Linear(nn.Module):
    def __init__(self, tensor_size, out_features, activation="relu", dropout=0.,
                 normalization=None, pre_nm=False, weight_nm=False, bias=True, *args, **kwargs):
        super(Linear, self).__init__()
        # Checks
        assert type(tensor_size) in [list, tuple], "Linear -- tensor_size must be tuple or list"
        assert len(tensor_size) > 1, "Linear -- tensor_size must be of length > 1 (tensor_size[0] = BatchSize)"
        if len(tensor_size) > 2: # In case, last was a convolution or 2D input
            tensor_size = (tensor_size[0], int(np.prod(tensor_size[1:])))
        assert type(out_features) is int, "Linear -- out_features must be int"
        assert dropout >= 0. and dropout < 1., "Linear -- dropout must be in the range 0. - 1."
        assert normalization in [None, "batch"], "Linear's normalization must be None/'batch'"
        assert type(pre_nm) is bool, "Linear -- pre_nm must be boolean"
        assert type(weight_nm) is bool, "Linear -- weight_nm must be boolean"
        assert type(bias) is bool, "Linear -- bias must be boolean"
        activation = activation.lower()
        self.pre_nm = pre_nm
        # Modules
        if dropout > 0.:
            self.dropout = nn.Dropout(dropout)
        if pre_nm:
            if normalization == "batch":
                self.Normalization = nn.BatchNorm1d(tensor_size[1])
            act = Activations(activation)
            if act is not None:
                self.Activation = act

        if weight_nm:
            """ https://arxiv.org/pdf/1602.07868.pdf """
            self.Linear = nn.utils.weight_norm(nn.Linear(tensor_size[1] // (2 if activation in ("maxo", "rmxo") and pre_nm else 1),
                                    out_features // (2 if activation in ("maxo", "rmxo") and not pre_nm else 1), bias=bias), name='weight')
        else:
            self.Linear = nn.Linear(tensor_size[1] // (2 if activation in ("maxo", "rmxo") and pre_nm else 1),
                                    out_features * (2 if activation in ("maxo", "rmxo") and not pre_nm else 1), bias=bias)
            torch.nn.init.orthogonal_(self.Linear.weight)
        if not pre_nm:
            if normalization == "batch":
                self.Normalization = nn.BatchNorm1d(out_features*(2 if activation in ("maxo", "rmxo") and not pre_nm else 1))
            act = Activations(activation)
            if act is not None:
                self.Activation = act
        # out tensor size
        self.tensor_size = (tensor_size[0], out_features)

    def forward(self, tensor):
        if tensor.dim() > 2:
            tensor = tensor.view(tensor.size(0), -1)
        if hasattr(self, "dropout"):
            tensor = self.dropout(tensor)
        if self.pre_nm:
            if hasattr(self, "Normalization"):
                tensor = self.Normalization(tensor)
            if hasattr(self, "Activation"):
                tensor = self.Activation(tensor)
            tensor = self.Linear(tensor)
        else:
            tensor = self.Linear(tensor)
            if hasattr(self, "Normalization"):
                tensor = self.Normalization(tensor)
            if hasattr(self, "Activation"):
                tensor = self.Activation(tensor)
        return tensor


# from core.NeuralLayers import Activations
# tensor_size = (2, 3, 10, 10)
# x = torch.rand(*tensor_size)
# test = Linear(tensor_size, 16, "relu", 0., "batch", True, False, bias=False)
# test(x).size()
