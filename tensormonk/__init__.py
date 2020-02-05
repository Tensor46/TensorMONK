""" TensorMONK """

__all__ = ["activations", "architectures", "layers", "loss",
           "normalizations", "regularizations", "essentials",
           "data", "plots", "utils", "thirdparty",
           "detection"]

from . import detection
from . import activations, architectures, layers,  loss
from . import normalizations, regularizations, essentials
from . import data, plots, utils, thirdparty

__version__ = "0.5.0"
