""" TensorMONK :: Normalizations """

__all__ = ["Normalizations", "FrozenBatch2D",
           "NormAbsMaxDynamic", "NormAbsMax2d"]

from .normalizations import Normalizations, FrozenBatch2D
from .norm_absmax import NormAbsMaxDynamic, NormAbsMax2d
