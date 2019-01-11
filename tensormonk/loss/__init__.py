""" TensorMONK :: loss """

__all__ = ["CapsuleLoss", "Categorical", "TripletLoss", "DiceLoss"]

from .categorical import Categorical
from .other import CapsuleLoss, TripletLoss
from .segloss import DiceLoss

del categorical
del other
del segloss
