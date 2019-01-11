""" TensorMONK :: loss """

__all__ = ["CapsuleLoss", "Categorical", "TripletLoss", "DiceLoss"]

from .categorical import Categorical
from .capsuleloss import CapsuleLoss
from .metricloss import TripletLoss
from .segloss import DiceLoss

del categorical, capsuleloss, metricloss, segloss
