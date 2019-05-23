""" TensorMONK :: loss """

__all__ = ["CapsuleLoss", "Categorical", "TripletLoss", "DiceLoss",
           "MultiBoxLoss"]

from .categorical import Categorical
from .capsuleloss import CapsuleLoss
from .metricloss import TripletLoss
from .multibox import MultiBoxLoss
from .segloss import DiceLoss

del categorical, capsuleloss, metricloss, segloss, multibox
