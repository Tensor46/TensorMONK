""" TensorMONK :: loss """

__all__ = ["CapsuleLoss", "Categorical", "DiceLoss",
           "MetricLoss", "MultiBoxLoss", "TripletLoss"]

from .categorical import Categorical
from .capsuleloss import CapsuleLoss
from .metricloss import TripletLoss, MetricLoss
from .multibox import MultiBoxLoss
from .segloss import DiceLoss

del categorical, capsuleloss, metricloss, segloss, multibox
