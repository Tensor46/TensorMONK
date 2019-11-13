""" TensorMONK :: loss """

__all__ = ["CapsuleLoss", "Categorical", "DiceLoss",
           "MetricLoss", "MultiBoxLoss", "TripletLoss",
           "BalancedL1Loss", "IOULoss"]

from .categorical import Categorical
from .capsuleloss import CapsuleLoss
from .metricloss import TripletLoss, MetricLoss
from .multibox import MultiBoxLoss
from .segloss import DiceLoss
from .iou_loss import IOULoss
from .balanced_l1loss import BalancedL1Loss

del categorical, capsuleloss, metricloss, segloss, multibox, iou_loss
