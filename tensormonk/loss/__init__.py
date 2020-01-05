""" TensorMONK :: loss """

__all__ = ["CapsuleLoss", "Categorical", "DiceLoss",
           "MetricLoss", "MultiBoxLoss", "TripletLoss",
           "BalancedL1Loss", "IOULoss",
           "LabelLoss", "BoxesLoss", "PointLoss",
           "AdversarialLoss"]

from .categorical import Categorical
from .capsuleloss import CapsuleLoss
from .metricloss import TripletLoss, MetricLoss
from .multibox import MultiBoxLoss
from .segloss import DiceLoss
from .iou_loss import IOULoss
from .balanced_l1loss import BalancedL1Loss
from .label_loss import LabelLoss
from .boxes_loss import BoxesLoss
from .point_loss import PointLoss
from .adversarial_loss import AdversarialLoss

del (categorical, capsuleloss, metricloss, segloss, multibox, iou_loss,
     balanced_l1loss, label_loss, boxes_loss, point_loss)
