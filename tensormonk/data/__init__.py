""" TensorMONK :: data """

__all__ = ["DataSets", "FewPerLabel", "FolderITTR"]

from .datasets import DataSets
from .fewperlabel import FewPerLabel
from .folderittr import FolderITTR

del datasets, fewperlabel, folderittr
