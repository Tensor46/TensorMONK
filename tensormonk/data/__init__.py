""" TensorMONK :: data """

__all__ = ["DataSets", "FewPerLabel", "FolderITTR",
           "Flip", "ElasticSimilarity",
           "RandomBlur", "RandomNoise", "RandomTransforms"]

from .datasets import DataSets
from .fewperlabel import FewPerLabel
from .folderittr import FolderITTR
from .transforms import Flip, ElasticSimilarity, RandomBlur, \
    RandomNoise, RandomTransforms

del datasets, fewperlabel, folderittr, transforms
