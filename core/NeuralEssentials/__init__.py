""" TensorMONK's :: NeuralEssentials                                         """

from .basemodel import BaseModel
from .savemodel import SaveModel
from .loadmodel import LoadModel
from .folderittr import FolderITTR
from .makemodel import MakeCNN, MakeAE, MakeModel
# make MakeModel more flexible and remove (MakeCNN & MakeAE)
from .mnist import MNIST
from .cifar10 import CIFAR10
from .visuals import MakeGIF, VisPlots
from .transforms import Transforms
from .fewperlabel import FewPerLabel

del basemodel
del savemodel
del loadmodel
del folderittr
del makemodel
del mnist
del cifar10
del visuals
del transforms
del fewperlabel
