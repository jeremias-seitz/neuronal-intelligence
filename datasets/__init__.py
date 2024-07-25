from .base import IDataset
from .cifar10 import Cifar10
from .cifar100 import Cifar100
from .mnist_base import MNIST
from .mnist_variants import PermutedMNIST, RotatedMNIST, SplitMNIST
from .target_transforms import TaskShuffledLabels
from .tiny_imagenet import TinyImageNet