import torch
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms, datasets
import numpy as np
from typing import List, Union
from PIL.Image import Image

from .mnist_base import MNIST
from .sub_dataset import SubDataset
from .base import IDataset
    

class PermutedMNIST(MNIST):
    """
    Dataset class for permuted MNIST. A pixel-wise permutation is applied to the MNIST images. Within each task, the 
    permutation is the same for all images but different to the permutations from other tasks. 
    """
    def __init__(self, num_tasks:int, path:str, seed:int=53423):
        """
        Args:
            num_tasks (int): Number of tasks.
            path (str): Path to the directory where the dataset is / should be stored
            seed (int): Seed for the random number generator.
        """
        self._num_tasks = num_tasks
        self._path = path
        self._seed = seed
        np.random.seed(seed=seed)

    def get_transform(self):
        """
        Creates the permutation transform. This assumes that the images are not rescaled and are of dimension 28x28
        pixels.
        Due to an unknown reason, the creation of the transform needs to be wrapped into a function (this). Otherwise
        the transforms that are created are random but identical for all tasks. With this function, the returned
        transforms are and most importantly stay different from each other.

        Returns:
            torchvision.transforms.Compose: Permutation transform for fixed size images 28x28 pixels
        """
        permutation = torch.from_numpy(np.random.permutation(784))  # 28x28=784

        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Lambda(lambda x: x.view(-1)[permutation].view(1, 28, 28))])

        return transform
    

class RotatedMNIST(MNIST):
    """
    Dataset class for rotated MNIST. A rotation is applied to the MNIST images. Within each task, the rotation is the
    same for all images but different to the rotations from other tasks. 
    """
    def __init__(self, num_tasks:int, path:str, seed:int=53423):
        """
        Args:
            num_tasks (int): Number of tasks.
            path (str): Path to the directory where the dataset is / should be stored
            seed (int): Seed for the random number generator.
        """
        self._num_tasks = num_tasks
        self._path = path
        self._seed = seed
        np.random.seed(seed=seed)

    def get_transform(self):
        """
        Creates the permutation transform.
        Due to some strange reason, we need to wrap the transform creation into a function. Otherwise the transform
        would be identical, and always the latest permutation would be applied.

        Returns:
            torchvision.transforms.Compose: Rotation transform
        """
        angle = np.random.randint(low=0, high=359)
        rotation_transform = MyRotateTransform(angle=angle) 

        transform = transforms.Compose([
                                        rotation_transform,
                                        transforms.ToTensor(),
                                        ])
        return transform
    

class MyRotateTransform:
    """
    Rotation transform that rotates an image by a given angle.
    """
    def __init__(self, angle: int):
        """
        Args:
            angle (int): Rotation angle
        """
        self.angle = angle

    def __call__(self, x: Union[Image, torch.Tensor]):
        """
        Args:
            x (PIL.Image.Image or torch.Tensor): Input image
        Returns:
            PIL.Image.Image or torch.Tensor: Rotated image
        """
        return transforms.functional.rotate(img=x, angle=self.angle, interpolation=transforms.InterpolationMode.BICUBIC)
    

class SplitMNIST(IDataset):
    """
    Dataset class for split MNIST. Splits MNIST into smaller subsets only containing specific digits for different
    tasks.
    """
    def __init__(self, num_tasks:int, num_classes_per_task:int, path:str, target_transform=None, task_permutation:List[int]=None):
        """
        Args:
            num_tasks (int): Number of tasks.
            seed (int): Seed for the random number generator.
        """
        n_classes_total = num_tasks * num_classes_per_task
        if n_classes_total > 10 or n_classes_total < 1:
            raise ValueError(f"The number of classes was set to {n_classes_total}. For MNIST, numbers between " \
                             f"1 and 10 are valid. \n The provided arguments were 'num_tasks'={num_tasks} and " \
                             f"'num_classes_per_task'={num_classes_per_task}.")

        if task_permutation is not None:
            task_order = task_permutation
        else:
            task_order = [task for task in range(num_tasks)]

        self.class_labels_per_task = [[i + task * num_classes_per_task for i in range(num_classes_per_task)] for task in task_order]
        self.target_transform = target_transform

        # Setup transforms
        normalize = transforms.Normalize(mean=[0.1307], std=[0.3081])
        
        train_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        # Define datasets
        self.training_data = datasets.MNIST(
            root=path,
            train=True,
            download=True,
            transform=train_transforms,
        )

        self.test_data = datasets.MNIST(
            root=path,
            train=False,
            download=True,
            transform=test_transforms,
        )

    def get_data_loaders(self, batch_size: int, shuffle:bool=True, num_workers:int=4):

        train_data_loaders = [
            DataLoader(SubDataset(original_dataset=self.training_data, 
                                  sub_labels=class_labels, 
                                  target_transform=self.target_transform), 
                       batch_size=batch_size, 
                       shuffle=shuffle, 
                       num_workers=num_workers,
                       ) for class_labels in self.class_labels_per_task
        ]

        test_data_loaders = [
            DataLoader(SubDataset(original_dataset=self.test_data, 
                                  sub_labels=class_labels, 
                                  target_transform=self.target_transform), 
                       batch_size=batch_size, 
                       shuffle=shuffle, 
                       num_workers=num_workers,
                       ) for class_labels in self.class_labels_per_task
        ]

        return train_data_loaders, test_data_loaders

    def get_joint_data_loaders(self, batch_size: int, shuffle:bool=True, num_workers:int=4):

        train_datasets = [SubDataset(original_dataset=self.training_data, 
                                     sub_labels=class_labels, 
                                     target_transform=self.target_transform) for class_labels in self.class_labels_per_task]

        train_data_loaders = [
            DataLoader(dataset=ConcatDataset(train_datasets), 
                       batch_size=batch_size, 
                       shuffle=shuffle, 
                       num_workers=num_workers)]

        test_data_loaders = [
            DataLoader(SubDataset(original_dataset=self.test_data, 
                                  sub_labels=class_labels, 
                                  target_transform=self.target_transform), 
                       batch_size=batch_size, 
                       shuffle=shuffle, 
                       num_workers=num_workers,
                       ) for class_labels in self.class_labels_per_task
        ]

        return train_data_loaders, test_data_loaders
