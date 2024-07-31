from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets
import numpy as np
from abc import abstractmethod
from typing import Tuple, List

from .base import IDataset
    

class MNIST(IDataset):
    """
    Dataset base class for MNIST.
    """
    @abstractmethod
    def __init__(self, path:str, **kwargs):
        self._path = path  # important to store the path variable when overriding this method

    @abstractmethod
    def get_transform(self, **kwargs):
        """
        Function that returns a transform that is applied to all inputs of a single task. Will be called once for every
        task.
        """

    def get_data_loaders(self, batch_size: int, shuffle:bool=True, num_workers:int=4) -> Tuple[List, List]:

        train_datasets = []
        test_datasets = []
        np.random.seed(self._seed)

        for _ in range(self._num_tasks):

            transform = self.get_transform()  # apply the same transform to train and test set

            train_datasets.append(datasets.MNIST(root=self._path,
                                                 train=True,
                                                 download=True,
                                                 transform=transform,
                                                 target_transform=self._target_transform))

            test_datasets.append(datasets.MNIST(root=self._path,
                                                train=False,
                                                download=True,
                                                transform=transform,
                                                target_transform=self._target_transform))
            
        train_data_loaders = [
            DataLoader(dataset=train_datasets[i],
                       batch_size=batch_size, 
                       shuffle=shuffle, 
                       num_workers=num_workers)
                       for i in self._task_order
        ]
        
        test_data_loaders = [
            DataLoader(dataset=test_datasets[i],
                       batch_size=batch_size, 
                       shuffle=shuffle, 
                       num_workers=num_workers)
                       for i in self._task_order
        ]

        return train_data_loaders, test_data_loaders

    def get_joint_data_loaders(self, batch_size: int, shuffle:bool=True, num_workers:int=4) -> Tuple[List, List]:

        train_datasets = []
        test_datasets = []

        train_data_loaders = []
        np.random.seed(self._seed)

        for _ in range(self._num_tasks):

            transform = self.get_transform()  # apply the same transform to train and test set

            train_datasets.append(datasets.MNIST(root=self._path,
                                                 train=True,
                                                 download=True,
                                                 transform=transform,
                                                 target_transform=self._target_transform))
                                           
            test_datasets.append(datasets.MNIST(root=self._path,
                                                train=False,
                                                download=True,
                                                transform=transform,
                                                target_transform=self._target_transform))

        train_data_loaders = [
            DataLoader(dataset=ConcatDataset(train_datasets),
                       batch_size=batch_size, 
                       shuffle=shuffle, 
                       num_workers=num_workers)
        ]
        
        test_data_loaders = [
            DataLoader(dataset=test_datasets[i],
                       batch_size=batch_size, 
                       shuffle=shuffle, 
                       num_workers=num_workers)
                       for i in self._task_order
        ]
            
        return train_data_loaders, test_data_loaders
