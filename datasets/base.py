from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from typing import List, Tuple


class IDataset(ABC):
    """
    Interface class for datasets. There are two methods that return two lists of dataloaders, one for the training and
    one for the testing set. One method is designed for the sequential training of multiple tasks, the other for joint
    training while the testing is still done task-wise. Joint-training is a control to obtain an upper performance 
    bound since most neural networks prefer training everything at the same time.
    """
    name = ""
    
    @abstractmethod
    def get_data_loaders(self, batch_size: int, shuffle:bool=True, num_workers: int=4) -> Tuple[List[Dataset], List[Dataset]]:
        """
        Returns the train and the test data as two separate lists of training and testing data loaders respectively.

        Args:
            batch_size (int): Batch size
            shuffle (bool): Enable/disable shuffling of data samples to obtain random batches
            num_workers (int): Number of workers that retrieve data

        Returns:
            List[Dataset]: List of data loader objects for training
            List[Dataset]: List of data loader objects for testing
        """

    @abstractmethod
    def get_joint_data_loaders(self, batch_size: int, shuffle:bool=True, num_workers:int=4) -> Tuple[List[Dataset], List[Dataset]]:
        """
        Returns train and test data in two separate lists of dataloaders. There will only be a single 
        dataloader containing all the training data.

        Args:
            batch_size (int): Batch size
            shuffle (bool): Enable/disable shuffling of data samples to obtain random batches
            num_workers (int): Number of workers that retrieve data

        Returns:
            List[Dataset]: List containing a single data loader object containing all the data for training
            List[Dataset]: List of data loader objects for testing
        """
        