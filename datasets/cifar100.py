from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms, datasets
from typing import List

from .base import IDataset
from .sub_dataset import SubDataset


class Cifar100(IDataset):
    """
    Dataset class for CIFAR100.
    """
    def __init__(self, num_tasks:int, num_classes_per_task:int, path:str, target_transform=None, image_size:int=32, task_permutation:List[int]=None):
        """
        Args:
            num_tasks (int): Number of tasks
            num_classes_per_task (int): Number of classes per task
            path (str): Path to the directory where the dataset is / should be stored
            target_transform: Target transformation
            task_permutation (List[int]): Task order after shuffling
        """
        n_classes_total = num_tasks * num_classes_per_task
        if n_classes_total > 100 or n_classes_total < 1:
            raise ValueError(f"The number of classes was set to {n_classes_total}. For Cifar100, numbers between " \
                             f"1 and 100 are valid. \n The provided arguments were 'num_tasks'={num_tasks} and " \
                             f"'num_classes_per_task'={num_classes_per_task}.")

        if task_permutation is not None:
            task_order = task_permutation
        else:
            task_order = [task for task in range(num_tasks)]

        self.class_labels_per_task = [[i + task * num_classes_per_task for i in range(num_classes_per_task)] for task in task_order]
        self.target_transform = target_transform

        # Setup transforms
        normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                         std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
        
        train_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop((image_size, image_size), scale=(0.2, 1.0)),
            transforms.ToTensor(),
            normalize,
        ])
        test_transforms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ])

        # Define datasets
        self.training_data = datasets.CIFAR100(
            root=path,
            train=True,
            download=True,
            transform=train_transforms,
        )

        self.test_data = datasets.CIFAR100(
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