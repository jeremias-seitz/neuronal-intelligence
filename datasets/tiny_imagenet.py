from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torch import FloatTensor, LongTensor, div
import pickle
from typing import List, Tuple

from .base import IDataset
from .sub_dataset import SubDataset


class ImageNetDataset(Dataset):
    """
    Dataset class for ImageNet
    Adjusted from: https://github.com/ehuynh1106/TinyImageNet-Transformers/blob/main/dataset.py
    """
    def __init__(self, dataset, labels, transform=None, normalize=None):
        super(ImageNetDataset, self).__init__()
        assert(len(dataset) == len(labels))
        self.dataset = dataset
        self.labels = labels
        self.transform = transform
        self.normalize = normalize

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        if self.transform:
            data = self.transform(data)

        data = div(data.type(FloatTensor), 255)
        if self.normalize:
            data = self.normalize(data)

        return data, self.labels[idx]


class TinyImageNet(IDataset):
    """
    Dataset class for TinyImageNet.
    Adjusted from: https://github.com/ehuynh1106/TinyImageNet-Transformers/blob/main/dataset.py
    """
    def __init__(self, num_tasks:int, num_classes_per_task:int, path: str, target_transform=None, image_size:int=32, task_permutation:List[int]=None):
        """
        Args:
            num_tasks (int): Number of tasks.
            num_classes_per_task (int): Number of classes per task.
            path (str): Path to the directory where the dataset is / should be stored
        """
        n_classes_total = num_tasks * num_classes_per_task
        if n_classes_total > 200 or n_classes_total < 1:
            raise ValueError(f"The number of classes was set to {n_classes_total}. For TinyImagenet, numbers between " \
                             f"1 and 200 are valid. \n The provided arguments were 'num_tasks'={num_tasks} and " \
                             f"'num_classes_per_task'={num_classes_per_task}.")

        if task_permutation is not None:
            task_order = task_permutation
        else:
            task_order = [task for task in range(num_tasks)]

        self.class_labels_per_task = [[i + task * num_classes_per_task for i in range(num_classes_per_task)] for task in task_order]
        self.target_transform = target_transform
        self._path = path
        self._image_size = image_size

    def get_data_loaders(self, batch_size: int, shuffle:bool=True, num_workers:int=4):

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        train_transforms = transforms.Compose([
            transforms.Resize(self._image_size, interpolation=InterpolationMode.BICUBIC),
            transforms.RandAugment(num_ops=2, magnitude=9),
        ])

        test_transforms = transforms.Compose([
            transforms.Resize(self._image_size, interpolation=InterpolationMode.BICUBIC),
        ])

        # Load training data
        with open(self._path + 'train_dataset.pkl', 'rb') as f_train:
            train_data, train_labels = pickle.load(f_train)

        train_dataset = ImageNetDataset(dataset=train_data,
                                        labels=train_labels.type(LongTensor),
                                        transform=train_transforms,
                                        normalize=transforms.Compose([normalize]))

        train_data_loaders = [
            DataLoader(SubDataset(original_dataset=train_dataset, sub_labels=class_labels), 
                       shuffle=shuffle,
                       batch_size=batch_size,
                       num_workers=num_workers,
                       pin_memory=True,
                       drop_last=True,) for class_labels in self.class_labels_per_task
        ]
        f_train.close()

        # Load validation data as test set
        with open(self._path + 'val_dataset.pkl', 'rb') as f_val:
            val_data, val_labels = pickle.load(f_val)

        val_dataset = ImageNetDataset(dataset=val_data,
                                      labels=val_labels.type(LongTensor),
                                      transform=test_transforms,
                                      normalize=transforms.Compose([normalize]))
                                      
        test_data_loaders = [
            DataLoader(SubDataset(original_dataset=val_dataset, sub_labels=class_labels), 
                       batch_size=batch_size, 
                       shuffle=shuffle, 
                       num_workers=num_workers,
                       pin_memory=True,
                       drop_last=True,) for class_labels in self.class_labels_per_task
        ]

        f_val.close()

        return train_data_loaders, test_data_loaders

    def get_joint_data_loaders(self, batch_size: int, shuffle:bool=True, num_workers:int=4) -> Tuple[List[Dataset], List[Dataset]]:
        """
        ToDo: implement
        """
