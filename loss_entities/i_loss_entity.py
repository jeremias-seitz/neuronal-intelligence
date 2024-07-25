import torch
from typing import Tuple, List
from omegaconf import dictconfig
from abc import ABC, abstractmethod

from utils import get_device_from_config
from network import NetworkPropagator


class ILossEntity(ABC):
    """
    Interface class for supervised training algorithms. The proposed call signatures should be followed unless changes
    have been made to the corresponding trainer class in the 'trainer' module.

    This class plans for the following functions to be implemented:
     - 'prepare_task': Computations in preparation for a new task
     - '_compute_loss': Loss computation, based on the provided loss function, that can be modified to add e.g. a 
        regularization term.
     - 'train_batch': Training method for a single batch. This is where algorithm specific functionality such as e.g.
        running estimate of the Fischer diagonal can be implemented.
     - 'test_batch': Testing method for a single batch.
    """
    def __init__(self,
                 model: torch.nn.Module, 
                 loss_function: torch.nn.Module, 
                 optimizer: torch.optim,
                 configuration: dictconfig.DictConfig,
                 network_propagator: NetworkPropagator,
                 data_loaders: List[torch.utils.data.DataLoader],
                 **kwargs) -> None:
        """
        Args:
            model (torch.nn.Module): Model used for training.
            loss_function (torch.nn.Module): Loss function
            optimizer (torch.optim.optimizer.Optimizer): Optimizer
            configuration (dictconfig.DictConfig): Hydra config file
        """
        self.model = model
        self.loss_fn = loss_function
        self.optimizer = optimizer
        self.config = configuration
        self.device = get_device_from_config(configuration=configuration)
        self.data_loaders = data_loaders
        self.network_propagator = network_propagator
        self.is_initial_task = True
    
    @abstractmethod
    def prepare_task(self, task_id: int):
        """
        Perform computations and prepare for a new task
        """

    @abstractmethod
    def _compute_loss(self, predictions:torch.tensor, targets:torch.tensor) -> torch.tensor:
        """
        Calculate the loss with potentially added functionality such as e.g. a regularization term.

        Args:
            predictions (torch.tensor): Network predictions
            targets (torch.tensor): True labels

        Returns:
            torch.tensor: loss
        """

    @abstractmethod
    def train_batch(self, predictions: torch.tensor, targets: torch.tensor, task_id: int) -> Tuple[torch.tensor, int]:
        """
        Supervised training of a single batch adjusted to the implemented algorithm.

        Args:
            predictions (torch.tensor): Network predictions
            targets (torch.tensor): True labels
            task_id (int): Task ID

        Returns:
            torch.tensor: Batch loss
            int: Number of correct predictions
        """

    @abstractmethod
    def test_batch(self, predictions: torch.tensor, targets: torch.tensor, task_id: int) -> Tuple[torch.tensor, int]:
        """
        Testing of a single batch adjusted to the implemented algorithm.

        Args:
            predictions (torch.tensor): Network predictions
            targets (torch.tensor): True labels
            task_id (int): Task ID

        Returns:
            torch.tensor: Batch loss
            int: Number of correct predictions
        """
