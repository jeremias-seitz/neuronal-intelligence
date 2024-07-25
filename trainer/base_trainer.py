import torch
from typing import Tuple, List
from tqdm import tqdm
from omegaconf import dictconfig

from callbacks import Callback
from network import NetworkPropagator
from loss_entities import ILossEntity
from utils import get_device_from_config


class BaseTrainer():
    """
    Implementation of a trainer class for supervised learning. This is the main class the user interfaces with. 
    """
    def __init__(self, 
                 model: torch.nn.Module, 
                 loss_entity: ILossEntity, 
                 callbacks: List[Callback],
                 configuration: dictconfig.DictConfig,
                 network_propagator: NetworkPropagator,
                 ) -> None:
        """
        Args:
            model (torch.nn.Module): Model used for training and testing
            loss_entity (ILossEntity): Computes the loss and implements the specific train and test functionalities
            callbacks (list): List of callbacks that will be called during the training and validation process at specific
                time points in the process.
            configuration (dictconfig.DictConfig): Hydra configuration file
            network_propagator (NetworkPropagator): Object taking care of forwarding data through the network
                according to the learning scenario (e.g. single vs multi-headed output).
        """
        self._model = model.to(get_device_from_config(config=configuration))
        self._loss_entity = loss_entity
        self._network_propagator = network_propagator

        self._callbacks = callbacks
        self._dispatch_callbacks('on_init', trainer=self, config=configuration)

    def prepare_task(self, task_id: int):
        """
        Prepare all components for a new task.

        Args:
            task_id (int): Task ID
        """
        self._loss_entity.prepare_task(task_id=task_id)
        self._network_propagator.prepare_task(task_id=task_id)

        self._dispatch_callbacks('on_new_task', task_id=task_id)

    def train_task(self, task_id: int, data_loader: torch.utils.data.DataLoader, 
                   enable_logging:bool=True) -> Tuple[float, float]:
        """
        Supervised training of a task.

        Args:
            task_id (int): Task ID
            data_loader (torch.utils.data.DataLoader): Data loader for the training data
            enable_logging (bool): Flag to enable/disable logging (passed to the callback functions).

        Returns:
            float: Accuracy [%]
            float: Training loss
        """
        # Preparation
        self._model.train()
        self._dispatch_callbacks('on_training_start', task_id=task_id, enable_logging=enable_logging)

        # Training
        train_loss, num_correct_predictions_total = 0, 0
        for (data, labels) in tqdm(data_loader):
            predictions, labels = self._network_propagator.get_predictions_and_labels(inputs=data, 
                                                                                      labels=labels, 
                                                                                      task_id=task_id)
            loss, num_correct_predictions = self._loss_entity.train_batch(predictions=predictions,
                                                                          targets=labels,
                                                                          task_id=task_id)
            train_loss += loss
            num_correct_predictions_total += num_correct_predictions

            self._dispatch_callbacks('on_batch_end', task_id=task_id)

        # Statistics
        accuracy = 100 * num_correct_predictions_total / len(data_loader.dataset)
        print(f"[{task_id+1}] Train -> Accuracy: {accuracy:>0.1f}%, Avg loss: {train_loss/len(data_loader):>8f}")

        self._dispatch_callbacks('on_training_end', task_id=task_id, enable_logging=enable_logging)

        return accuracy, train_loss

    def test_task(self, task_id: int, data_loader: torch.utils.data.DataLoader, 
                  enable_logging:bool=True) -> Tuple[float, float]:
        """
        Test a task.

        Args:
            task_id (int): Task ID
            data_loader (torch.utils.data.DataLoader): Data loader for the testing data
            enable_logging (bool): Flag to enable/disable logging (passed to the callback functions).

        Returns:
            float: Accuracy [%]
            float: Testing loss
        """
        # Preparation
        self._model.eval()

        self._dispatch_callbacks('on_validation_start', task_id=task_id, enable_logging=enable_logging)

        # Testing
        test_loss, num_correct_predictions_total = 0, 0
        with torch.no_grad():
            for (data, labels) in data_loader:
                predictions, labels = self._network_propagator.get_predictions_and_labels(inputs=data, 
                                                                                          labels=labels, 
                                                                                          task_id=task_id)
                loss, num_correct_predictions = self._loss_entity.test_batch(predictions=predictions,
                                                                             targets=labels,
                                                                             task_id=task_id)
                test_loss += loss
                num_correct_predictions_total += num_correct_predictions

        # Statistics
        accuracy = 100 * num_correct_predictions_total / len(data_loader.dataset)
        print(f"[{task_id+1}] Test  -> Accuracy: {accuracy:>0.1f}%, Avg loss: {test_loss / len(data_loader):>8f}")

        self._dispatch_callbacks('on_validation_end', task_id=task_id, accuracy=accuracy, loss=test_loss.item(), enable_logging=enable_logging)

        return accuracy, test_loss

    def _dispatch_callbacks(self, hook_name: str, **kwargs):
        """
        Calls the hooks for all stored callback functions.

        Args:
            hook_name (str): Name of the hook to the specific function to be called.
            **kwargs (any): Arbitrary keyword arguments passed to all callbacks
        """
        for callback in self._callbacks:
            getattr(callback, hook_name)(**kwargs)

    def _print_task_info(self, task_id: int, data_loader: torch.utils.data.DataLoader):
        """
        Print information about a task for debugging.

        Args:
            task_id (int): ID of the task (only for printing purposes)
            data_loader (torch.utils.data.DataLoader): Data loader containing the data to be analysed
        """
        n_samples = 0
        min_label = 1e10
        max_label = 0
        for data, labels in data_loader:
            # This assumes contiguous, positive and numerical labels
            min_label = min(torch.min(labels), min_label)
            max_label = max(torch.max(labels), max_label)
            n_samples += len(data)

        print(f"[{task_id+1}] num batches: {len(data_loader)}")
        print(f"[{task_id+1}] num samples: {n_samples}")
        print(f"[{task_id+1}] labels: {min_label}-{max_label}")
        