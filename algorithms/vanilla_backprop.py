import torch
from typing import Tuple

from .i_algorithm import IAlgorithm


class VanillaBackprop(IAlgorithm):
    """
    Vanilla implementation of regular supervised training of a classification task.
    """
    def prepare_task(self, task_id: int):
        """
        Nothing to prepare for.
        """
        pass

    def _compute_loss(self, predictions:torch.tensor, targets:torch.tensor) -> torch.tensor:
        """
        Calculate the loss.

        Args:
            predictions (torch.tensor): Network predictions
            targets (torch.tensor): True labels

        Returns:
            torch.tensor: loss
        """
        loss = self.loss_fn(predictions, targets).to(self.device)

        return loss

    def train_batch(self, predictions: torch.tensor, targets: torch.tensor, task_id: int) -> Tuple[torch.tensor, int]:
        """
        Supervised training with vanilla backpropagation for a single batch.

        Args:
            predictions (torch.tensor): Network predictions
            targets (torch.tensor): True labels
            task_id (int): Task ID

        Returns:
            torch.tensor: Batch loss
            int: Number of correct predictions
        """
        loss = self._compute_loss(predictions=predictions, targets=targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        num_correct_predictions = torch.sum(predictions.argmax(1) == targets).item()

        return loss.detach(), num_correct_predictions

    def test_batch(self, predictions: torch.tensor, targets: torch.tensor, task_id: int) -> Tuple[torch.tensor, int]:
        """
        Testing for a single batch.

        Args:
            predictions (torch.tensor): Network predictions
            targets (torch.tensor): True labels
            task_id (int): Task ID

        Returns:
            torch.tensor: Batch loss
            int: Number of correct predictions
        """
        loss = self._compute_loss(predictions=predictions, targets=targets)
        
        num_correct_predictions = torch.sum(predictions.argmax(1) == targets).item()
        
        return loss.detach(), num_correct_predictions
