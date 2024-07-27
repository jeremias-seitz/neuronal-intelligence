import torch
from typing import Tuple
from abc import abstractmethod

from .i_loss_entity import IAlgorithm



class RegularizationLoss(IAlgorithm):
    """
    Prototype class for implementing regularization-based continual learning algorithms. 
    """
    def prepare_task(self, task_id: int):
        """
        Increases the internal task counter and evaluates the parameter importance.
        """
        if self.is_initial_task:
            self.prepare_initial_task()
            self.task_list = []
            self.is_initial_task = False
        else:
            self._evaluate_importance()
        
        if task_id not in self.task_list:
            self.task_list.append(task_id)
        
        for param_name, param in self.params.items():
            self.params_prev_task[param_name] = param.clone().detach().to(self.device)

    @abstractmethod
    def prepare_initial_task(self):
        """
        Init replacement
        """

    @abstractmethod
    def _evaluate_importance(self):
        """
        Evaluate the importance parameter
        """

    def _compute_loss(self, predictions:torch.tensor, targets:torch.tensor, use_regularization:bool=True) -> torch.tensor:
        """
        Calculate the loss with or without the regularization term. This method works for both per-parameter as well as
        per-neuron importance values.

        Args:
            predictions (torch.tensor): Network predictions
            targets (torch.tensor): True labels
            is_regularization (bool): Enable/disable regularization loss

        Returns:
            torch.tensor: loss
        """
        loss = self.loss_fn(predictions, targets).to(self.device)

        if use_regularization:

            regularization_loss = 0

            for name, param in self.params.items():

                delta_theta_current_task = torch.sub(self.params_prev_task[name], param)
                importance = self.importance[name].clone()

                # Add singleton dimensions to enable multiplication in case dimensions don't match.
                # This will be the case when importances are per-neuron parameters.
                while importance.dim() < delta_theta_current_task.dim():
                    importance.unsqueeze_(dim=1)
                
                regularization_loss += ( importance * (delta_theta_current_task ** 2) ).sum()

            loss += self.regularization_strength * regularization_loss

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
        loss = self._compute_loss(predictions=predictions, targets=targets, use_regularization=True)
        
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
        loss = self._compute_loss(predictions=predictions, targets=targets, use_regularization=True)
        
        num_correct_predictions = torch.sum(predictions.argmax(1) == targets).item()
        
        return loss.detach(), num_correct_predictions
    