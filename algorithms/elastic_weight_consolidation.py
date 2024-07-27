import torch
from typing import Tuple

from .regularization_loss import RegularizationLoss


class ElasticWeightConsolidation(RegularizationLoss):
    """
    Implementation of the Elastic Weight Consolidation algorithm presented in the paper:

    'Overcoming catastrophic forgetting in neural networks'
    http://arxiv.org/abs/1612.00796

    The Fisher Information Matrix can be seen as the covariance matrix of the expected loss-gradient. Accordingly, its
    diagonal can be used as an estimate for a parameter importance metric. This algorithm implements an estimation of
    that diagonal and uses it as a weight in a L2 parameter regularization scheme. Note that here a simplified
    estimation is used that computes the diagonal of the Empirical Fischer Matrix.
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

    def prepare_initial_task(self):
        """
        Init replacement
        """
        self.params = {name: param for name, param in self.model.named_parameters() if param.requires_grad}
        self.params_prev_task = {name: param.clone().detach() for name, param in self.params.items()}
        self.importance = {name: torch.zeros_like(input=param, requires_grad=False, 
                                                  device=self.device) for name, param in self.params.items()}
        self.task_list = []
        self.regularization_strength = self.config.loss_entity.regularization_strength

    def _evaluate_importance(self):
        """
        Evaluate the importance parameter as the diagonal of the Empirical Fischer Matrix.
        """
        self.model.train()
        task_id = self.task_list[-1]

        for (data, labels) in self.data_loaders[task_id]:
            predictions, labels = self.network_propagator.get_predictions_and_labels(inputs=data, 
                                                                                     labels=labels, 
                                                                                     task_id=task_id)
            
            loss = self._compute_loss(predictions=predictions, targets=labels, use_regularization=False)
            self.optimizer.zero_grad()
            loss.backward()

            # Compute the importance as the diagonal of the Empirical Fisher Matrix
            for name, importance in self.importance.items():
                importance += (self.params[name].grad ** 2) * len(data) / len(self.data_loaders[task_id])

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
