import torch
from typing import Tuple

from .regularization_loss import RegularizationLoss


class SynapticIntelligence(RegularizationLoss):
    """
    Implementation of the Synaptic Intelligence algorithm presented in the paper:

    'Continual Learning Through Synaptic Intelligence'
    http://arxiv.org/abs/1703.04200

    The per-parameter importance is estimated as the contribution from changing that parameter to the decrease in loss.
    This importance is then used in a L2 parameter regularization scheme.
    """
    def prepare_initial_task(self):
        """
        Init replacement
        """
        self.params = {name: param for name, param in self.model.named_parameters() if param.requires_grad}

        self.params_prev_task = {name: param.clone().detach() for name, param in self.params.items()}
        self.importance = {name: torch.zeros_like(input=param, requires_grad=False, 
                                                  device=self.device) for name, param in self.params.items()}
        self.path_integrals = {name: torch.zeros_like(input=param, requires_grad=False, 
                                                      device=self.device) for name, param in self.params.items()}

        self.task_list = []
        self.damping = self.config.loss_entity.damping
        self.regularization_strength = self.config.loss_entity.regularization_strength

    def _evaluate_importance(self):
        """
        Evaluate the importance parameter by dividing the path integrals by the amount the parameters changed during
        training.
        """
        for name, importance in self.importance.items():
            
            delta_theta = torch.sub(self.params_prev_task[name], self.params[name].clone().detach())
            importance += self.path_integrals[name] / (delta_theta**2 + self.damping)

            self.path_integrals[name].zero_()

    def train_batch(self, predictions: torch.tensor, targets: torch.tensor, task_id: int) -> Tuple[torch.tensor, int]:
        """
        During the supervised training, the unregularized (!) gradient is used to compute a path integral, i.e. the 
        contribution from the change in parameter to the decrease in loss.

        Args:
            predictions (torch.tensor): Network predictions
            targets (torch.tensor): True labels
            task_id (int): Task ID

        Returns:
            torch.tensor: Batch loss
            int: Number of correct predictions
        """
        # Store pre parameter update parameter values
        params_before_update = {n: p.clone().detach() for n, p in self.params.items()}

        # Compute unregularized gradients
        loss_unregularized = self._compute_loss(predictions=predictions, targets=targets, use_regularization=False)
        self.optimizer.zero_grad()
        loss_unregularized.backward(retain_graph=True)
        unregularized_gradients = {n: p.grad.clone().detach() for n, p in self.params.items() if p.grad is not None}
        
        # Regular parameter update
        loss = self._compute_loss(predictions=predictions, targets=targets, use_regularization=True)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update the path integrals
        for n, p in unregularized_gradients.items():
            batch_delta = torch.sub(self.params[n].clone().detach(), params_before_update[n])
            task_delta = torch.sub(self.params[n].clone().detach(), self.params_prev_task[n])

            self.path_integrals[n] -= (p * batch_delta) / (task_delta**2 + self.damping)

        num_correct_predictions = torch.sum(predictions.argmax(1) == targets).item()

        return loss.detach(), num_correct_predictions
