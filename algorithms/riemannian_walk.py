import torch
from typing import Tuple

from .regularization_loss import RegularizationLoss


class RiemannianWalk(RegularizationLoss):
    """
    Implementation of the Riemannian Walk algorithm presented in the paper (without the memory component):

    'Riemannian Walk for Incremental Learning: Understanding Forgetting and Intransigence'
    http://arxiv.org/abs/1801.10112

    An efficient version of Elastic Weight Consolidation is combined Synaptic Intelligence to obtain a
    regularization-based continual learning algorithm. Note that the memory component that would also be part of the
    Riemannian Walk has been omitted in this implementation.
    """
    def prepare_initial_task(self):
        """
        Init replacement. Prepare required variables and read configuration.
        """
        self.params = {name: param for name, param in self.model.named_parameters() if param.requires_grad}

        self.params_prev_task = {name: param.clone().detach() for name, param in self.params.items()}
        self.s = {name: torch.zeros_like(input=param, requires_grad=False, 
                                                  device=self.device) for name, param in self.params.items()}
        self.s_running = {name: torch.zeros_like(input=param, requires_grad=False, 
                                                  device=self.device) for name, param in self.params.items()}
        self.f = {name: torch.zeros_like(input=param, requires_grad=False, 
                                                  device=self.device) for name, param in self.params.items()}
        self.f_running = {name: torch.zeros_like(input=param, requires_grad=False, 
                                                  device=self.device) for name, param in self.params.items()}

        self.task_list = []
        self.damping = self.config.loss_entity.damping
        self.alpha = self.config.loss_entity.alpha
        self.regularization_strength = self.config.loss_entity.regularization_strength

    def _evaluate_importance(self):
        """
        Evaluate the importance parameters
        """
        for name in self.params.keys():
            
            self.f[name] = self.f_running[name].clone()
            self.s[name] = 0.5*self.s_running[name].clone()
            self.s_running[name] = self.s[name].clone()

    def train_batch(self, predictions: torch.tensor, targets: torch.tensor, task_id: int) -> Tuple[torch.tensor, int]:
        """
        Supervised training with vanilla backpropagation for a single batch. A running estimation of the diagonal of
        the Fischer Information Matrix is performed.

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
        loss = self._compute_loss(predictions=predictions, targets=targets, use_regularization=True)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        gradients = {n: p.grad.clone().detach() for n, p in self.params.items() if p.grad is not None}

        # Update the path integrals
        for n, p in gradients.items():
            fisher = p.data ** 2
            self.f_running[n] = self.alpha*fisher + (1-self.alpha)*self.f_running[n]

            batch_delta = torch.sub(self.params[n].clone().detach(), params_before_update[n])
            path_integral = -p*batch_delta
            fisher_distance = 0.5*(self.f_running[n]*batch_delta**2)
            self.s_running[n] += path_integral/(fisher_distance + self.damping)

        num_correct_predictions = torch.sum(predictions.argmax(1) == targets).item()

        return loss.detach(), num_correct_predictions

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
                importance = self.f[name] + self.s[name]
                
                regularization_loss += ( importance * (delta_theta_current_task ** 2) ).sum()

            loss += self.regularization_strength * regularization_loss

        return loss
    