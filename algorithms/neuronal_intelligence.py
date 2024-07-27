import torch
from typing import Tuple

from .regularization_loss import RegularizationLoss


class NeuronalIntelligence(RegularizationLoss):
    """
    Extension to the Synaptic Intelligence algorithm (see class `SynapticIntelligence` in this module) from:

    'Continual Learning Through Synaptic Intelligence'
    http://arxiv.org/abs/1703.04200

    The per-parameter importance values are combined into per-neuron importance values to alleviate memory
    requirements. All parameters attributed to a neuron receive the same weight in the L2 parameter regularization
    scheme.

    The definition of a neuron can be somewhat arbitrary. While straightforward for a linear layer, a neuron in a
    convolutional architecture here is taken to be a single filter. What it should be for a transformer remains to be
    seen. The current implementation simply averages over all but the first dimension in every parameter group which
    works for linear and convolutional layers. 1D layers (1 parameter per parameter group per neuron) such as e.g.
    batchnorm are unaffected by the averaging and treated identical to Synaptic Intelligence.
    """
    def prepare_initial_task(self):
        """
        Inititialization that has to be done only once before the first task.

        Note that all parameters in the `self.params` variable will be regularized. If not all named parameter groups
        should be regularized, adjust the `self.params` variable.
        """
        self.params = {name: param for name, param in self.model.named_parameters() if param.requires_grad}
        self.params_prev_task = {name: param.clone().detach() for name, param in self.params.items()}


        self.importance = {name: torch.zeros(param.size(0), requires_grad=False, 
                                             device=self.device) for name, param in self.params.items()}
        self.task_importance = {name: torch.zeros(param.size(0), requires_grad=False, 
                                                  device=self.device) for name, param in self.params.items()}

        self.task_list = []
        self.damping = self.config.algorithm.damping
        self.regularization_strength = self.config.algorithm.regularization_strength

    def _evaluate_importance(self):
        """
        Accumulate the neuronal importance values at the end of a task.
        """
        for name, importance in self.importance.items():
            
            importance += self.task_importance[name] 
            self.task_importance[name].zero_()

    def train_batch(self, predictions: torch.tensor, targets: torch.tensor, task_id: int) -> Tuple[torch.tensor, int]:
        """
        Supervised training with backpropagation for a single batch. Unregularized gradients are collected over each
        neuron and used to compute batch-wise importance values. These are accumulated during the task and then added
        to the overall importance values in `self._evaluate_importance()`.

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

            # Compute instantaneous synaptic importance values ...
            synaptic_task_importance = (p * batch_delta) / (task_delta**2 + self.damping)

            # ... and perform the mean over a neuron
            mean_dimensions = (tuple(range(1, batch_delta.dim())))  
            if len(mean_dimensions) > 0:
                self.task_importance[n] -= synaptic_task_importance.mean(dim=mean_dimensions)
            else:
                self.task_importance[n] -= synaptic_task_importance

        num_correct_predictions = torch.sum(predictions.argmax(1) == targets).item()

        return loss.detach(), num_correct_predictions
