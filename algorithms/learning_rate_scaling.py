import torch
from typing import Tuple

from .i_algorithm import IAlgorithm


class NeuronalLearningRateScaling(IAlgorithm):
    """
    Based on Neuronal Intelligence (see class in this module), the Neuronal Learning Rate Scaling algorithm ranks the
    neurons depending on their importance separately for each layer of the network. The updates of all parameters in
    each neuron are then scaled as a function of the relative importance of a neuron within the layer. Important
    neurons remain unchanged, unimportant ones remain plastic.

    For an optimizer only implementation of the above algorithm, see the Continual Adam and Continual SGD classes in
    the optimizer module.

    Note that this implementation was only tested successfully with the SGD optimizer. Using e.g. Adam introduces
    additional scaling that might counteract the intended scaling. This was fixed in the optimizer only approach. 
    """
    def prepare_task(self, task_id: int):
        """
        Preparation before a new task is started.

        Args:
            task_id (int): Index of the task to be prepared
        """
        if self.is_initial_task:
            self._initial_preparation()
            self.task_list = []
            self.is_initial_task = False
        else:
            self._evaluate_importance_and_availability()
        
        if task_id not in self.task_list:
            self.task_list.append(task_id)
        
    def _initial_preparation(self):
        """
        Inititialization that has to be done only once before the first task.
        """
        self.params = {name: param for name, param in self.model.named_parameters() if param.requires_grad}

        self.importance = {name: torch.zeros(param.size(0), requires_grad=False, 
                                             device=self.device) for name, param in self.params.items()}
        
        self.task_importance = {name: torch.zeros(param.size(0), requires_grad=False, 
                                                  device=self.device) for name, param in self.params.items()}

        self.neuronal_availability = {name: torch.ones(param.size(0), requires_grad=False, 
                                             device=self.device) for name, param in self.params.items()}
        
        self.steepness_factor = self.config.loss_entity.steepness_factor
        self.offset_factor = self.config.loss_entity.offset_factor
        self.lr = self.config.loss_entity.learning_rate

    def _evaluate_importance_and_availability(self) -> None:
        """
        Evaluate the importance parameter and compute the availability. The availability of a parameter itself depends
        on its importance and scales the parameter update. It takes values between 0 and 1.
        """
        for name, importance in self.importance.items():
            
            importance += self.task_importance[name] 

            self.neuronal_availability[name] = self._compute_availability(importance)

            self.task_importance[name].zero_()

    def _compute_availability(self, importance_vector: torch.tensor) -> torch.tensor:
        """
        The availability (in [0,1]) is used as a neuronal learning rate scaling, i.e. it scales the update of
        all parameters in a neuron.

        Args:
            importance_vector (torch.tensor): Vector containing neuronal importance values

        Returns:
            torch.tensor: Vector containing the neuronal availability values
        """
        mean, std = torch.mean(importance_vector), torch.std(importance_vector)
        offset = mean - std
        steepness = 5/std
        exp = torch.exp(-(importance_vector - self.offset_factor*offset)*self.steepness_factor*steepness)
        output = exp/(1+exp)
        return output.requires_grad_(False)

    def train_batch(self, predictions: torch.tensor, targets: torch.tensor, task_id: int) -> Tuple[torch.tensor, int]:
        """
        Supervised training with backpropagation for a single batch.

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

        # Compute gradients
        loss = self._compute_loss(predictions=predictions, targets=targets)
        self.optimizer.zero_grad()
        loss.backward()
        unscaled_gradients = {n: p.grad.clone().detach() for n, p in self.params.items() if p.grad is not None}
        
        # Scale the gradients
        for n, p in unscaled_gradients.items():
            availability = self.neuronal_availability[n].clone().detach()
            while(availability.ndim < self.params[n].grad.ndim): availability.unsqueeze_(dim=1)  # match dimensions to enable broadcasting
            self.params[n].grad = torch.mul(self.params[n].grad, availability)

        self.optimizer.step()

        # Update the path integrals
        for n, p in unscaled_gradients.items():
            batch_delta = torch.sub(self.params[n].clone().detach(), params_before_update[n])

            # Compute instantaneous synaptic importance values and take the mean over each neuron.
            si_update = (p * batch_delta)
            mean_dimensions = (tuple(range(1, batch_delta.dim())))  
            if len(mean_dimensions) > 0:
                self.task_importance[n] -= si_update.mean(dim=mean_dimensions)
            else:
                self.task_importance[n] -= si_update

        num_correct_predictions = torch.sum(predictions.argmax(1) == targets).item()

        return loss.detach(), num_correct_predictions

    def test_batch(self, predictions: torch.tensor, targets: torch.tensor, task_id: int) -> Tuple[torch.tensor, int]:
        """
        Testing of a single batch.

        Args:
            predictions (torch.tensor): Network predictions
            targets (torch.tensor): True labels
            task_id (int): Task ID

        Returns:
            torch.tensor: Batch loss
            int: Number of correct predictions
        """
        loss = self.loss_fn(predictions, targets).to(self.device)
        
        num_correct_predictions = torch.sum(predictions.argmax(1) == targets).item()
        
        return loss.detach(), num_correct_predictions

    def _compute_loss(self, predictions: torch.tensor, targets: torch.tensor) -> torch.tensor:
        """
        Computation of the loss. Since there is no regularization term, the vanilla loss is returned.

        Args:
            predictions (torch.tensor): Network predictions
            targets (torch.tensor): True labels

        Returns:
            torch.tensor: Loss
        """
        return self.loss_fn(predictions, targets).to(self.device)
    