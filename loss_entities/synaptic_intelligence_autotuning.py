import torch
from typing import Tuple

from .regularization_loss import ILossEntity


class SynapticIntelligenceAutotuning(ILossEntity):
    """
    Extension of the Synaptic Intelligence alrogithm (see class `SynapticIntelligence` in this module).

    This algorithm scales the importance values such that all previous and the current task receive the same weight in
    the loss function (regularization vs regular loss e.g. cross-entropy). Since this might not be optimal, the
    regularization strength can still be specified as a hyperparameter. However, note that (in contrast to the original
    implementation) this is a multiplicative factor. A value of 1 means balance between old and new tasks, a value of 
    e.g. 2 means that older tasks are twice as important as the latest task. 
    """
    def prepare_task(self, task_id: int):
        """
        Increases the internal task counter and evaluates the parameter importance.
        """
        if self.is_initial_task:
            self.prepare_initial_task(task_id=task_id)
            self.is_initial_task = False
        else:
            self.prepare_new_task(task_id=task_id)        

    def prepare_initial_task(self, task_id: int):
        """
        Create all variables required for regularization.

        Args:
            task_id (int): Task index of the first task
        """
        # if isinstance(self.model, VisionTransformer):
        #     self.params = {name: param for name, param in self.model.named_parameters() if param.requires_grad and 'in_proj_weight' in name}
        # else:
        self.params = {name: param for name, param in self.model.named_parameters() if param.requires_grad}

        self.params_prev_task = {name: param.clone().detach() for name, param in self.params.items()}
        self.importance = {name: torch.zeros_like(input=param, requires_grad=False, 
                                                  device=self.device) for name, param in self.params.items()}
        self.path_integrals = {name: torch.zeros_like(input=param, requires_grad=False, 
                                                      device=self.device) for name, param in self.params.items()}

        self.task_list = []
        self.damping = self.config.loss_entity.damping
        self.regularization_strength = self.config.loss_entity.regularization_strength

        self._task_loss_before_training = self._evaluate_task_performance(task_id=task_id)

    def prepare_new_task(self, task_id: int):
        """
        Update importances and prepare variables for the next task.

        Args:
            task_id (int): Index of the task to be prepared
        """
        self._evaluate_importance(task_id=task_id-1)
        
        for param_name, param in self.params.items():
            self.params_prev_task[param_name] = param.clone().detach().to(self.device)

        self._task_loss_before_training = self._evaluate_task_performance(task_id=task_id)

    def _evaluate_task_performance(self, task_id: int):
        """
        Compute the unregularized loss over the full dataset.

        Args:
            task_id (int): Index of the task to be evaluated
        """
        loss_dataset = 0.0

        with torch.no_grad():
            for (inputs, labels) in self.data_loaders[task_id]:

                predictions, labels = self.network_propagator.get_predictions_and_labels(inputs=inputs, 
                                                                                        labels=labels, 
                                                                                        task_id=task_id)

                loss_dataset += self.loss_fn(predictions, labels).item()

        print(f"Loss training set on task {task_id + 1}: {loss_dataset}")

        return loss_dataset

    def _evaluate_importance(self, task_id: int):
        """
        Evaluate the importance parameters. In theory, the importances should be chosen such that the regularization
        term - when plugging in the overall parameter change during the training of a task - equals the change in
        unregularized loss during that same training of a task. This can be obtained by appropriately scaling the 
        importance values.

        Args:
            task_id (int): Task index
        """
        task_loss_after_training = self._evaluate_task_performance(task_id=task_id)

        unregularized_loss_delta = self._task_loss_before_training - task_loss_after_training

        sum_path_integrals = 0   
        for _, path_integral in self.path_integrals.items():
            sum_path_integrals += path_integral.sum().item()

        # Scales the path integrals such that their sum is equal to the total decrease in unregularized training loss.
        overestimation_factor = unregularized_loss_delta / sum_path_integrals

        regularized_loss_delta = 0.0
        importance_single_task = {}
        for name in self.params.keys():
            delta_theta_squared = torch.sub(self.params_prev_task[name], self.params[name].clone().detach()) ** 2
            importance_single_task[name] = overestimation_factor * self.path_integrals[name] / (delta_theta_squared + self.damping)
            regularized_loss_delta += (importance_single_task[name] * delta_theta_squared).sum().item()

        # This can be achieved by the following scaling. Note that this assumes equal weight for both old
        # and new tasks which might not necessarily be the optimal strategy.
        unreg_vs_reg_loss_factor = unregularized_loss_delta / regularized_loss_delta
        
        # Saturate the loss factor (also improves numerical stability)
        if unreg_vs_reg_loss_factor > 100:
            unreg_vs_reg_loss_factor = 100
            print("Loss factor saturated, set to 100.")

        for name, importance in self.importance.items():
            importance += unreg_vs_reg_loss_factor * importance_single_task[name]
            self.path_integrals[name].zero_()

        print(f"Estimated regularization strength: {overestimation_factor*unreg_vs_reg_loss_factor}")

    def train_batch(self, predictions: torch.tensor, targets: torch.tensor, task_id: int) -> Tuple[torch.tensor, int]:
        """
        Supervised training with vanilla backpropagation for a single batch.

        Args:
            predictions (torch.tensor): Network predictions
            targets (torch.tensor): True labels
            task_id (int): Task index

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
    