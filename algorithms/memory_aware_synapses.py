import torch

from .regularization_loss import RegularizationLoss


class MAS(RegularizationLoss):
    """
    Implementation of the Memory Aware Synapses algorithm presented in the paper:

    'Memory Aware Synapses: Learning what (not) to forget'
    http://arxiv.org/abs/1711.09601

    Synaptic importance is estimated by how sensitive the input-output mapping is with respect to a change in
    parameters. The importance is then used as a weight in a L2 parameter reguarization scheme.
    The authors show that their approach (if done layer-wise and locally) is proportional to the activations of both
    the pre- as well as the post-synaptic neuron (very similar to Hebb's rule).
    """
    def prepare_task(self, task_id: int):
        """
        Increase the internal task counter and evaluates the parameter importance.
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
        Prepare variables and read configuration
        """
        self.params = {name: param for name, param in self.model.named_parameters() if param.requires_grad}
        self.params_prev_task = {name: param.clone().detach() for name, param in self.params.items()}
        self.importance = {name: torch.zeros_like(input=param, requires_grad=False, 
                                                  device=self.device) for name, param in self.params.items()}
        self.task_list = []
        self.regularization_strength = self.config.loss_entity.regularization_strength

    def _evaluate_importance(self):
        """
        Evaluate the importance parameter
        """
        self.model.train()
        task_id = self.task_list[-1]

        for (data, labels) in self.data_loaders[task_id]:
            predictions, labels = self.network_propagator.get_predictions_and_labels(inputs=data, 
                                                                                     labels=labels, 
                                                                                     task_id=task_id)
            
            predictions.pow_(2)
            
            loss = self._compute_loss(predictions=predictions, targets=labels, use_regularization=False)
            self.optimizer.zero_grad()
            loss.backward()

            for name, importance in self.importance.items():
                importance += self.params[name].grad.abs() / len(self.data_loaders[task_id])
