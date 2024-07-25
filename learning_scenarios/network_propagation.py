import torch
from typing import Tuple
from omegaconf import dictconfig

from learning_scenarios import IScenario, ClassIncrementalLearning


class NetworkPropagator():
    """
    Class that propagates inputs through the network. In cases when not all output neurons should be active (e.g. 
    multiple output heads) this class applies an output mask and adjusts the data labels accordingly to enable 
    proper loss calculation. 
    """
    def __init__(self, 
                 model: torch.nn.Module, 
                 configuration: dictconfig.DictConfig,
                 learning_scenario: IScenario,
                 device: str,
                 ) -> None:
        """
        Args:
            model (torch.nn.Module): Model used for training.
            configuration (dictconfig.DictConfig): Hydra config file
            learning_scenario (IScenario): Learning scenario (Class / Domain / Task incremental learning)
            device (str): Device string 'cuda:<GPUID>' or 'cpu'
        """
        # Vital components
        self._model = model
        self._config = configuration
        self._device = device

        # Learning scenario (task / domain / class incremental learning)
        self._learning_scenario = learning_scenario
        self._label_offset = 0
        self._task_counter = 0
        self._last_task_id = -1
        self._was_training = False

    def prepare_task(self, task_id:int):
        """
        Depending on the learning scenario, different tasks might require different output neurons to be active. This 
        function serves as a wrapper to the ILearningScenario class and sets the output mask as well as the 
        corresponding label offset (important for multi-headed output).

        Note that the output mask of the learning scenario is a binary mask with the same dimensions as the output.
        This allows to have an arbitrary selection of relevant outputs and is used here to implement a multi-headed
        output with a single output layer containing all heads. The main benefit is that no model modification is
        necessary and they can be used straight out of the box.

        The label offset is a scalar that sets the lowest label in a task to zero such that e.g. the cross-entropy loss
        computation is feasible where the labels act as indices for the correct class. This assumes that all labels are
        numerical and contiguous within each task. And this allows the standard loss functions to be used without
        modification.

        Args:
            task_id (int): Task ID
        """
        # The Class Incremental learning scenario is special since output neurons - once activated - should always 
        # remain active. This is mainly relevant if earlier tasks get tested (or even retrained) later on.
        if isinstance(self._learning_scenario, ClassIncrementalLearning):
            task_id_masking = max(self._task_counter, task_id)
        else:
            task_id_masking = task_id

        if self._model.training:  # check for the joint training flag in the config only when training
            self._output_mask = self._learning_scenario.get_output_mask(task_id=task_id_masking,
                                                                        is_joint_training=self._config.is_joint_training)
            self._label_offset = self._learning_scenario.get_label_offset(task_id=task_id_masking,
                                                                          is_joint_training=self._config.is_joint_training)
        else:
            self._output_mask = self._learning_scenario.get_output_mask(task_id=task_id_masking)
            self._label_offset = self._learning_scenario.get_label_offset(task_id=task_id_masking)

    def get_predictions_and_labels(self, inputs: torch.tensor, labels: torch.tensor, task_id: int) \
        -> Tuple[torch.tensor, torch.tensor]:
        """
        Returns the masked outputs after forwarding the inputs through the network as well as the labels after
        applying the appropriate offset depending on the learning scenario. This is relevant when the network employs
        multiple output heads or in general when not all output neurons should be enabled. 

        Args:
            inputs (torch.tensor): Network inputs
            labels (torch.tensor): Class labels
            task_id (int): Task ID

        Returns:
            torch.tensor: Masked network output
            torch.tensor: Offset labels
        """
        if task_id != self._last_task_id or self._model.training != self._was_training:
            self.prepare_task(task_id=task_id)

        masked_predictions = self._model.forward(inputs.to(self._device))[:, self._output_mask]
        offset_labels = labels.to(self._device) + self._label_offset

        self._last_task_id = task_id
        self._was_training = self._model.training

        return masked_predictions, offset_labels
