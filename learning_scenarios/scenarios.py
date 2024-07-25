from abc import ABC, abstractmethod
from typing import List
import numpy as np


class IScenario(ABC):
    """
    The scenario interface class is a helper class that facilitates training and testing of a network by providing a 
    label offset as well as an output neuron mask depending on the learning scenario. This becomes relevant whenever 
    not all output neurons should be active for a single task, which is the case in e.g. a multi-headed network. To 
    account for the different number of outputs, the labels have to be adjusted accordingly.
    The three different scenarios of continual learning - task, domain and class incremental learning - differ in two 
    ways:

    - The availability of a context, i.e. which task is currently being trained / tested that is only available for 
        task incremental learning.
    - The selection of output nodes that are currently active. Task IL is multi-headed, domain and class IL are
        single-headed.

    Note that this class assumes the output of the model/backbone to be a linear layer.

    Also note that the output mask is a binary mask with the same dimensions as the output. This allows to have an
    arbitrary selection of relevant outputs and is used here to implement a multi-headed output with a single output
    layer containing all heads. The main benefit is that no model modification is necessary and they can be used
    straight out of the box (given the output is a linear layer).

    The label offset is a scalar that sets the lowest label in a task to zero such that e.g. the cross-entropy loss
    computation is feasible where the labels act as indices for the correct class. This assumes that all labels are
    numerical and contiguous within each task. And this allows the standard loss functions to be used without
    modification.
    """
    @abstractmethod
    def __init__(self, num_tasks:int, num_classes_per_task:int) -> None:
        pass

    def get_output_mask(self, task_id:int, is_joint_training:bool=False) -> List[int]:
        """
        Returns a mask for the output layer determining which nodes should be active, i.e. to be considered as output.
        The one special case is when performing joint training, where all output nodes should be active.

        Args:
            task_id (int): Task index
            is_joint_training (bool): Flag to indicate joint training (testing is unaffected)

        Returns:
            List(int): Mask indicating which outputs are relevant for the given task
        """
        if is_joint_training:
            return np.unique(self._output_mask).tolist()
        else:
            return self._output_mask[task_id]
    
    def get_label_offset(self, task_id:int, is_joint_training:bool=False) -> int:
        """
        Returns the label offset that is required in a multi-headed setup. It is chosen such that the lowest label of
        the current task is zero. This is necessary for the loss computation where label zero needs to correspond to
        the first unmasked output, the label one to the second output and so on. One can skip this (i.e. set the offset
        to zero) for single-headed readouts and for joint training. 

        Args:
            task_id (int): Task index
            is_joint_training (bool): Flag to indicate joint training (testing is unaffected)

        Returns:
            int: Label offset
        """
        if is_joint_training:
            return 0
        else:
            return self._label_offset[task_id]
        
    def get_output_dim(self) -> int:
        """
        Returns the required output dimension of the network.

        Returns:
            int: Number of outputs
        """
        return self._output_dim


class ClassIncrementalLearning(IScenario):
    """
    Class incremental learning expands the single-headed output layer for each new task. This is the hardest learning 
    scenario since output neurons corresponding to previous task are not masked out. As a result, their readout weights  
    can still experience gradient flow and can accordingly be subject to change.
    """
    def __init__(self, num_tasks: int, num_classes_per_task: int) -> None:
        """
        Args:
            num_tasks (int): Number of tasks
            num_classes_per_task (int): Number of classes per task
        """
        self._output_mask = [[i for i in range((task_idx + 1) * num_classes_per_task)] for task_idx in range(num_tasks)]
        self._label_offset = [0 for _ in self._output_mask]
        self._output_dim = num_tasks * num_classes_per_task


class DomainIncrementalLearning(IScenario):
    """
    Domain incremental learning is single-headed and always uses the same output nodes for all tasks and for training
    as well as testing. It can be used when the input distribution is changing but the number of classes remains constant.
    """
    def __init__(self, num_tasks: int, num_classes_per_task: int) -> None:
        """
        Args:
            num_tasks (int): Number of tasks
            num_classes_per_task (int): Number of classes per task
        """
        self._output_mask = [[i for i in range(num_classes_per_task)] for _ in range(num_tasks)]
        self._label_offset = [0 for _ in self._output_mask]
        self._output_dim = num_classes_per_task

    
class TaskIncrementalLearning(IScenario):
    """
    In the task incremental learning scenario the context is always known. The output is multi-headed with the 
    context defining the head to be used during training as well as testing. As a result, the single output layer is
    split into multiple heads, one for each task, but all the rest of the model is shared between the tasks.
    """
    def __init__(self, num_tasks: int, num_classes_per_task: int) -> None:   
        """
        Args:
            num_tasks (int): Number of tasks
            num_classes_per_task (int): Number of classes per task
        """
        self._output_mask = [[i + task_idx * num_classes_per_task for i in range(num_classes_per_task)] 
                                      for task_idx in range(num_tasks)]
        self._label_offset = [-idx[0] for idx in self._output_mask]
        self._output_dim = num_tasks * num_classes_per_task
