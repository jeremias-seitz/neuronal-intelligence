from typing import List


class TaskShuffledLabels:
    """
    Build a mapping that assigns a new label to each old label according to the change in task order. This is relevant
    only when the tasks are shuffled and the order accordingly becomes randomized. Objects from this class are used as
    target transforms that can be passed to torch data loader objects. It stores a dictionary whose keys are the old
    labels and whose values are the new labels.

    Note that this transform preserves the label order within each task. Also note that this transform assumes the
    class labels to be numerical and contiguous.
    """
    def __init__(self, num_tasks:int, num_classes_per_task:int, num_outputs:int, task_permutation:List[int]) -> None:
        """
        Args:
            num_tasks (int): Number of tasks
            num_classes_per_task (int): Number of classes per task
            num_outputs (int): Output dimension
            task_permutation (List[int]): Task order after shuffling
        """
        # Only perform the shuffle for task/class incremental learning, but not for domain incremental learning
        self._do_shuffle = num_tasks * num_classes_per_task == num_outputs  # Returns false only for domain IL

        self._map = {}
        idx = 0
        for task_idx in range(num_tasks):
            for class_idx in range(num_classes_per_task):
                self._map[num_classes_per_task*task_permutation[task_idx] + class_idx] = idx
                idx += 1

    def __call__(self, label: int) -> int:
        """
        Maps the old to the new task shuffled labels

        Args:
            label (int): Tensor class labels

        Returns:
            int: Task shuffled class labels
        """
        if self._do_shuffle:
            return self._map[label]
        else:
            return label

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
    