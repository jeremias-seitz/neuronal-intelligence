import torch
import os

from .base import Callback


class SaveModel(Callback):
    """
    Callback to save the model after training each task. The following variables will be saved:
    - 'model_state_dict': state dictionary of the model
    - 'optimizer_state_dict': state dictionary of the optimizer
    - 'config': hydra configuration file
    - 'task_id': index of the task, will be 0 <= 'task_idx' < 'n_tasks'
    """
    def __init__(self, path: str, filename: str):
        """
        Args:
            path(str): path to directory where the savefile should be stored
            filename(str): file name
        """
        self._path = path
        self._filename = filename

    def on_init(self, trainer, config):
        """
        Create the directory if it does not already exist and reference importance objects.

        Args:
            trainer (trainer.BaseTrainer): Trainer object (main object for user interaction)
            config (dictconfig.DictConfig): Hydra configuration file
        """
        if not os.path.exists(self._path_directory):
            os.makedirs(self._path_directory)
        self._trainer = trainer
        self._model = trainer.model
        self._optimizer = trainer.optimizer
        self._config = config

    def on_new_task(self, task_id: int, **kwargs):
        """
        Save model and selected variables.

        Args:
            task_id (int): Task index 
        """
        savefile_name = f"{self._filename}_task{task_id}"
        
        torch.save({
            'model_state_dict': self._model.state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict(),
            'config': self._config,
            'task_id': task_id,
            }, f"{self._path}{savefile_name}.pt")
        
        print(f"model saved: {savefile_name}")


class ResetModel(Callback):

    def on_init(self, trainer, config):
        """
        Set variables.

        Args:
            trainer (trainer.BaseTrainer): Trainer object (main object for user interaction)
            config (dictconfig.DictConfig): Hydra configuration file
        """
        self._model = trainer.model

    def on_new_task(self, **kwargs):
        """
        Reset all model parameters.

        See: https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819/10
        """
        @torch.no_grad()
        def weight_reset(module: torch.nn.Module):
            # reset parameters if the module contains the method
            reset_parameters = getattr(module, "reset_parameters", None)
            if callable(reset_parameters):
                module.reset_parameters()

        # Applies fn recursively to every submodule see: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
        self._model.apply(fn=weight_reset)
