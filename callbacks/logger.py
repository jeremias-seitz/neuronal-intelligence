import wandb
from utils import generate_wandb_run_name_from_hydra
from statistics import mean
from omegaconf import OmegaConf

from .base import Callback


class WANDBLogger(Callback):
    """
    Logger callback for Weights & Biases (wandb) https://wandb.ai/home.
    This logger is highly specific to a continual learning classification setting.
    """
    def __init__(self, enable_wandb: bool, experiment_name: str, entity: str):
        self._enabled = enable_wandb
        self._experiment_name = experiment_name
        self._entity = entity

    def on_init(self, trainer, config):
        """
        Setup wandb configuration and prepare logging variables.

        Args:
            trainer (trainer.BaseTrainer): Trainer object (main object for user interaction)
            config (dictconfig.DictConfig): Hydra configuration file
        """
        self._trainer = trainer
        self._model = trainer.model
        self._config = config

        run_name = generate_wandb_run_name_from_hydra(config=config)
        wandb.config = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)

        if self._enabled:
            self.run = wandb.init(project=self._experiment_name, 
                       entity=self._entity, 
                       name=run_name, 
                       settings=wandb.Settings(start_method="thread"),
                       config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True))
        else:
            self.run = wandb.init(project=self._experiment_name, 
                       entity=self._entity, 
                       name=run_name, 
                       settings=wandb.Settings(start_method="thread"), 
                       mode="disabled")
            
        # statistics
        self._mean_accuracy = config.n_tasks*[None]
        self._accuracy_initial = config.n_tasks*[None]
        self._accuracy_final = config.n_tasks*[None]

        self._mean_loss = config.n_tasks*[None]
        self._loss_initial = config.n_tasks*[None]
        self._loss_final = config.n_tasks*[None]

    def on_validation_end(self, task_id: int, **kwargs):
        """
        Log accuracy and loss values to wandb.
        
        Args:
            task_id (int): Task index
            enable_logging (bool): Flag to enable/disable logging
            accuracy (float): Ratio of correctly predicted labels for classification
            loss (float): Test loss
        """
        if not kwargs["enable_logging"]:
            return
        
        accuracy = kwargs["accuracy"]
        loss = kwargs["loss"]

        initial_evaluation = False

        # accuracy evaluation
        if self._accuracy_initial[task_id] is None:
            self._accuracy_initial[task_id] = accuracy
            initial_evaluation = True
        self._accuracy_final[task_id] = accuracy

        if self._mean_accuracy[task_id] is None:
            self._mean_accuracy[task_id] = mean([a for a in self._accuracy_final if a is not None])

        # loss evaluation
        if self._loss_initial[task_id] is None:
            self._loss_initial[task_id] = loss
        self._loss_final[task_id] = loss

        if self._mean_accuracy[task_id] is None:
            self._mean_loss[task_id] = mean([l for l in self._loss_final if l is not None])

        # The initial_eval flag is used to prevent multiple logs if the task
        # is relearned at a later time. Note that the mean values are not updated
        # in those cases.
        if initial_evaluation:
            wandb.log({"task_id": task_id,
                       "initial_accuracy": self._accuracy_initial[task_id],
                       "initial_loss": self._loss_initial[task_id],
                       "mean_accuracy": self._mean_accuracy[task_id],
                       "mean_loss": self._mean_loss[task_id]})
        
        # Last task has been learnt
        if task_id >= self._config.n_tasks - 1:
            accuracy_delta = [final - initial for initial, final in zip(self._accuracy_initial, self._accuracy_final)]
            for i in range(len(self._accuracy_final)):
                wandb.log({"task_id": i,
                           "final_accuracy": self._accuracy_final[i],
                           "final_loss": self._loss_final[i],
                           "delta_accuracy": accuracy_delta[i]})
