from .base import Callback


class ResetOptimizer(Callback):

    def on_init(self, trainer, config):
        """
        Set variables.

        Args:
            trainer (trainer.BaseTrainer): Trainer object (main object for user interaction)
            config (dictconfig.DictConfig): Hydra configuration file
        """
        self._model = trainer.model
        self._optimizer = trainer.optimizer

    def on_new_task(self, **kwargs):
        """
        Reset the optimizer by calling the constructor with the default values.
        """
        opt = self._optimizer
        super(type(opt), opt).__init__(self._model.parameters(), opt.defaults)
