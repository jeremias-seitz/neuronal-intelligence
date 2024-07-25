class Callback():
    """
    Base class for callback functions. All callback functions defined here will be called by the trainer class at
    specific times during the training and testing process. The actual callback function implementations should inherit
    from this class and override only the relevant callbacks.
    See the 'BaseTrainer' class in the 'trainer' module for the exact times the callback functions are called. To add
    more callback functions, add the new signature here, add the signature call to the trainer class and implement the
    actual functionality in a child class in this module.
    """
    def __init__(self):
        pass
    
    def on_init(self, trainer, config):
        self._trainer = trainer
        self._model = trainer.model
        self._config = config

    # task related
    def on_new_task(self, task_id: int, **kwargs):
        pass

    def on_task_end(self, task_id: int, **kwargs):
        pass

    # training
    def on_training_start(self, task_id: int, **kwargs):
        pass

    def on_training_end(self, task_id: int, **kwargs):
        pass

    # during training
    def on_batch_end(self, task_id: int, **kwargs):
        pass

    # validation
    def on_validation_start(self, task_id: int, **kwargs):
        pass

    def on_validation_end(self, task_id: int, **kwargs):
        pass

