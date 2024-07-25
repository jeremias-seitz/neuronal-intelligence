from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau, CosineAnnealingLR


class LRSchedulerWrapper:
    """
    Wrapper class for several LRSchedulers from Pytorch. Takes care of passing the proper parameters to the different
    types of schedulers.
    """
    def set_scheduler(self, lr_scheduler: LRScheduler) -> None:
        """
        Set the learning rate scheduler.
        """
        self._scheduler = lr_scheduler

    def step(self, **kwargs):
        """
        Wrapper function that assumes that all arguments required for all the different schedulers are passed at once 
        and then selects the relevant parameters to match the call signature of the 'step()' function of the specific
        scheduler.
        If a new learning rate scheduler is to be implemented, add an elif statement here with the proper 'step()' 
        call signature and make sure the required keyword arguments are passed to the call to this function (currently
        line 111 in main.py).
        """
        if isinstance(self._scheduler, ReduceLROnPlateau):
            self._scheduler.step(kwargs['loss'])
        elif isinstance(self._scheduler, CosineAnnealingLR):
            self._scheduler.step()
