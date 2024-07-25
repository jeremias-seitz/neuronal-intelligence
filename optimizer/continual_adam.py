import numpy as np
import torch


class ContinualADAMOptimizer(torch.optim.Optimizer):
    """
    Extension of the ADAM optimizer with the Neuronal Intelligence (NI) algorithm:
    
    NI computes importance scores for each neuron based on their accumulated contribution to the decrease in loss 
    over all tasks. Within each layer, neurons are ranked depending on their importance. The resulting rank is then
    fed into a sigmoid-like function that assigns an availability score with values between 0 and 1 to each neuron.
    The ADAM update for all parameters of a neuron is scaled by the corresponding availability score. Unavailable 
    neurons won't have their parameters updated while available neurons update as normal. During all the learning
    process, temporary importance scores for the current task are being accumulated. Whenever 'consolidate_importances'
    is called (usually at the end of a task), the temporary importance values will be added to overall importance 
    scores that are then used to compute the neuronal availabilities as described above.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, steepness_factor=0.5, offset_factor=0.4):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, steepness_factor=steepness_factor, offset_factor=offset_factor)
        super().__init__(params, defaults)
        self._is_initial_task = True
        self._offset_factor = offset_factor
        self._steepness_factor = steepness_factor

    def step(self):
        """
        Perform a parameter update.
        """
        loss = None
        for group in self.param_groups:
            for p in group['params']:
                grad = p.grad.data
                params = p.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # grad exponential average
                    state['m'] = torch.zeros_like(params)
                    # squared grad exponential average
                    state['v'] = torch.zeros_like(params)
                    # neuronal availability
                    state['availability'] = torch.ones(params.size(0)).to(params.device)
                    while state['availability'].ndim < params.ndim: state['availability'].unsqueeze_(dim=1)  # unsqueeze allows broadcasting but does not increase memory requirements
                    # squared grad exponential average
                    state['ni_accumulator'] = torch.zeros(params.size(0)).to(params.device)
                    # squared grad exponential average
                    state['ni'] = torch.zeros(params.size(0)).to(params.device)

                m, v = state['m'], state['v']

                if group['weight_decay'] != 0:
                    params.mul_(1 - group['lr'] * group['weight_decay'])

                b1, b2 = group['betas']
                state['step'] += 1

                # Momentum
                m = state['m'] = torch.mul(m, b1) + ( 1 - b1 )*grad
                
                # RMS
                v = state['v'] = torch.mul(v, b2) + ( 1 - b2 )*( grad*grad )

                # Adjusted learning rate
                at = group['lr'] * np.sqrt(1 - b2 ** state['step']) / (1 - b1 ** state['step'])

                # Parameter update
                synaptic_delta_batch = -at * torch.mul(m, state['availability']) / ( torch.sqrt(v) + group['eps'] )
                p.data = params + synaptic_delta_batch

                # Accumulate neuronal importance
                synaptic_task_importance = grad * synaptic_delta_batch

                mean_dimensions = (tuple(range(1, synaptic_delta_batch.ndim)))
                if len(mean_dimensions) > 0:
                    state['ni_accumulator'] -= synaptic_task_importance.mean(dim=mean_dimensions)
                else:
                    state['ni_accumulator'] -= synaptic_task_importance

        return loss
    
    def consolidate_importances(self):
        """
        Sum the neuronal task importance and evaluate the neuronal availability.
        """
        if self._is_initial_task:
            self._is_initial_task = False
            return
        
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]

                # Update importances
                state['ni'] += state['ni_accumulator']
                state['ni_accumulator'].zero_()

                # The availability is defined as a sigmoid over the range of importance values
                # Important neurons have an availability of close to zero, unimportant ones close to one
                ni = state['ni']
                mean = torch.mean(ni)
                std = torch.std(ni)
                offset = mean - std
                steepness = 5/std
                exp = torch.exp(-(ni - self._offset_factor*offset)*self._steepness_factor*steepness)
                state['availability'] = exp/(1+exp)

                # Add dimensions to enable broadcasting
                while(state['availability'].ndim < p.grad.ndim):
                    state['availability'].unsqueeze_(dim=1)  

    def reset(self):
        """
        Reset the optimizer.
        This has the same effect as reinstantiating, but the initial parameters need not be known.
        """
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]

                state['step'] = 0
                # grad exponential average
                state['m'] = torch.zeros_like(p.data)
                # squared grad exponential average
                state['v'] = torch.zeros_like(p.data)
                
