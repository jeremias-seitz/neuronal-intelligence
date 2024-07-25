from torch import nn
from typing import List


class MLP(nn.Module):
    """
    MLP with ReLu activations and potential dropout.
    """
    def __init__(self, input_dim: int=28*28, 
                 hidden_dims: List[int]=[2000,2000],
                 output_dim: int=10, 
                 dropout: float=0):
        """
        Args:
            input_dim (int): Input dimension
            hidden_dims (List[int]): List of hidden dimensions, where every entry results in another linear layer
            output_dim (int): Output dimension
            dropout (float): Dropout ratio in [0, 1) where 1 would be 100%
        """
        super().__init__()
        self.flatten = nn.Flatten()
        self.mlp = nn.Sequential()

        last_dim = input_dim

        for idx, hidden_dim in enumerate(hidden_dims):
            self.mlp.add_module(name=f'lin{idx}', module=nn.Linear(last_dim, hidden_dim))
            if dropout > 0:
                self.mlp.add_module(name=f'drop{idx}', module=nn.Dropout(p=dropout))
            self.mlp.add_module(name=f'relu{idx}', module=nn.ReLU())
            last_dim = hidden_dim
        
        self.mlp.add_module(name='out', module=nn.Linear(last_dim, output_dim))

    def forward(self, x):
        x = self.flatten(x)
        logits = self.mlp(x)
        return logits
