from torchvision import models
from torch import nn


def vit_b_16(output_dim: int, dropout: float=0) -> nn.Module:
    """
    Vision Transformer base model using 16x16 pixel image patches.
    
    The output layer is replaced with a linear layer required for the multi-headed output implementation.

    Args:
        output_dim (int): Output dimension
        dropout (float): Dropout ratio in [0, 1) where 1 would be 100%
    """
    vit = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT, progress=True)

    vit.heads = nn.Linear(vit.heads.head.in_features, output_dim)

    if dropout > 0:
        for module in vit.modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout

    return vit


def vit_b_32(output_dim: int, dropout: float=0) -> nn.Module:
    """
    Vision Transformer base model using 32x32 pixel image patches.
    
    The output layer is replaced with a linear layer required for the multi-headed output implementation.

    Args:
        output_dim (int): Output dimension
        dropout (float): Dropout ratio in [0, 1) where 1 would be 100%
    """
    vit = models.vit_b_32(weights=models.ViT_B_32_Weights.DEFAULT, progress=True)

    vit.heads = nn.Linear(vit.heads.head.in_features, output_dim)

    if dropout > 0:
        for module in vit.modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout

    return vit