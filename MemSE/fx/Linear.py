import torch
import torch.nn as nn
from MemSE.nn import Padder

__all__ = ['fuse_linear_bias']

def fuse_linear_bias(linear: nn.Linear):
    assert isinstance(linear, nn.Linear)
    if linear.bias is None:
        return linear
    fused_linear = nn.Linear(linear.weight.shape[1] + 1, linear.weight.shape[0], bias=False)
    biases = linear.bias.repeat_interleave((linear.weight.shape[0]//linear.bias.shape[0])).unsqueeze(1)
    fused_linear.weight.data.copy_(torch.cat((linear.weight, biases), dim=1))
    seq = nn.Sequential(
        Padder((0, 1), value=1., gamma_value=0.),
        fused_linear,
    )
    return seq