from __future__ import annotations
from MemSE.nn.definitions import TYPES_HANDLED
import torch.nn as nn


def add_type(func):
    TYPES_HANDLED[func.__name__] = func.memristored
    return func


class MemSELayer(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        """Base class for MemSE layer that shows all needed elements. Remember to decorate class with @add_type decorator for correct registration in the quantizer.
        """
        super().__init__(*args, **kwargs)
        
    def forward(self, x):
        raise ValueError('This property should be set in chilren class')
    
    @staticmethod
    def memse(layer: MemSELayer, memse_dict):
        raise ValueError('This property should be set in chilren class')
    
    @staticmethod
    def memse_monte_carlo(layer: MemSELayer, memse_dict):
        raise ValueError('This property should be set in chilren class')
    
    @property
    def out_features(self):
        raise ValueError('This property should be set in chilren class')
    
    @property
    def memristored(self):
        raise ValueError('This property should be set in chilren class')