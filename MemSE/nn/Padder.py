from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn
from MemSE.nn.base_layer import MemSELayer, MemSEReturn

from MemSE.utils import default


class Padder(MemSELayer):
    def initialize_from_module(self, padding: Tuple, value: float = 0., gamma_value: Optional[float] = None) -> None:
        self.padding = padding
        self.value = value
        self.gamma_value = default(gamma_value, value)
        
    def functional_base(self, x, padding=None, value=None, *args, **kwargs):
        return nn.functional.pad(x, self.padding if padding is None else padding, value=self.value if value is None else value)
    
    def memse(self, previous_layer:MemSEReturn, *args, **kwargs):
        x = previous_layer.out
        gamma = previous_layer.gamma
        gamma_shape = previous_layer.gamma_shape
        power = previous_layer.power

        x = self.functional_base(x)
        if gamma_shape is not None:
            pad_gamma_shape = x.shape + x.shape[1:]
            pad_gamma = gamma
        else:
            pad_gamma_shape = gamma_shape
            if len(gamma.shape) == 7:  # chwchw
                padding = self.padding + (0, 0) + self.padding
            else:
                padding = self.padding + self.padding
            pad_gamma = self.functional_base(gamma, padding=padding, value=self.gamma_value)
        
        pad_gamma.extra_info = 'Gamma out of padder'
        return MemSEReturn(x, pad_gamma, pad_gamma_shape, power)
