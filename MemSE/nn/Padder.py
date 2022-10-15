from __future__ import annotations
from typing import Optional, Tuple

import torch.nn as nn

from MemSE.utils import default
# TODO called in network manipulations, perform the base padding on mu and on gamma if it is provided, to add in SUPPORTED_OPS + remove management from Linear
# TODO same for reshape
# TODO same for flatten

class Padder(nn.Module):
    def __init__(self, padding: Tuple, value: float = 0., gamma_value: Optional[float] = None) -> None:
        super().__init__()
        self.padding = padding
        self.value = value
        self.gamma_value = default(gamma_value, value)
        
    def forward(self, x):
        return nn.functional.pad(x, self.padding, value=self.value)   
    
    @staticmethod
    def memse(padder: Padder, memse_dict):
        pad_mu = padder.forward(memse_dict['mu'])
        gamma, gamma_shape = memse_dict['gamma'], memse_dict['gamma_shape']
        if gamma_shape is not None:
            pad_gamma_shape = pad_mu.shape + pad_mu.shape[1:]
            pad_gamma = gamma
        else:
            pad_gamma_shape = gamma_shape
            if len(gamma.shape) == 7:  # chwchw
                padding = padder.padding + (0, 0) + padder.padding
            else:
                padding = padder.padding + padder.padding
            pad_gamma = nn.functional.pad(gamma, padding, value=padder.gamma_value)
        pad_gamma.extra_info = 'Gamma out of padder'
        memse_dict['current_type'] = 'Padder'
        memse_dict['mu'] = pad_mu
        memse_dict['gamma'] = pad_gamma
        memse_dict['gamma_shape'] = pad_gamma_shape