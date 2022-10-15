from __future__ import annotations
from typing import Iterable

import torch
import torch.nn as nn

__all__ = ['Reshaper', 'Flattener']
# TODO called in network manipulations, perform the base padding on mu and on gamma if it is provided, to add in SUPPORTED_OPS + remove management from Linear
# TODO same for reshape
# TODO same for flatten

class Reshaper(nn.Module):
    def __init__(self, shape: Iterable[int]) -> None:
        super().__init__()
        self.shape = shape
        
    def forward(self, x):
        return torch.reshape(x, (-1,) + self.shape)
    
    @staticmethod
    def memse(reshaper: Reshaper, memse_dict):
        pad_mu = reshaper.forward(memse_dict['mu'])
        gamma, gamma_shape = memse_dict['gamma'], memse_dict['gamma_shape']
        if gamma_shape is not None:
            pad_gamma_shape = pad_mu.shape + pad_mu.shape[1:]
            pad_gamma = gamma
        else:
            pad_gamma_shape = gamma_shape
            pad_gamma = torch.reshape(gamma, pad_mu.shape + pad_mu.shape[1:])
        pad_gamma.extra_info = 'Gamma out of reshaper'
        memse_dict['current_type'] = reshaper.__class__.__name__
        memse_dict['mu'] = pad_mu
        memse_dict['gamma'] = pad_gamma
        memse_dict['gamma_shape'] = pad_gamma_shape
        

class Flattener(Reshaper):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        
    def forward(self, x):
        return torch.flatten(x, start_dim=1)
