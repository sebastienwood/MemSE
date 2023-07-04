from __future__ import annotations
from typing import Iterable

import torch
import torch.nn as nn

from MemSE.nn.base_layer import MemSELayer, MemSEReturn
from MemSE.nn.map import register_memse_mapping

__all__ = ['Reshaper', 'Flattener']


class Reshaper(MemSELayer):
    def initialize_from_module(self, shape: Iterable[int]) -> None:
        self.shape = shape
 
    def functional_base(self, x, *args, **kwargs):
        return torch.reshape(x, (-1,) + self.shape)

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
            pad_gamma = torch.reshape(gamma, x.shape + x.shape[1:])
        pad_gamma.extra_info = 'Gamma out of reshaper'
        return MemSEReturn(x, pad_gamma, pad_gamma_shape, power)


@register_memse_mapping()
class Flattener(Reshaper):
    def initialize_from_module(self) -> None:
        pass

    def functional_base(self, x, *args, **kwargs):
        return torch.flatten(x, start_dim=1)

    @classmethod
    @property
    def dropin_for(cls):
        return set([torch.flatten])
