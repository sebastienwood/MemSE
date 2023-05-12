import torch

from MemSE.nn.base_layer import MontecarloReturn, MemSEReturn
from MemSE.nn.map import register_memse_mapping

__all__ = ["Mean"]


@register_memse_mapping(dropin_for=set(["mean"]))
def Mean(previous_layer, *args, **kwargs):
    match previous_layer:
        case MemSEReturn():
            raise ValueError(
                "MemSE equations have not been defined for the Add operator."
            )
        case MontecarloReturn():
            return MontecarloReturn(
                torch.mean(previous_layer.out, *args, **kwargs),
                previous_layer.power,
            )
        case _:
            return torch.mean(previous_layer, *args, **kwargs)
