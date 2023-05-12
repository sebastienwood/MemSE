from MemSE.nn.base_layer import MontecarloReturn, MemSEReturn
import torch
import operator

from MemSE.nn.map import register_memse_mapping


@register_memse_mapping(dropin_for=set([operator.add, torch.add, "add"]))
def Add(previous_layer, previous_layer_2, *args, **kwargs):
	match previous_layer:
		case MemSEReturn():
			assert type(previous_layer_2) == MemSEReturn
			raise ValueError('MemSE equations have not been defined for the Add operator.')
		case MontecarloReturn():
			assert type(previous_layer_2) == MontecarloReturn
			return MontecarloReturn(
				torch.add(previous_layer.out, previous_layer_2.out, *args, **kwargs),
				previous_layer.power
			)
		case _:
			return torch.add(previous_layer, previous_layer_2, *args, **kwargs)
