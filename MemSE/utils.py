from typing import Optional, Tuple
import torch
import torch.nn as nn
import numpy as np
import gc
import sys
import random

from MemSE.nn import Conv2DUF

def maybe_cuda_from_numpy(tensor, device_id=0, dtype=None, use_cuda:bool=False):
	if type(tensor) is np.ndarray:
		tensor = torch.from_numpy(tensor)
	return tensor.to(f'cuda:{device_id}', dtype=dtype, non_blocking=True) if use_cuda else tensor

def mse(original, other):
	return ((original - other)**2).mean()

def print_compare(original, other):
	print(original.shape)
	print(other.shape)
	print(mse(original, other))
	assert np.allclose(original, other), 'diff'


def n_vars_computation(model: nn.Module) -> Tuple[int, int]:
	n_vars_column, n_vars_layer = 0, 0
	for _, module in model.named_modules():
		if isinstance(module, nn.Linear) or isinstance(module, Conv2DUF):
			n_vars_column += module.out_features#[0]
			n_vars_layer += 1
	return n_vars_column, n_vars_layer


def memory_debug(cuda_profile:bool=True) -> list:
	objs = []
	storage = set()
	for obj in gc.get_objects():
		try:
			if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data) and (obj.data.is_cuda if cuda_profile else True)) and not obj.storage() in storage:
				objs.append(obj)
				storage.add(obj.storage())
		except:
			pass
	return sorted(objs, key=lambda item: memory_usage(item), reverse=True)


def memory_report(cuda_profile:bool=True, nth_first:Optional[int]=None) -> None:
	res = memory_debug(cuda_profile)
	s = f'Total memory used {round(sum(memory_usage(k) for k in res), 2)}MB for {len(res)} tensors \n'
	for idx, k in enumerate(res):
		if nth_first is not None and idx == nth_first - 1: break
		s += f'{round(memory_usage(k), 2)}MB of shape {k.shape} {f"({k.extra_info})" if hasattr(k, "extra_info") else ""}\n'
	print(s)

def count_parameters(model) -> int:
	return sum(p.numel() for p in model.parameters() if p.requires_grad)


def memory_usage_theoric(tensor) -> int:
	return tensor.numel() * tensor.element_size()


def memory_usage_cpu(tensor) -> float:
	"""Return the memory usage of a tensor on CPU side in MB

	Args:
		tensor (_type_): _description_

	Returns:
		_type_: _description_
	"""
	assert tensor.device.type == 'cpu'
	return sys.getsizeof(tensor.storage())/1024/1024


def memory_usage_gpu(tensor) -> float:
    return tensor.storage().size() * tensor.storage().element_size() /1024/1024


def memory_usage(tensor) -> float:
    if tensor.device.type == 'cpu':
        return memory_usage_cpu(tensor)
    elif tensor.device.type == 'cuda':
        return memory_usage_gpu(tensor)
    else:
        raise ValueError(f'{tensor.device.type} is not recognized')


def default(val, dval):
	return val if val is not None else dval


def torchize(val):
	if isinstance(val, np.ndarray):
		return torch.from_numpy(val)
	elif isinstance(val, (float, int)):
		return torch.tensor(val)
	elif isinstance(val, torch.Tensor):
		return val
	else:
		raise RuntimeError(f'{type(val)=} is not a recognized type')


def listify(val):
	if isinstance(val, list):
		return val
	else:
		return [val]


def numpify(val: torch.Tensor):
    assert isinstance(val, torch.Tensor)
    return val.detach().cpu().numpy()


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)