import warnings
from MemSE.definitions import SUPPORTED_OPS, UNSUPPORTED_OPS
import torch
import torch.nn as nn
import numpy as np
import gc

from typing import Iterator

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

def net_param_iterator(model: nn.Module) -> Iterator:
  ignored = []
  for _, module in model.named_modules():
    if type(module) in SUPPORTED_OPS.keys():
      yield module
    elif type(module) in UNSUPPORTED_OPS:
      raise ValueError(f'The network is using an unsupported operation {type(module)}')
    else:
      warnings.warn(f'The network is using an operation that is not supported or unsupported, ignoring it ({type(module)})')
      ignored.append(type(module))
  #print(set(ignored))

def n_vars_computation(model: nn.Module) -> int:
  n_vars_column, n_vars_layer = 0, 0
  for _, module in model.named_modules():
    if isinstance(module, nn.Linear):
      n_vars_column += module.out_features#[0]
      n_vars_layer += 1
    elif isinstance(module, Conv2DUF):
      n_vars_column += module.channel_out
      n_vars_layer += 1
  return n_vars_column, n_vars_layer

def memory_debug(cuda_profile:bool=True):
	for obj in gc.get_objects():
		try:
			if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data) and (obj.data.is_cuda if cuda_profile else True)):
				print(type(obj), obj.size())
		except:
			pass

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def memory_usage(tensor):
  return tensor.numel() * tensor.element_size()