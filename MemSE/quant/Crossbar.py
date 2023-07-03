import enum
import torch
import torch.nn as nn

from collections import defaultdict
from typing import Dict, Optional, Tuple
from MemSE.definitions import WMAX_MODE

__all__ = ['CrossBar']


class CROSSBAR_TYPE(enum.Enum):
	ALL = enum.auto()
	POS = enum.auto()
	NEG = enum.auto()


class CrossBar(object):
	def __init__(self, module: nn.Module, tensors: Dict, module_name: str) -> None:
		self.module = module
		self.manager: Optional[MemristorQuant] = None
		self.quanted, self.noised, self.c_one = False, False, False
		self.tensors = tensors
		for k, v in self.tensors.items():
			v.extra_info = f'QParam {k} of {module_name}'
		self.saved_tensors = {k:v.data.clone().cpu() for k, v in tensors.items()}
		for k, v in self.saved_tensors.items():
			v.extra_info = f'QParam (original) {k} of {module_name}'
		self.intermediate_tensors = {}

	@property
	def _unified_view(self):
		if len(self.tensors) == 1:
			return list(self.tensors.values())[0]
		else:
			return torch.cat(tuple(x if len(x.shape) == 2 else x.unsqueeze(1) for x in self.tensors.values()), dim=1)

	@property
	def out_features(self):
		return self.module.out_features

	@property
	def Wmax(self):
		if torch.all(self.module.Wmax == 0.):
			raise ValueError('Wmax has probably not been init. correctly (value = 0)')
		return self.module.Wmax

	@Wmax.setter
	def Wmax(self, val):
		if isinstance(val, torch.Tensor) and val.numel() == self.module.Wmax.numel():
			self.module.Wmax.data.copy_(val)
		else:
			self.module.Wmax.fill_(val)

	@property
	def Gmax(self):
		if torch.all(self.module.Gmax == 0.):
			raise ValueError('Gmax has probably not been init. correctly (value = 0)')
		return self.module.Gmax

	@Gmax.setter
	def Gmax(self, val):
		if isinstance(val, torch.Tensor) and val.numel() == self.module.Gmax.numel():
			self.module.Gmax.data.copy_(val)
		else:
			self.module.Gmax.data.fill_(val)

	@property
	def c(self):
		if self.c_one:
			return torch.ones_like(self.Wmax)
		return self.Gmax / self.Wmax

	def info(self):
		print(f'General info on Crossbar of {self.module.__class__.__name__}')
		for k, v in self.tensors.items():
			print(f'{k=} of shape {v.shape}')
		print('-'*10)

	def update_w_max(self, mode: WMAX_MODE):
		if mode in [WMAX_MODE.LAYERWISE, WMAX_MODE.ALL]:
			res = torch.max(torch.abs(self._unified_view))
		elif mode == WMAX_MODE.COLUMNWISE:
			res = torch.max(torch.abs(self._unified_view), dim=1).values
		else:
			raise ValueError("Not a valid WMAX_MODE")
		self.Wmax = res
		return res

	def unquant(self):
		if self.quanted:
			for k in self.tensors.keys():
				self.tensors[k].data.copy_(self.saved_tensors[k].to(self.tensors[k].data))
			self.intermediate_tensors.clear()
		self.quanted = False
		self.noised = False
		self.scaled = False

	def quant(self, N: int, c_one: bool = False, scaled: bool = True):
		if self.quanted:
			self.unquant()
		self.c_one = c_one
		for k in self.tensors.keys():
			true_value = self.tensors[k].data
			self.saved_tensors[k].copy_(true_value.clone().cpu())
			self._quantize(true_value, N)
			self.intermediate_tensors[k] = true_value.clone().cpu()
			self.intermediate_tensors[k].extra_info = f'QParam (cache) {k} of {self.module.__class__.__name__}'
			self.tensors[k].data.copy_(true_value)
		if scaled:
			self.rescale()
		self.quanted = True

	def renoise(self, std_noise: float, keep: CROSSBAR_TYPE = CROSSBAR_TYPE.ALL):
		raise RuntimeError('This renoise method has been deprecated, please use FORWARD_MODE.MONTECARLO for noisy forwards')
		assert self.quanted, 'Cannot renoise the original representation'
		noise = defaultdict(lambda: None)
		for k in self.tensors.keys():
			noise[k] = self._renoise(self.tensors[k], self.intermediate_tensors[k], std_noise, keep=keep)
		self.rescale()
		self.noised = True
		return noise

	def denoise(self):
		raise RuntimeError('This renoise method has been deprecated, please use FORWARD_MODE.MONTECARLO for noisy forwards')
		assert self.quanted, 'Cannot renoise the original representation'
		for k in self.tensors.keys():
			self.tensors[k].data.copy_(self.intermediate_tensors[k])
		self.rescale()
		self.noised = False

	def rescale(self, tensors=None):
		if tensors is None:
			tensors = self.tensors.values()
			self.scaled = True
		else:
			self.scaled = False
		for v in tensors:
			inp_shape = 'i' if len(v.shape) == 1 else 'ij'
			v.data.copy_(torch.einsum(f'{inp_shape},i -> {inp_shape}', v.data, 1/self.c))

	@torch.no_grad()
	def _quantize(self, tensor, N: int) -> None:
		self.scaled = False
		delta = self.Gmax / N if not self.c_one else self.Wmax / N
		inp_shape = 'i' if len(tensor.shape) == 1 else 'ij'
		tensor.copy_(torch.einsum(f'{inp_shape},i -> {inp_shape}', torch.floor(torch.einsum(f'{inp_shape},i -> {inp_shape}', tensor, (self.c/delta))), delta))