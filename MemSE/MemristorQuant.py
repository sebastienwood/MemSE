import torch
import torch.nn as nn
from MemSE.definitions import WMAX_MODE
from MemSE.nn.definitions import TYPES_HANDLED
from MemSE.utils import default, torchize

from typing import Union

__all__ = ['MemristorQuant']


class CrossBar(object):
	def __init__(self, module: nn.Module) -> None:
		self.module = module
		self.quanted, self.noised, self.c_one = False, False, False
		existing_k = [k for k in TYPES_HANDLED[type(module)] if hasattr(module, k) and getattr(module, k) is not None]
		self.tensors = {k:getattr(module,k) for k in existing_k}
		for k, v in self.tensors.items():
			v.extra_info = f'QParam {k} of {module.__class__.__name__}'
		self.saved_tensors = {k:getattr(module,k).data.clone().cpu() for k in existing_k}
		for k, v in self.saved_tensors.items():
			v.extra_info = f'QParam (original) {k} of {module.__class__.__name__}'
		self.intermediate_tensors = {}

		module.register_buffer('Wmax', torch.tensor([0.] * module.out_features))
		module.register_parameter('Gmax', nn.Parameter(torch.tensor([0.] * module.out_features)))

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

	def quant(self, N: int, c_one: bool = False):
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
		self.rescale()
		self.quanted = True

	def renoise(self, std_noise: float):
		assert self.quanted, 'Cannot renoise the original representation'
		for k in self.tensors.keys():
			self.tensors[k].data.copy_(self.intermediate_tensors[k])
			shape, device = self.tensors[k].shape, self.tensors[k].device
			self.tensors[k].data += torch.normal(mean=0., std=std_noise, size=shape, device=device)
			self.tensors[k].data -= torch.normal(mean=0., std=std_noise, size=shape, device=device)
		self.rescale()
		self.noised = True

	def denoise(self):
		assert self.quanted, 'Cannot renoise the original representation'
		for k in self.tensors.keys():
			self.tensors[k].data.copy_(self.intermediate_tensors[k])
		self.rescale()
		self.noised = False

	def rescale(self):
		for v in self.tensors.values():
			inp_shape = 'i' if len(v.shape) == 1 else 'ij'
			v.data.copy_(torch.einsum(f'{inp_shape},i -> {inp_shape}', v.data, 1/self.c))

	@torch.no_grad()
	def _quantize(self, tensor, N: int) -> None:
		delta = self.Gmax / N if not self.c_one else self.Wmax / N
		inp_shape = 'i' if len(tensor.shape) == 1 else 'ij'
		tensor.copy_(torch.einsum(f'{inp_shape},i -> {inp_shape}', torch.floor(torch.einsum(f'{inp_shape},i -> {inp_shape}', tensor, (self.c/delta))), delta))
		

class MemristorQuant(object):
	def __init__(self,
				 model: nn.Module,
				 N: int = 128,
				 wmax_mode:Union[str, WMAX_MODE] = WMAX_MODE.ALL,
				 Gmax=0.1,
				 std_noise:float=1.) -> None:
		super().__init__()
		self.model = model
		model.__attached_memquant = self
		self.quanted = False
		self.noised = False
		self.N = N
		self.std_noise = std_noise
		self.init_wmax(wmax_mode)
		Gmax = default(Gmax, 0.1)
		self.crossbars = []
		for m in model.modules():
			if type(m) in TYPES_HANDLED.keys():
				self.crossbars.append(CrossBar(m))
		self.update_w_max()
		self.init_gmax(Gmax)

	def broadcast(self, fx: str, *args, **kwargs):
		for cb in self.crossbars:
			getattr(cb, fx)(*args, **kwargs)

	@property
	def std_noise(self):
		return self._std_noise

	@std_noise.setter
	def std_noise(self, std_noise):
		self._std_noise = std_noise
		self.param_update()

	@property
	def N(self):
		return self._N

	@N.setter
	def N(self, N):
		self._N = N
		self.param_update()

	@property
	def Gmax(self):
		return torch.cat([t.Gmax for t in self.crossbars])

	@Gmax.setter
	def Gmax(self, val):
		self.init_gmax(val)
  
	@property
	def Wmax(self):
		return torch.cat([t.Wmax for t in self.crossbars])

	def param_update(self):
		if self.quanted:
			self.unquant()
			self.quant(self._last_c_one)

	def __call__(self, input):
		return self.forward(input)

	def __del__(self):
		self.unquant()

	def forward(self, input):
		# WARNING: do not use this forward for learning, it does 2 forward with 1 batch
		#res_reliable = self.model(input).detach()
		#self.quant()
		self.renoise()
		res = self.model(input)
		#self.unquant()
		#self.MSE = torch.mean(torch.square(res - res_reliable))
		return res

	def init_gmax(self, Gmax):
		Gmax = torchize(Gmax)
		if self.wmax_mode == WMAX_MODE.ALL:
			assert Gmax.numel() == 1
			for t in self.crossbars:
				t.Gmax = Gmax.item()

		elif self.wmax_mode == WMAX_MODE.LAYERWISE:
			if Gmax.dim() == 0:
				Gmax = Gmax.repeat(len(self.crossbars))
			for t, g in zip(self.crossbars, Gmax.tolist()):
				t.Gmax = g

		elif self.wmax_mode == WMAX_MODE.COLUMNWISE:
			size_column = sum([t.out_features for t in self.crossbars])
			if Gmax.dim() == 0:
				Gmax = Gmax.repeat(size_column)
			assert Gmax.numel() == size_column and len(Gmax.shape) == 1, Gmax
			Gmax = torch.split(Gmax, [t.out_features for t in self.crossbars])
			for t, g in zip(self.crossbars, Gmax):
				t.Gmax = g

		else:
			raise ValueError('Not a supported wmax mode')
		#self._initial_Gmax = copy.deepcopy(Gmax)
		self.param_update()

	def init_wmax(self, wmax_mode):
		if isinstance(wmax_mode, str):
				wmax_mode = WMAX_MODE[wmax_mode.upper()]
		self.wmax_mode = wmax_mode
		self.param_update()

	@staticmethod
	def memory_usage(tensor):
		'''Return memory usage in MB'''
		return tensor.element_size() * tensor.nelement() / 1e6

	def memory_info(self):
		for i in range(len(self.saved_params)):
			print(f'Tensor {i} of shape {self.saved_params[i].shape}')
			print(f'Saved version dtype {self.saved_params[i].dtype} on {self.saved_params[i].device} (taking {self.memory_usage(self.saved_params[i])}) MB')
			print(f'Current version dtype {self.actual_params[i].dtype} on {self.actual_params[i].device} (taking {self.memory_usage(self.actual_params[i])}) MB')

	def quant(self, c_one=False):
		self._last_c_one = c_one
		self.broadcast('unquant')
		self.update_w_max()
		self.broadcast('quant', N=self.N, c_one=c_one)
		self.quanted = True

	def renoise(self):
		self.broadcast('renoise', std_noise=self.std_noise)

	def denoise(self):
		self.broadcast('denoise')

	def rescale(self):
		self.broadcast('rescale')

	def unquant(self):
		self.broadcast('unquant')
		self.quanted = False
		self.noised = False

	@torch.no_grad()
	def update_w_max(self) -> None:
		res = [t.update_w_max(self.wmax_mode) for t in self.crossbars]
		if self.wmax_mode == WMAX_MODE.ALL:
			res = max(res)
			for t in self.crossbars:
				t.Wmax = res
