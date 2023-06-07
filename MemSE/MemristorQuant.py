import torch
import torch.nn as nn
from MemSE.definitions import WMAX_MODE
from MemSE.utils import default, torchize
from MemSE.quant import CrossBar

from typing import Union

__all__ = ['MemristorQuant']


class MemristorQuant(object):
	def __init__(self,
				 model: nn.Module,
				 N: int = 128,
				 wmax_mode:Union[str, WMAX_MODE] = WMAX_MODE.LAYERWISE,
				 Gmax=0.1,
				 std_noise:float=1.,
     			 tia_resistance:float=1.) -> None:
		super().__init__()
		self.quanted = False
		self.noised = False
		self.N = N
		self.std_noise = std_noise
		self.tia_resistance = tia_resistance
		Gmax = default(Gmax, 0.1)
		self.crossbars = []
		for m in model.modules():
			for att in vars(m):
				if isinstance(getattr(m, att), CrossBar):
					self.crossbars.append(getattr(m, att))
					getattr(m, att).manager = self
		assert len(self.crossbars) > 0
		self.init_wmax(wmax_mode)
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
		if self.wmax_mode == WMAX_MODE.ALL:
			res = torch.cat([t.Wmax for t in self.crossbars]).unique_consecutive()
			assert res.numel() == 1
			return res
		elif self.wmax_mode == WMAX_MODE.LAYERWISE:
			res = torch.cat([t.Wmax.unique_consecutive() for t in self.crossbars])
			assert res.numel() == len(self.crossbars)
			return res
		elif self.wmax_mode == WMAX_MODE.COLUMNWISE:
			return [t.Wmax for t in self.crossbars]
		else:
			raise ValueError('Not a valid Wmax mode')

	def param_update(self):
		if self.quanted:
			self.quant(self._last_c_one)

	def __call__(self, input):
		return self.forward(input)

	def __del__(self):
		self.unquant()

	def init_gmax(self, Gmax):
		Gmax = torchize(Gmax)
		Gmax.clamp_(0.)
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
		self.update_w_max()
		self.param_update()

	def init_gmax_as_wmax(self):
		for cb in self.crossbars:
			cb.Gmax = cb.Wmax

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
		if len(self.crossbars) == 0:
			return
		res = [t.update_w_max(self.wmax_mode) for t in self.crossbars]
		if self.wmax_mode == WMAX_MODE.ALL:
			res = max(res)
			for t in self.crossbars:
				t.Wmax = res
