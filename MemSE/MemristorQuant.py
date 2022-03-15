import torch
import torch.nn as nn
import numpy as np
import copy

__all__ = ['MemristorQuant']


class MemristorQuant(object):
	def __init__(self, model: nn.Module, types_handled = [nn.Linear], N: int = 100, wmax_mode='all', Gmax=0.1, std_noise:float=1.) -> None:
		super().__init__()
		self.model = model
		self.saved_params = []
		self.actual_params = []
		self.n_vars = []
		self.Wmax = []
		self.intermediate_params = {}
		self.c = {}
		self.wmax_mode = wmax_mode
		for m in model.modules():
			if type(m) in types_handled:
				self.saved_params.append(m.weight.data.clone().cpu())
				self.actual_params.append(m.weight)
				self.n_vars.append(m.out_features)
		for m in self.saved_params:
			self.Wmax.append(self._Wmax(m))
		self.quanted = False
		self.N = N

		if wmax_mode == 'network':
			if isinstance(Gmax, float):
				self.Gmax = np.full((len(self.saved_params)), Gmax)
			elif isinstance(Gmax, np.ndarray) and Gmax.size == 1:
				self.Gmax = np.full((len(self.saved_params)), Gmax[0])
			elif isinstance(Gmax, np.ndarray) and Gmax.size == len(self.saved_params):
				self.Gmax = np.copy(Gmax)
			else:
				raise ValueError('In network mode, expecting either float or ndarray of size 1')
		elif wmax_mode == 'layer':
			assert isinstance(Gmax, np.ndarray) and Gmax.size == len(self.saved_params)
			self.Gmax = np.copy(Gmax)
		elif wmax_mode == 'column':
			assert isinstance(Gmax, np.ndarray) and Gmax.size == sum(self.n_vars)
			self.Gmax = np.split(np.copy(Gmax), np.cumsum(self.n_vars))[:-1]
		else:
			raise ValueError('Not a supported wmax mode')
		assert len(self.Gmax) == len(self.saved_params), 'Gmax is not of the right size'

		self._initial_Gmax = copy.deepcopy(self.Gmax)
		self.std_noise = std_noise
		#print(f"Initialized memquant with {len(self.saved_params)} parameters quantified")

	def __call__(self, input):
		return self.forward(input)

	def __del__(self):
		self.unquant()

	def forward(self, input):
		# WARNING: do not use this forward for learning, it does 2 forward with 1 batch
		res_reliable = self.model(input).detach()
		self.quant()
		res = self.model(input)
		self.unquant()
		self.MSE = torch.mean(torch.square(res - res_reliable))
		return res

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
		if self.quanted:
			self.unquant()
		for i in range(len(self.saved_params)):
			true_value = self.actual_params[i].data
			self.saved_params[i].copy_(true_value.clone().cpu())
			self._quantize(true_value, i, c_one)
			self.intermediate_params[i] = true_value.clone().cpu()
			self.actual_params[i].data.copy_(true_value)
		self.quanted = True

	def renoise(self):
		for i, inter in self.intermediate_params.items():
			self.actual_params[i].data.copy_(self.intermediate_params[i])
			self.actual_params[i].data += torch.normal(mean=0., std=self.std_noise, size=self.actual_params[i].shape, device=self.actual_params[i].device)
			self.actual_params[i].data -= torch.normal(mean=0., std=self.std_noise, size=self.actual_params[i].shape, device=self.actual_params[i].device)
			if self.wmax_mode in ['network', 'layer']:
				self.actual_params[i].data /= self.c[i]
			else:
				self.actual_params[i].data.copy_(torch.einsum('ij,i -> ij', self.actual_params[i].data, 1/self.c[i]))

	def unquant(self):
		if self.quanted:
			for i in range(len(self.saved_params)):
				self.actual_params[i].data.copy_(self.saved_params[i].to(self.actual_params[i].data))
		self.quanted = False

	@torch.no_grad()
	def _quantize(self, tensor, layer_idx=None, c_one=False) -> None:
		Wmax = self._Wmax(tensor, layer_idx)
		if c_one:
			c = 1.
			if self.wmax_mode in ['network', 'layer']:
				self.Gmax[layer_idx] = Wmax
			else:
				self.Gmax[layer_idx] = self.Wmax[layer_idx] # fill in place
		Gmax = torch.from_numpy(self.Gmax[layer_idx]).to(tensor) if self.wmax_mode == 'column' else self.Gmax[layer_idx]
		if not c_one:
			c = Gmax / Wmax
		if layer_idx is not None:
			self.c[layer_idx] = c
		delta = Gmax / self.N

		if self.wmax_mode in ['network', 'layer']:
			tensor.copy_((torch.floor(tensor * c / delta)) * delta)
		else:
			tensor.copy_(torch.einsum('ij,i -> ij', torch.floor(torch.einsum('ij,i -> ij', tensor, (c/delta))), delta))

	@torch.no_grad()
	def _Wmax(self, tensor, layer_idx=None):
		assert len(tensor.shape) == 2, 'Only works for 2d tensors !'
		if self.wmax_mode == 'network':
			res = max([torch.max(torch.abs(t)) for t in self.saved_params])
		elif self.wmax_mode == 'layer':
			res= torch.max(torch.abs(tensor))
		elif self.wmax_mode == 'column':
			res = torch.max(torch.abs(tensor), dim=1).values
		if layer_idx is not None:
			self.Wmax[layer_idx] = res
		return res