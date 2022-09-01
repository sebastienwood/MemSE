from __future__ import annotations
import math
import torch
import torch.nn as nn
import numpy as np
from numba import njit, prange

from .op import conv2duf_op
from .utils import double_conv, energy_vec_batched, gamma_add_diag, gamma_to_diag, padded_mu_gamma

class Conv2DUF(nn.Module):
	def __init__(self, conv, input_shape, output_shape):
		super().__init__()
		assert len(output_shape) == 3, f'chw or cwh with no batch dim ({output_shape})'
		self.c = conv
		self.output_shape = output_shape

		self.register_parameter('original_weight', nn.Parameter(conv.weight.detach().clone()))
		self.weight = self.original_weight.view(self.c.weight.size(0), -1).t()
		if conv.bias is not None:
			self.register_parameter('bias', nn.Parameter(conv.bias.detach().clone()))
		else:
			self.bias = None

	def forward(self, x):
		# inp_unf = self.unfold_input(x)
		# out_unf = inp_unf.transpose(1, 2).matmul(self.weight.to(x)).transpose(1, 2)
		# out = out_unf.view(x.shape[0], *self.output_shape)
		# if self.bias is not None:
		# 	out += self.bias[:, None, None]
		# return out
		return torch.nn.functional.conv2d(x, self.weight.t().reshape(self.original_weight.shape), bias=self.bias, **self.conv_property_dict)

	def _mse_var(self, conv2duf, memse_dict, ct, w, sigma):
		if self.__slow == True:
			return self.slow_mse_var(conv2duf, memse_dict, ct, w, sigma)
		else:
			return self.mse_var(conv2duf, memse_dict, ct, w, sigma)

	@property
	def kernel_size(self):
		return self.c.kernel_size

	@property
	def padding(self):
		return self.c.padding

	@property
	def dilation(self):
		return self.c.dilation

	@property
	def groups(self):
		return int(self.c.groups)

	@property
	def stride(self):
		return self.c.stride

	@property
	def out_features(self):
		return self.output_shape.numel()

	@property
	def conv_property_dict(self):
		return {
			'stride': self.stride,
			'padding': self.padding,
			'dilation': self.dilation,
			'groups': self.groups
		}

	def unfold_input(self, x):
		return nn.functional.unfold(x, self.c.kernel_size, self.c.dilation, self.c.padding, self.c.stride)

	@staticmethod
	def memse(conv2duf: Conv2DUF, memse_dict):
		mu, gamma, gamma_shape = memse_dict['mu'], memse_dict['gamma'], memse_dict['gamma_shape']
		ct = conv2duf.weight.learnt_Gmax / conv2duf.weight.Wmax  # Only one column at most (vector of weights)

		if memse_dict['compute_power']:
			Gpos = torch.clip(conv2duf.original_weight, min=0)
			Gneg = torch.clip(-conv2duf.original_weight, min=0)
			new_mu_pos, new_gamma_pos, _ = conv2duf._mse_var(conv2duf, memse_dict, ct, Gpos, memse_dict['sigma'])
			new_mu_neg, new_gamma_neg, _ = conv2duf._mse_var(conv2duf, memse_dict, ct, Gneg, memse_dict['sigma'])

			P_tot = energy_vec_batched(ct, conv2duf.weight, gamma, mu, new_gamma_pos, new_mu_pos, new_gamma_neg, new_mu_neg, memse_dict['r'], gamma_shape=memse_dict['gamma_shape'])
		else:
			P_tot = 0.

		mu, gamma, gamma_shape = conv2duf._mse_var(conv2duf, memse_dict, ct, conv2duf.original_weight, memse_dict['sigma'] * math.sqrt(2))

		memse_dict['P_tot'] += P_tot
		memse_dict['current_type'] = 'Conv2DUF'
		memse_dict['mu'] = mu
		memse_dict['gamma'] = gamma
		memse_dict['gamma_shape'] = gamma_shape

	@staticmethod
	def mse_var(conv2duf: Conv2DUF, memse_dict, c, weights, sigma):
		mu = conv2duf(memse_dict['mu']) * memse_dict['r']
		gamma = memse_dict['gamma'] if memse_dict['gamma_shape'] is None else torch.zeros(memse_dict['gamma_shape'], device=mu.device, dtype=mu.dtype)

		c0 = sigma ** 2 / c ** 2

		gamma_n = double_conv(gamma, weights, **conv2duf.conv_property_dict)
		gamma_n = conv2duf_op(gamma_n, gamma, memse_dict['mu'], c0, weight_shape=weights.shape, **conv2duf.conv_property_dict)
		gamma_n *= memse_dict['r'] ** 2
		return mu, gamma_n, None
