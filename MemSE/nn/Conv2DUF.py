from __future__ import annotations
import math
import torch
import torch.nn as nn
import numpy as np

from .op import conv2duf_op
from .utils import double_conv, energy_vec_batched, gamma_add_diag, gamma_to_diag, padded_mu_gamma

class Conv2DUF(nn.Module):
	def __init__(self, conv, input_shape, output_shape, slow: bool = False):
		super().__init__()
		assert len(output_shape) == 3, f'chw or cwh with no batch dim ({output_shape})'
		self.c = conv
		self.__slow = slow

		#g = 2
		#t = torch.nn.Conv2d(conv.weight.shape[1]*g, conv.weight.shape[0]*g, conv.weight.shape[2], conv.stride, conv.padding, conv.dilation, groups=g, bias=False)
		#print(conv.weight.shape)
		#print(t.weight.shape)

		self.output_shape = output_shape
		#exemplar = self.unfold_input(torch.rand(input_shape))
		self.original_weight = conv.weight.detach().clone()
		self.weight = self.original_weight.view(self.c.weight.size(0), -1).t()
		#self.weight = torch.repeat_interleave(self.weight, exemplar.shape[-1], dim=0)
		self.bias = conv.bias.detach().clone() if conv.bias is not None else None

	def change_impl(self, slow: bool = False):
		self.__slow = slow

	def forward(self, x):
		inp_unf = self.unfold_input(x)
		#out_unf = torch.einsum('bfp,pfc->bcp', inp_unf, self.weight)
		out_unf = inp_unf.transpose(1, 2).matmul(self.weight).transpose(1, 2)
		out = out_unf.view(x.shape[0], *self.output_shape)
		if self.bias is not None:
			out += self.bias[:, None, None]
		return out

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
		#batch_len = memse_dict['mu'].shape[0]
		#image_shape = memse_dict['mu'].shape[1:]
		#l = memse_dict['mu'].shape[2]
		#mu, gamma, gamma_shape = padded_mu_gamma(memse_dict['mu'], memse_dict['gamma'], gamma_shape=memse_dict['gamma_shape'])
		# TODO don't need these if using convolutions all the way (no need for prepad + reshape)
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

		#mu = torch.reshape(mu, (batch_len,int(mu.numel()/batch_len//(l*l)),l,l))

		#if gamma_shape is not None:
		#	gamma_shape = [batch_len,*mu.shape[1:],*mu.shape[1:]]
		#else:
		#	gamma = torch.reshape(gamma, (batch_len,*mu.shape[1:],*mu.shape[1:]))
		
		memse_dict['P_tot'] += P_tot
		memse_dict['current_type'] = 'Conv2DUF'
		memse_dict['mu'] = mu
		memse_dict['gamma'] = gamma
		memse_dict['gamma_shape'] = gamma_shape

	@staticmethod
	def mse_var(conv2duf: Conv2DUF, memse_dict, c, weights, sigma):
		mu = conv2duf(memse_dict['mu']) * memse_dict['r']
		gamma = memse_dict['gamma'] if memse_dict['gamma_shape'] is None else torch.zeros(memse_dict['gamma_shape'])

		gamma_diag = gamma_to_diag(gamma)
		c0 = sigma ** 2 / c ** 2
		first_comp = (gamma_diag + memse_dict['mu'] ** 2) * c0
		first_comp = nn.functional.conv2d(input=first_comp, weight=torch.ones_like(weights), bias=None, **conv2duf.conv_property_dict)
		#first_comp += nn.functional.conv2d(input=gamma_diag, weight=weights ** 2, bias=None, **conv2duf.conv_property_dict)

		gamma_n = double_conv(gamma, weights, **conv2duf.conv_property_dict)
		## working temp removed gamma_add_diag(gamma, first_comp)

		gamma = conv2duf_op(gamma_n, gamma, memse_dict['mu'], c0, weight_shape=weights.shape, stride=conv2duf.stride)
		gamma = gamma * memse_dict['r'] ** 2
		return mu, gamma, None

	@staticmethod
	def slow_mse_var(conv2duf: Conv2DUF, memse_dict, c, weights, sigma):
		'''A reliable but slow version of mse_var'''
		gamma = memse_dict['gamma'] if memse_dict['gamma_shape'] is None else torch.zeros(memse_dict['gamma_shape'])
		mu_res = (conv2duf(memse_dict['mu']) * memse_dict['r']).cpu().numpy()
		mu, gamma, _ = padded_mu_gamma(memse_dict['mu'], gamma, gamma_shape=None, square_reshape=False, padding=conv2duf.padding)
		mu, gamma = mu.cpu().numpy(), gamma.cpu().numpy()
		w = weights.cpu().numpy()
		#k_ = w.shape[2] / 2
		gamma_res = np.zeros(mu_res.shape + mu_res.shape[1:])
		r_2 = memse_dict['r'] ** 2
		ratio = (sigma ** 2 / c ** 2).cpu().detach().numpy()
		if np.ndim(ratio)==0:
			ratio = np.repeat(ratio, gamma_res.shape[1])

		conv2duf.inner_loop(gamma_res, ratio, w, mu, gamma)
									
		gamma_res *= r_2
		return torch.from_numpy(mu_res), torch.from_numpy(gamma_res), None

	@staticmethod
	def inner_loop(gamma_res, ratio, w, mu, gamma):
		for bi in range(gamma_res.shape[0]):
			for c0 in range(gamma_res.shape[1]):
				for i0 in range(gamma_res.shape[2]):
					for j0 in range(gamma_res.shape[3]):
						for c0p in range(gamma_res.shape[4]):
							for i0p in range(gamma_res.shape[5]):
								for j0p in range(gamma_res.shape[6]):
									# DOUBLE CONV
									for ci in range(w.shape[1]):
										for ii in range(w.shape[2]):
											for ji in range(w.shape[3]):
												if c0 == c0p:
													gamma_res[bi, c0, i0, j0, c0p, i0p, j0p] += ratio[c0] * (mu[bi, ci, i0+ii, j0+ji] * mu[bi, ci, i0p+ii, j0p+ji] + gamma[bi, ci, i0+ii, j0+ji, ci, i0p+ii, j0p+ji])
												for cj in range(w.shape[1]):
													for ij in range(w.shape[2]):
														for jj in range(w.shape[3]):
															gamma_res[bi, c0, i0, j0, c0p, i0p, j0p] += w[c0,ci,ii,ji] * w[c0p, cj, ij, jj] * gamma[bi, ci, i0+ii, j0+ji, cj, i0p+ij, j0p+jj]
									
						# DIAGONALE == VAR
						gamma_res[bi, c0, i0, j0, c0, i0, j0] = 0
						for ci in range(w.shape[1]):
							for ii in range(w.shape[2]):
								for ji in range(w.shape[3]):
									g_2 = gamma[bi, ci, i0+ii, j0+ji, ci, i0+ii, j0+ji]
									gamma_res[bi, c0, i0, j0, c0, i0, j0] += ratio[c0] * (mu[bi, ci, i0+ii, j0+ji]**2 + g_2) + g_2 * w[c0, ci, ii, ji] ** 2
									for cj in range(w.shape[1]):
										for ij in range(w.shape[2]):
											for jj in range(w.shape[3]):
												if ci != cj or ii != ij or ji != jj:
													gamma_res[bi, c0, i0, j0, c0, i0, j0] += w[c0,ci,ii,ji] * w[c0, cj, ij, jj] * gamma[bi, ci, i0+ii, j0+ji, cj, i0+ij, j0+jj]
