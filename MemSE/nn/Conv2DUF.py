from __future__ import annotations
from numpy import diag
import torch
import torch.nn as nn
import numpy as np

from .utils import double_conv, energy_vec_batched, gamma_add_diag, gamma_to_diag, padded_mu_gamma

class Conv2DUF(nn.Module):
	def __init__(self, conv, input_shape, output_shape):
		super().__init__()
		assert len(output_shape) == 3, f'chw or cwh with no batch dim'
		self.c = conv

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

	def forward(self, x):
		inp_unf = self.unfold_input(x)
		#out_unf = torch.einsum('bfp,pfc->bcp', inp_unf, self.weight)
		out_unf = inp_unf.transpose(1, 2).matmul(self.weight).transpose(1, 2)
		out = out_unf.view(x.shape[0], *self.output_shape)
		if self.bias is not None:
			out += self.bias[:, None, None]
		return out

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
		batch_len = memse_dict['mu'].shape[0]
		image_shape = memse_dict['mu'].shape[1:]
		l = memse_dict['mu'].shape[2]
		#mu, gamma, gamma_shape = padded_mu_gamma(memse_dict['mu'], memse_dict['gamma'], gamma_shape=memse_dict['gamma_shape'])
		# TODO don't need these if using convolutions all the way (no need for prepad + reshape)
		mu, gamma, gamma_shape = memse_dict['mu'], memse_dict['gamma'], memse_dict['gamma_shape']
		ct = conv2duf.weight.learnt_Gmax / conv2duf.weight.Wmax # Only one column at most (vector of weights)

		if memse_dict['compute_power']:
			Gpos = torch.clip(conv2duf.original_weight, min=0)
			Gneg = torch.clip(-conv2duf.original_weight, min=0)
			new_mu_pos, new_gamma_pos, _ = Conv2DUF.mse_var(conv2duf, memse_dict, ct, Gpos)
			new_mu_neg, new_gamma_neg, _ = Conv2DUF.mse_var(conv2duf, memse_dict, ct, Gneg)

			P_tot = energy_vec_batched(ct, conv2duf.weight, gamma, mu, new_gamma_pos, new_mu_pos, new_gamma_neg, new_mu_neg, memse_dict['r'], gamma_shape=memse_dict['gamma_shape'])
		else:
			P_tot = 0.

		mu, gamma, gamma_shape = Conv2DUF.mse_var(conv2duf, memse_dict, ct, conv2duf.original_weight)

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
	def mse_var(conv2duf: Conv2DUF, memse_dict, c, weights):
		mu = conv2duf(memse_dict['mu']) * memse_dict['r']
		gamma = memse_dict['gamma'] if memse_dict['gamma_shape'] is None else torch.zeros(memse_dict['gamma_shape'])

		gamma_diag = gamma_to_diag(gamma)
		first_comp = (gamma_diag + memse_dict['mu'] ** 2) * memse_dict['sigma'] ** 2 / c ** 2
		first_comp = nn.functional.conv2d(input=first_comp, weight=torch.ones_like(weights), bias=None, **conv2duf.conv_property_dict)
		first_comp += nn.functional.conv2d(input=gamma_diag, weight=weights ** 2, bias=None, **conv2duf.conv_property_dict)
		
		gamma = double_conv(gamma, weights, **conv2duf.conv_property_dict)	
		gamma_add_diag(gamma, first_comp)
		gamma = gamma * memse_dict['r'] ** 2
		return mu, gamma, None

	@staticmethod
	def slow_mse_var(conv2duf: Conv2DUF, memse_dict, c, weights):
		'''A reliable but slow version of mse_var'''
		gamma = memse_dict['gamma'] if memse_dict['gamma_shape'] is None else torch.zeros(memse_dict['gamma_shape'])
		mu_res = (conv2duf(memse_dict['mu']) * memse_dict['r']).cpu().numpy()
		mu, gamma, _ = padded_mu_gamma(memse_dict['mu'], memse_dict['gamma'], gamma_shape=None)
		mu, gamma = mu.cpu().numpy(), gamma.cpu().numpy()
		w = conv2duf.c.weight.cpu().numpy()
		k_ = w.shape[2]
		gamma_res = np.zeros(mu.shape+mu.shape[1:])
		r_2 = memse_dict['r'] ** 2
		ratio = (memse_dict['sigma'] ** 2 / c ** 2).cpu().numpy()

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
												for cj in range(w.shape[1]):
													for ij in range(w.shape[2]):
														for jj in range(w.shape[3]):
															gamma_res[bi, c0, i0, j0, c0p, i0p, j0p] += w[c0,ci,ii,ji] * w[c0p, cj, ij, jj] * gamma[bi, ci, i0+ii-k_, j0+ji-k_, cj, i0p+ij-k_, j0p+jj-k_]
									
						# DIAGONALE == VAR
						for ci in range(w.shape[1]):
							for ii in range(w.shape[2]):
								for ji in range(w.shape[3]):
									g_2 = gamma[bi, ci, i0+ii, j0+ji, ci, i0+ii, j0+ji]**2
									gamma_res[bi, c0, i0, j0, c0, i0, j0] = ratio[c0] * (mu[bi, ci, i0+ii, j0+ji]**2 + g_2) + g_2 * w[c0, ci, ii, ji] ** 2
									for cj in range(w.shape[1]):
										for ij in range(w.shape[2]):
											for jj in range(w.shape[3]):
												if c0p != c0 or i0p != i0 or j0p != j0:
													gamma_res[bi, c0, i0, j0, c0, i0, j0] += w[c0,ci,ii,ji] * w[c0p, cj, ij, jj] * gamma[bi, ci, i0+ii-k_, j0+ji-k_, cj, i0p+ij-k_, j0p+jj-k_]

									
		gamma_res *= r_2
		return mu_res, gamma_res
