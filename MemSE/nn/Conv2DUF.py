from __future__ import annotations
import torch
import torch.nn as nn

from .utils import double_conv, energy_vec_batched, gamma_to_diag, padded_mu_gamma

class Conv2DUF(nn.Module):
	def __init__(self, conv, input_shape, output_shape):
		super().__init__()
		assert len(output_shape) == 3, f'chw or cwh with no batch dim'
		self.c = conv
		self.output_shape = output_shape
		#exemplar = self.unfold_input(torch.rand(input_shape))
		self.original_weight = conv.weight.detach().clone()
		self.weight = self.original_weight.view(self.c.weight.size(0), -1).t()
		#self.weight = torch.repeat_interleave(self.weight, exemplar.shape[-1], dim=0)
		self.bias = conv.bias.detach().clone()

	def forward(self, x):
		inp_unf = self.unfold_input(x)
		#out_unf = torch.einsum('bfp,pfc->bcp', inp_unf, self.weight)
		out_unf = inp_unf.transpose(1, 2).matmul(self.weight).transpose(1, 2)
		out = out_unf.view(*self.output_shape)
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
		return self.c.groups

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
		mu, gamma, gamma_shape = padded_mu_gamma(memse_dict['mu'], memse_dict['gamma'], gamma_shape=memse_dict['gamma_shape'])
		# TODO don't need these if using convolutions all the way (no need for prepad + reshape)
		ct = torch.ones(conv2duf.weight.shape[0], device=mu.device, dtype=mu.dtype) * conv2duf.weight.learnt_Gmax / conv2duf.weight.Wmax

		if memse_dict['compute_power']:
			Gpos = torch.clip(conv2duf.original_weight, min=0)
			Gneg = torch.clip(-conv2duf.original_weight, min=0)
			new_mu_pos, new_gamma_pos, _ = Conv2DUF.mse_var(conv2duf, memse_dict, Gpos)
			new_mu_neg, new_gamma_neg, _ = Conv2DUF.mse_var(conv2duf, memse_dict, Gneg)

			P_tot = energy_vec_batched(ct, conv2duf.weight, gamma, mu, new_gamma_pos, new_mu_pos, new_gamma_neg, new_mu_neg, memse_dict['r'], gamma_shape=memse_dict['gamma_shape'])
		else:
			P_tot = 0.

		mu, gamma, gamma_shape = Conv2DUF.mse_var(conv2duf, memse_dict, conv2duf.original_weight)

		mu = torch.reshape(mu, (batch_len,int(mu.numel()/batch_len//(l*l)),l,l))
		if gamma_shape is not None:
			gamma_shape = [batch_len,*mu.shape[1:],*mu.shape[1:]]
		else:
			gamma = torch.reshape(gamma, (batch_len,*mu.shape[1:],*mu.shape[1:]))
		
		memse_dict['P_tot'] += P_tot
		memse_dict['current_type'] = 'Conv2DUF'
		memse_dict['mu'] = mu
		memse_dict['gamma'] = gamma
		memse_dict['gamma_shape'] = gamma_shape

	@staticmethod
	def mse_var(conv2duf: Conv2DUF, memse_dict, c, weights):
		mu = conv2duf(memse_dict['mu']) * memse_dict['r']
		gamma = memse_dict['gamma']
		gamma_diag = gamma_to_diag(gamma)
		first_comp = (gamma_diag + memse_dict['mu'] ** 2) * memse_dict['sigma'] ** 2 / c ** 2
		first_comp = nn.functional.conv2d(first_comp, torch.ones_like(weights), **conv2duf.conv_property_dict)
		conv_sq = weights ** 2
		first_comp += nn.functional.conv2d(gamma_diag, conv_sq, **conv2duf.conv_property_dict)

		
		gamma = double_conv(gamma, weights, **conv2duf.conv_property_dict)
		return mu, gamma, None
