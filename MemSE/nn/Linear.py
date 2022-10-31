from __future__ import annotations
import math
import torch
import opt_einsum as oe

from typing import Optional, List

from MemSE.nn.MemSELayer import MemSELayer
from .utils import diagonal_replace, energy_vec_batched


#@torch.jit.script
def linear_layer_vec_batched(mu, gamma: torch.Tensor, G, sigma_c, r:float, gamma_shape:Optional[List[int]]=None, gamma_only_diag:bool=False):
	# TODO work in diagonal mode () / symmetric matrix storage for gamma ?
	new_gamma = torch.zeros(0)
	new_mu = r * oe.contract('ij,bj->bi', G, mu)

	sigma_c_sq = torch.square(sigma_c)
	mu_sq = torch.square(mu)

	gg = oe.contract('bij->bi', oe.contract('bij,i->bij', mu_sq.unsqueeze(dim=1).expand((mu.shape[0],G.shape[0],)+mu.shape[1:]), sigma_c_sq)) #1st term
	if gamma_shape is not None and not gamma_only_diag:
		new_gamma = torch.zeros(gamma_shape[0],G.shape[0],G.shape[0], dtype=gamma.dtype, device=mu.device)
		gamma_shape = None
	elif gamma_shape is None:
		if not gamma_only_diag:
			new_gamma = oe.contract('ij,bjk,kl->bil', r ** 2 * G, gamma, G.T)
		diag_gamma = torch.diagonal(gamma, dim1=1, dim2=2).clone()
		torch.diagonal(gamma, dim1=1, dim2=2).data.fill_(0)
		gg += oe.contract('ij,bjk,ik->bi', G, gamma, G)
		gg += oe.contract('bi,ji->bj', diag_gamma, torch.square(G)) #2nd term
		gg += oe.contract('bij,i->bi', diag_gamma.unsqueeze(dim=1).expand((diag_gamma.shape[0], G.shape[0],)+diag_gamma.shape[1:]), sigma_c_sq) # 3rd term
		torch.diagonal(gamma, dim1=1, dim2=2).data.copy_(diag_gamma) # ensure inplace didn't change original input

	gg *= r ** 2
	if not gamma_only_diag:
		#torch.diagonal(new_gamma, dim1=1, dim2=2).copy_(gg)
		new_gamma = diagonal_replace(new_gamma, gg)
	else:
		new_gamma = gg

	return new_mu, new_gamma, gamma_shape


#@torch.jit.script # see https://github.com/pytorch/pytorch/issues/49372
def linear_layer_logic(W:torch.Tensor,
					   mu:torch.Tensor,
					   gamma:torch.Tensor,
					   Gmax,
					   Wmax,
					   sigma:float,
					   r:float,
					   gamma_shape:Optional[List[int]]=None,
					   compute_power:bool = True):

	ct = Gmax/Wmax
	if ct.dim() == 0:
		ct = ct.expand(W.shape[0]).to(mu)
	sigma_p = sigma / ct
	sigma_c = sigma * math.sqrt(2) / ct

	if compute_power:
		Gpos = torch.clip(W, min=0)
		Gneg = torch.clip(-W, min=0)

		new_mu_pos, new_gamma_pos, _ = linear_layer_vec_batched(mu, gamma, Gpos, sigma_p, r, gamma_shape=gamma_shape, gamma_only_diag=True)
		new_mu_neg, new_gamma_neg, _ = linear_layer_vec_batched(mu, gamma, Gneg, sigma_p, r, gamma_shape=gamma_shape, gamma_only_diag=True)

		P_tot = energy_vec_batched(ct, W, gamma, mu, new_gamma_pos, new_mu_pos, new_gamma_neg, new_mu_neg, r, gamma_shape=gamma_shape)
	else:
		P_tot = 0.

	mu, gamma, gamma_shape = linear_layer_vec_batched(mu, gamma, W, sigma_c, r, gamma_shape=gamma_shape)
	return mu, gamma, P_tot, gamma_shape


def linear(linear, data):
	x, gamma, P_tot_i, gamma_shape = linear_layer_logic(linear.weight,
														data['mu'],
														data['gamma'],
														linear.Gmax,
														linear.Wmax,
														data['sigma'],
														data['r'],
														data['gamma_shape'],
														compute_power=data['compute_power'])
	x.extra_info = 'Mu out of linear'
	gamma.extra_info = 'Gamma out of linear'
	data['P_tot'] += P_tot_i
	data['current_type'] = 'Linear'
	data['mu'] = x
	data['gamma'] = gamma
	data['gamma_shape'] = gamma_shape


class Linear(MemSELayer):
	def __init__(self, linear) -> None:
		super().__init__()
		self.linear = linear
		assert linear.bias is None

	def forward(self, x):
		return self.linear(x)

	@property
	def out_features(self):
		return self.linear.out_features

	@property
	def memristored(self):
		return ['linear.weight']

	@staticmethod
	def memse(linear: Linear, memse_dict):
		x, gamma, P_tot_i, gamma_shape = linear_layer_logic(linear.weight,
															memse_dict['mu'],
															memse_dict['gamma'],
															linear.Gmax,
															linear.Wmax,
															memse_dict['sigma'],
															memse_dict['r'],
															memse_dict['gamma_shape'],
															compute_power=memse_dict['compute_power'])
		memse_dict['P_tot'] += P_tot_i
		memse_dict['current_type'] = 'Linear'
		memse_dict['mu'] = x
		memse_dict['gamma'] = gamma
		memse_dict['gamma_shape'] = gamma_shape