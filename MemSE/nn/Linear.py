from __future__ import annotations
import math
import torch
import opt_einsum as oe

from typing import Optional, List
from MemSE.nn.Padder import Padder

from MemSE.nn.map import register_memse_mapping
from MemSE.nn.base_layer.MemSELayer import MemSELayer, MemSEReturn
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


def linear_montecarlo(linear, memse_dict):
	ct = linear.Gmax / linear.Wmax
	w_noise = torch.normal(mean=0., std=linear._crossbar.manager.std_noise, size=linear.weight.shape, device=linear.weight.device)
	w_noise_n = torch.normal(mean=0., std=linear._crossbar.manager.std_noise, size=linear.weight.shape, device=linear.weight.device)
	sign_w = torch.sign(linear.weight)
	abs_w: torch.Tensor = oe.contract('co,c->co', torch.abs(linear.weight), ct).to(linear.weight)
	Gpos = torch.clip(torch.where(sign_w > 0, abs_w, 0.) + w_noise, min=0)
	Gneg = torch.clip(torch.where(sign_w < 0, abs_w, 0.) + w_noise_n, min=0)

	# PRECOMPUTE CONVS
	zp_mu = torch.nn.functional.linear(memse_dict['mu'], Gpos)
	zm_mu = torch.nn.functional.linear(memse_dict['mu'], Gneg)
	
	# ENERGY
	sum_x_gd: torch.Tensor = memse_dict['mu'] ** 2
	e_p_mem: torch.Tensor = torch.sum(torch.nn.functional.linear(sum_x_gd, abs_w), dim=(0), keepdim=True)
	e_p_tiap = torch.sum(((zp_mu * memse_dict['r']) ** 2) /  memse_dict['r'], dim=(0), keepdim=True)
	e_p_tiam = torch.sum(((zm_mu * memse_dict['r']) ** 2) /  memse_dict['r'], dim=(0), keepdim=True)
	
	memse_dict['P_tot'] += e_p_mem + e_p_tiap + e_p_tiam
	memse_dict['current_type'] = 'Linear'
	memse_dict['mu'] = memse_dict['r'] * (torch.einsum(f'co,c -> co', zp_mu, 1/ct) - torch.einsum(f'co,c -> co', zm_mu, 1/ct))


@register_memse_mapping()
class Linear(MemSELayer):
	def initialize_from_module(self, linear_module) -> None:
		assert isinstance(linear_module, torch.nn.Linear)
		self.weight = linear_module.weight
		self._out_features = linear_module.out_features
		if linear_module.bias is not None:
			fused_linear = torch.nn.Linear(linear_module.weight.shape[1] + 1, linear_module.weight.shape[0], bias=False)
			biases = linear_module.bias.repeat_interleave((linear_module.weight.shape[0]//linear_module.bias.shape[0])).unsqueeze(1)
			self.weight = torch.nn.Parameter(torch.cat((linear_module.weight, biases), dim=1))
			self._out_features = fused_linear.out_features
			self.padder = Padder((0, 1), value=1., gamma_value=0.)

	def functional_base(self, x, weight:Optional[torch.Tensor] = None):
		if self.padder:
			x = self.padder(x)
		return torch.nn.functional.linear(x, weight=self.weight if weight is None else weight)

	@classmethod
	@property
	def dropin_for(cls):
		return set([torch.nn.Linear])

	@property
	def out_features(self):
		return self._out_features

	@property
	def memristored(self):
		return {'weight': self.weight}

	@property
	def memristored_einsum(self) -> dict:
		return {
			'weight': 'co,c->co',
			'out': 'bc,c->bc',
		}

	def memse(self, previous_layer: MemSEReturn, *args, **kwargs):
		x = previous_layer.out
		gamma = previous_layer.gamma
		gamma_shape = previous_layer.gamma_shape
		power = previous_layer.power

		if self.padder:
			x, gamma, gamma_shape, power = self.padder(x, gamma, gamma_shape, power)
		ct = self.Gmax/self.Wmax
		if ct.dim() == 0:
			ct = ct.expand(self.weight.shape[0]).to(x)
		sigma_c = self.std_noise * math.sqrt(2) / ct

		if power:
			sigma_p = self.std_noise / ct
			Gpos = torch.clip(self.weight, min=0)
			Gneg = torch.clip(-self.weight, min=0)

			new_mu_pos, new_gamma_pos, _ = linear_layer_vec_batched(x, gamma, Gpos, sigma_p, self.tia_resistance, gamma_shape=gamma_shape, gamma_only_diag=True)
			new_mu_neg, new_gamma_neg, _ = linear_layer_vec_batched(x, gamma, Gneg, sigma_p, self.tia_resistance, gamma_shape=gamma_shape, gamma_only_diag=True)

			power.add_(energy_vec_batched(ct, self.weight, gamma, x, new_gamma_pos, new_mu_pos, new_gamma_neg, new_mu_neg, self.tia_resistance, gamma_shape=gamma_shape))

		x, gamma, gamma_shape = linear_layer_vec_batched(x, gamma, self.weight, sigma_c, self.tia_resistance, gamma_shape=gamma_shape)
		return MemSEReturn(x, gamma, gamma_shape, power)
