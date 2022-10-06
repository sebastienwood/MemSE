import math
import torch
import opt_einsum as oe

from typing import Optional, List
from .utils import diagonal_replace, energy_vec_batched, padded_mu_gamma


#@torch.jit.script
def linear_layer_vec_batched(mu, gamma: torch.Tensor, G, sigma_c, r:float, gamma_shape:Optional[List[int]]=None, gamma_only_diag:bool=False):
	# TODO work in diagonal mode () / symmetric matrix storage for gamma ?
	new_gamma = torch.zeros(0)
	new_mu = r * oe.contract('ij,bj->bi', G, mu)

	#TODO marginal opt if sigma_c_t is a unique value
	sigma_c_sq = torch.square(sigma_c)
	mu_sq = torch.square(mu)

	gg = oe.contract('bij->bi', oe.contract('bij,i->bij', mu_sq.unsqueeze(dim=1).expand((mu.shape[0],G.shape[0],)+mu.shape[1:]), sigma_c_sq)) #1st term

	if gamma_shape is not None and not gamma_only_diag:
		new_gamma = torch.zeros(gamma_shape[0],G.shape[0],G.shape[0], dtype=mu.dtype, device=mu.device)
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
	batch_len = mu.shape[0]
	image_shape = mu.shape[1:]

	###
	# Should not be managed here
	###
	if hasattr(W, '__padding'):
		mu, gamma, gamma_shape = padded_mu_gamma(mu, gamma, gamma_shape=gamma_shape, padding=W.__padding)
	else:
		mu = torch.reshape(mu, (batch_len, image_shape.numel()))
		if gamma_shape is not None: # gamma == 0 store only size
			gamma_shape = [batch_len,mu.shape[1],mu.shape[1]]
		else:
			gamma = torch.reshape(gamma,(batch_len,mu.shape[1],mu.shape[1]))

	if hasattr(W, '__bias'):
		mu = torch.nn.functional.pad(mu, (0, 1), value=1.)
		if gamma_shape is not None:
			gamma_shape = [batch_len,mu.shape[1]+1,mu.shape[1]+1]
		else:
			gamma = torch.nn.functional.pad(gamma, (0, 1, 0, 1))
	###
	#
	###

	ct = torch.ones(W.shape[0], device=mu.device, dtype=mu.dtype)*Gmax.to(mu.device)/Wmax.to(mu.device) # TODO automated test for columnwise validity
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

	if hasattr(W, '__padding'):
		mu = torch.reshape(mu, (batch_len,) + W.__output_shape[1:])
		if gamma_shape is not None:
			gamma_shape = [batch_len,*mu.shape[1:],*mu.shape[1:]]
		else:
			gamma = torch.reshape(gamma, (batch_len,*mu.shape[1:],*mu.shape[1:]))
	return mu, gamma, P_tot, gamma_shape


def linear(module, data):
	x, gamma, P_tot_i, gamma_shape = linear_layer_logic(module.weight,
														data['mu'],
														data['gamma'],
														module.weight.learnt_Gmax,
														module.weight.Wmax,
														data['sigma'],
														data['r'],
														data['gamma_shape'],
														compute_power=data['compute_power'])
	data['P_tot'] += P_tot_i
	data['current_type'] = 'Linear'
	data['mu'] = x
	data['gamma'] = gamma
	data['gamma_shape'] = gamma_shape