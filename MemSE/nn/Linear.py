import math
import torch
import opt_einsum as oe

from typing import Optional, List
from .utils import diagonal_replace


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
def padded_mu_gamma(mu, gamma: torch.Tensor, padding:int=1, gamma_shape:Optional[List[int]]=None):
	batch_len = mu.shape[0]
	padded_size = [mu.shape[1], mu.shape[2]+padding*2, mu.shape[3]+padding*2]

	pad_mu = torch.nn.functional.pad(mu, ((padding,) * 4))
	numel_image = pad_mu.shape[1:].numel()
	mu = torch.reshape(pad_mu, (batch_len,numel_image))

	if gamma_shape is not None:# gamma == 0 store only size
		gamma_shape = [batch_len,numel_image,numel_image]
	else:
		pad_gamma = torch.nn.functional.pad(gamma, ((padding,) * 4 + (0, 0) + (padding,) * 4))
		gamma = torch.reshape(pad_gamma, (batch_len,numel_image,numel_image))

	return mu, gamma, gamma_shape


#@torch.jit.script
def energy_vec_batched(c, G, gamma:torch.Tensor, mu, new_gamma_pos_diag:torch.Tensor, new_mu_pos, new_gamma_neg_diag:torch.Tensor, new_mu_neg, r:float, gamma_shape:Optional[List[int]]=None):
	if gamma_shape is not None:
		mu_r = oe.contract('i,ij,bj->b', c, torch.abs(G), mu)
	else:
		diag_gamma = torch.diagonal(gamma, dim1=1, dim2=2)
		mu_r = oe.contract('i,ij,bj->b', c, torch.abs(G), diag_gamma+mu)

	#diag_ngp = torch.diagonal(new_gamma_pos_diag, dim1=1, dim2=2)
	#diag_ngn = torch.diagonal(new_gamma_neg_diag, dim1=1, dim2=2)
	diags =  (new_gamma_pos_diag + torch.square(new_mu_pos) + new_gamma_neg_diag + torch.square(new_mu_neg)) # (diag_ngp + torch.square(new_mu_pos) + diag_ngn + torch.square(new_mu_neg))
	return mu_r + oe.contract('i,bi->b',torch.square(c), diags)/r


#@torch.jit.script # see https://github.com/pytorch/pytorch/issues/49372
def linear_layer_logic(W, mu, gamma:torch.Tensor, Gmax, Wmax, sigma:float, r:float, gamma_shape:Optional[List[int]]=None, compute_power:bool = True):
	batch_len = mu.shape[0]
	image_shape = mu.shape[1:]
	l = mu.shape[2]
	

	if W.shape[1] != image_shape.numel():
		conv = True
		mu, gamma, gamma_shape = padded_mu_gamma(mu, gamma, gamma_shape=gamma_shape)
	else:
		conv=False
		mu = torch.reshape(mu, (batch_len, image_shape.numel()))
		if gamma_shape is not None: # gamma == 0 store only size
			gamma_shape = [batch_len,mu.shape[1],mu.shape[1]]
		else:
			gamma = torch.reshape(gamma,(batch_len,mu.shape[1],mu.shape[1]))

	ct = torch.ones(W.shape[0], device=mu.device, dtype=mu.dtype)*Gmax/Wmax # TODO columnwise validity ?
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

	if conv:
		mu = torch.reshape(mu, (batch_len,int(mu.numel()/batch_len//(l*l)),l,l))
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