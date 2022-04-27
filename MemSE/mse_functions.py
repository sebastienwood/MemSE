import torch
import math
import torch.nn as nn
import numpy as np
import opt_einsum as oe
from pykeops.torch import LazyTensor

from typing import Optional, List
from MemSE.utils import net_param_iterator
from MemSE.MemristorQuant import MemristorQuant

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
		torch.diagonal(gamma, dim1=1, dim2=2).fill_(0.)
		gg += oe.contract('ij,bjk,ik->bi', G, gamma, G)
		gg += oe.contract('bi,ji->bj', diag_gamma, torch.square(G)) #2nd term
		gg += oe.contract('bij,i->bi', diag_gamma.unsqueeze(dim=1).expand((diag_gamma.shape[0], G.shape[0],)+diag_gamma.shape[1:]), sigma_c_sq) # 3rd term
		torch.diagonal(gamma, dim1=1, dim2=2).copy_(diag_gamma) # ensure inplace didn't change original input

	gg *= r ** 2
	if not gamma_only_diag:
		torch.diagonal(new_gamma, dim1=1, dim2=2).copy_(gg)
	else:
		new_gamma = gg

	return new_mu, new_gamma, gamma_shape

def sigmoid_coefs(order:int):
	c = np.zeros(order + 1, dtype=int)
	c[0] = 1
	for i in range(1, order + 1):
		for j in range(i, -1, -1):
			c[j] = -j * c[j - 1] + (j + 1) * c[j]
	return c

#@torch.jit.script
def dd_softplus(ten, beta:float=1, threshold:float=20, order: int = 2):
	x = ten * beta
	thresh_mask = (x < threshold).to(dtype=ten.dtype) * beta
	dsf = x.sigmoid()

	res = {1: dsf * thresh_mask,
		   2: dsf * (1 - dsf) * thresh_mask}
	if order > 2:
		for order_i in range(2, order + 1):
			c = torch.from_numpy(sigmoid_coefs(order_i)).to(dsf)
			ref = dsf * c[order_i]
			for i in range(order_i - 1, -1, -1):
				ref = dsf * (c[i] + ref)

			res[order_i + 1] = ref * thresh_mask

	return res[1], res[2], res

#@torch.jit.script
def softplus_vec_batched_old(mu, gamma:torch.Tensor, gamma_shape:Optional[List[int]]=None, beta:float=1, threshold:float=20):
	mu_r = torch.nn.functional.softplus(mu, beta=beta, threshold=threshold)
	d_mu, dd_mu, _ = dd_softplus(mu)

	if gamma_shape is not None:
		gamma = torch.zeros(gamma_shape[0],gamma_shape[1],gamma_shape[2],gamma_shape[3],gamma_shape[4],gamma_shape[5],gamma_shape[6], dtype=mu.dtype, device=mu.device)
		gamma_shape = None
		# TODO check if gamma init cannot be delegated further if its zero
		# TODO check if gamma is only diagonal elements to simplify first pass

	mu_r = mu_r + 0.5 * oe.contract('bcij,bcijcij->bcij', dd_mu, gamma)

	first_comp = oe.contract('bcij,bklm,bcijklm->bcijklm',d_mu,d_mu,gamma)
	second_comp = 0.25 * oe.contract('bcij,bklm->bcijklm', dd_mu, dd_mu)
	#second_comp.register_hook(lambda grad: print(grad.mean()))

	ga_r = first_comp - oe.contract('bcijklm,bcijcij,bklmklm->bcijklm', second_comp, gamma, gamma)

	return mu_r, ga_r, gamma_shape

def softplus_vec_batched(mu,
						  gamma:torch.Tensor,
						  gamma_shape:Optional[List[int]]=None,
						  degree_taylor:int=2,
						  beta:float=1,
						  threshold:float=20):
	mu_r = torch.nn.functional.softplus(mu, beta=beta, threshold=threshold)
	d_mu, dd_mu, softplus_d = dd_softplus(mu, order=6)

	if gamma_shape is not None:
		gamma = torch.zeros(gamma_shape[0],gamma_shape[1],gamma_shape[2],gamma_shape[3],gamma_shape[4],gamma_shape[5],gamma_shape[6], dtype=mu.dtype, device=mu.device)
		gamma_shape = None
		# TODO check if gamma init cannot be delegated further if its zero
		# TODO check if gamma is only diagonal elements to simplify first pass

	#second_comp.register_hook(lambda grad: print(grad.mean()))

	ga_r = oe.contract('bcij,bklm,bcijklm->bcijklm',d_mu,d_mu,gamma)
	second_comp = 0.25 * oe.contract('bcij,bklm->bcijklm', dd_mu, dd_mu)
	ga_r -= oe.contract('bcijklm,bcijcij,bklmklm->bcijklm', second_comp, gamma, gamma)

	ga_view = ga_r.view(ga_r.shape[0], ga_r.shape[1]*ga_r.shape[2]*ga_r.shape[3], -1)
	gg = ga_view.diagonal(dim1=1, dim2=2)
	print(gg.mean())
	gg.fill_(0.)
	print(ga_view.diagonal(dim1=1, dim2=2).mean())
	gg = gg.view(*ga_r.shape[:4])

	g_2 = oe.contract('bcijcij->bcij', gamma)
	if degree_taylor >= 2:
		mu_r += 0.5 * oe.contract('bcij,bcij->bcij', dd_mu, g_2)

		gg += oe.contract('bcij,bcij->bcij', softplus_d[1] ** 2, g_2)
	if degree_taylor >= 4:
		mu_r += 0.125 * oe.contract('bcij,bcij->bcij', softplus_d[4], g_2 ** 2)

		fourth_comp = 2 * (softplus_d[2] / 2) ** 2 + oe.contract('bcij,bcij->bcij', softplus_d[1], softplus_d[3])
		gg += oe.contract('bcij,bcij->bcij', fourth_comp, g_2 ** 2)
	if degree_taylor >= 6:
		mu_r += (15/720) * oe.contract('bcij,bcij->bcij', softplus_d[6], g_2 ** 4)

		six_comp = (softplus_d[3] / 6) ** 2 * 15
		six_comp += (1/4) * oe.contract('bcij,bcij->bcij', softplus_d[1], softplus_d[5])
		six_comp += (1/2) * oe.contract('bcij,bcij->bcij', softplus_d[2], softplus_d[4])
		gg += oe.contract('bcij,bcij->bcij', six_comp, g_2 ** 4)

	print(ga_view.diagonal(dim1=1, dim2=2).mean())
	return mu_r, ga_r, gamma_shape

@torch.jit.script
def avgPool2d_layer_vec_gamma_batched(gamma, kernel_size:int=2, stride:int=2, padding:int=0):
	bs = gamma.shape[0]
	nc = gamma.shape[1]
	img_shape = gamma.shape[1:4]
	ratio_sq = (1/(kernel_size ** 4))

	nice_view = gamma.reshape(bs, -1, img_shape[1], img_shape[2])
	nc_first = nice_view.shape[1]
	convolution_filter = torch.full((nc_first, 1, kernel_size, kernel_size), 1., device=gamma.device, dtype=gamma.dtype) # TODO maybe just a view is enough if it takes too much space
	first_res = torch.nn.functional.conv2d(nice_view, convolution_filter, stride=stride, padding=padding, groups=nc_first)

	first_res_shape = first_res.shape
	nice_view_res = first_res.view(bs, img_shape[0], img_shape[1], img_shape[2], nc, first_res_shape[2], first_res_shape[3])

	permuted = nice_view_res.permute(0, 4, 5, 6, 1, 2, 3)
	another_nice_view = permuted.reshape(bs, -1, img_shape[1], img_shape[2])
	nc_second = another_nice_view.shape[1]
	convolution_filter = torch.full((nc_second, 1, kernel_size, kernel_size), 1., device=gamma.device, dtype=gamma.dtype)
	second_res = torch.nn.functional.conv2d(another_nice_view, convolution_filter, stride=stride, padding=padding, groups=nc_second)

	second_res_shape = second_res.shape
	anv_res = second_res.view(bs, nc, first_res_shape[2], first_res_shape[3], nc, second_res_shape[2], second_res_shape[3])

	result = anv_res.permute(0, 4, 5, 6, 1, 2, 3)

	result *= ratio_sq
	return result


@torch.jit.script
def avgPool2d_layer_vec_mu_batched(mu, kernel_size:int=2, stride:int=2, padding:int=0):
	nc = mu.shape[1]
	ratio = (1/(kernel_size ** 2))

	convolution_filter = torch.full((nc, 1, kernel_size, kernel_size), ratio, device=mu.device, dtype=mu.dtype)
	new_mu =  torch.nn.functional.conv2d(mu, convolution_filter,stride=stride, padding=padding, groups=nc)
	return new_mu


@torch.jit.script
def avgPool2d_layer_vec_batched(mu, gamma:torch.Tensor, kernel_size:int=2, stride:int=2, padding:int=0, gamma_shape:Optional[List[int]]=None):
	Hin, Win = mu.shape[2], mu.shape[3]
	Hout = Wout = math.floor(1+(Hin+2*padding-kernel_size)/stride)
	nc = mu.shape[1]
	ratio = (1/(kernel_size ** 2))
	ratio_sq = ratio ** 2

	if gamma_shape is not None:
		gamma_shape = [mu.shape[0],nc,Hout,Wout,nc,Hout,Wout]
		new_gamma = gamma
	else:
		new_gamma = avgPool2d_layer_vec_gamma_batched(gamma, kernel_size, stride, padding)

	new_mu = avgPool2d_layer_vec_mu_batched(mu, kernel_size, stride, padding)
	return new_mu, new_gamma, gamma_shape


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
		mu = torch.reshape(mu,(batch_len,image_shape.numel()))
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

def _mm_pow_th_b(model, Gmax, Wmax, mu, sigma, r):
	gamma_shape = [*mu.shape,*mu.shape[1:]]
	gamma = torch.zeros(0)
	P_tot = torch.zeros(mu.shape[0], device=mu.device, dtype=mu.dtype)
	i = 0
	for s in net_param_iterator(model):
		if isinstance(s,nn.Linear):
			mu, gamma, P_tot_i, gamma_shape = linear_layer_logic(s.weight, mu, gamma, Gmax if isinstance(Gmax, float) else Gmax[i], Wmax[i], sigma, r, gamma_shape)
			P_tot += P_tot_i
			i+=1
		if isinstance(s,nn.Softplus):
			mu, gamma, gamma_shape = softplus_vec_batched(mu, gamma, gamma_shape)
		if isinstance(s, nn.AvgPool2d):
			mu, gamma, gamma_shape = avgPool2d_layer_vec_batched(mu, gamma, s.kernel_size, s.stride, s.padding, gamma_shape)
	return mu, gamma, P_tot

@torch.no_grad()
def compute_moments_power_th_batched(model, Gmax, Wmax, sigma, mu, r):
	return _mm_pow_th_b(model, Gmax, Wmax, mu, sigma, r)

def _problem_function_batched(Gmax, args, x, z, device_id=None, torch_dtype=torch.float32, Gmax_from_quanter:bool=False):
	model, sigma, r, N, mode = args
	if device_id is not None:
		model.to(torch.device(f'cuda:{device_id}'), dtype=torch_dtype)
		z = z.to(torch.device(f'cuda:{device_id}'), dtype=torch_dtype, non_blocking=True)
		x = x.clone().to(torch.device(f'cuda:{device_id}'), dtype=torch_dtype, non_blocking=True)
	else:
		x = x.clone().to(dtype=torch_dtype)

	quanter = MemristorQuant(model, N = N, wmax_mode=mode, Gmax=Gmax, std_noise = sigma)
	quanter.quant(c_one=True)
	if Gmax_from_quanter:
		Gmax = quanter.Gmax
	elif mode=='column':
		Gmax = [torch.from_numpy(i).to(x) for i in np.split(np.copy(Gmax), np.cumsum(quanter.n_vars))[:-1]]
	mu, gamma, P_tot = compute_moments_power_th_batched(model, Gmax, quanter.Wmax, sigma, x, r)
	max_mse = torch.amax(torch.diagonal(gamma, dim1=1, dim2=2)+torch.square(mu-z), dim=1)

	quanter.unquant()
	return torch.mean(P_tot, dim=0), torch.mean(max_mse, dim=0)

# TODO : dict lict approach for ops -> (stack-style calls)
# - support kwargs in definitions 
# - input/outputs (mu, gamma, gamma_shape, p_tot) should be passed around with a dict and updated accordingly
OPS_MSE = {nn.Softplus: softplus_vec_batched, nn.AvgPool2d: avgPool2d_layer_vec_batched}
