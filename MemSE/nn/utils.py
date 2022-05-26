import torch
import torch.nn as nn
import opt_einsum as oe

from typing import Optional, List

__all__ = ['mse_gamma', 'diagonal_replace', 'zero_but_diag_', 'quant_but_diag_']

def mse_gamma(tar, mu, gamma, verbose: bool = False):
	if len(tar.shape) != len(mu.shape):
		tar = tar.view_as(mu)
	vari = torch.diagonal(gamma, dim1=1, dim2=2)
	exp = torch.square(mu - tar)
	if verbose:
		res_v = vari.mean(dim=1).abs()
		res_e = exp.mean(dim=1).abs()
		tot = res_v + res_e
		print(f'VAR IMPORTANCE {res_v / tot}')
		print(f'EXP IMPORTANCE {res_e / tot}')
	return exp + vari


def diagonal_replace(tensor, diagonal):
	'''Backprop compatible diagonal replacement
	'''
	mask = torch.diag(torch.ones(diagonal.shape[1:], device=tensor.device)).unsqueeze_(0)
	out = mask * torch.diag_embed(diagonal) + (1 - mask) * tensor
	return out


def zero_but_diag_(tensor):
	diag = tensor.diagonal(dim1=1, dim2=2).data.clone()
	tensor.data.zero_()
	tensor.diagonal(dim1=1, dim2=2).data.copy_(diag)


def quant_but_diag_(tensor, quant_scheme):
	pass


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


def double_conv(tensor, weight, stride, padding, dilation, groups):
	'''A doubly convolution for tensor of shape [bijkijk]'''
	# TODO not so sure it works for grouped convolutions
	assert type(groups) == int or groups == 'adaptive'
	bs = tensor.shape[0]
	nc = tensor.shape[1]
	img_shape = tensor.shape[1:4]

	nice_view = tensor.reshape(bs, -1, img_shape[1], img_shape[2])
	nc_first = nice_view.shape[1]
	first_res = torch.nn.functional.conv2d(nice_view, weight, stride=stride, padding=padding, dilation=dilation, groups=nc_first / (weight.shape[1] * groups))

	first_res_shape = first_res.shape
	nice_view_res = first_res.view(bs, img_shape[0], img_shape[1], img_shape[2], nc, first_res_shape[2], first_res_shape[3])

	permuted = nice_view_res.permute(0, 4, 5, 6, 1, 2, 3)
	another_nice_view = permuted.reshape(bs, -1, img_shape[1], img_shape[2])
	nc_second = another_nice_view.shape[1]
	second_res = torch.nn.functional.conv2d(another_nice_view, weight, stride=stride, padding=padding, dilation=dilation, groups=nc_second / (weight.shape[1] * groups))

	second_res_shape = second_res.shape
	anv_res = second_res.view(bs, nc, first_res_shape[2], first_res_shape[3], nc, second_res_shape[2], second_res_shape[3])

	return anv_res.permute(0, 4, 5, 6, 1, 2, 3)


def gamma_to_diag(tensor, flatten=False):
	bs = tensor.shape[0]
	nc = tensor.shape[1]
	img_shape = tensor.shape[1:4]
	numel_image = img_shape.numel()
	view = torch.reshape(tensor, (bs,numel_image,numel_image))
	diag = torch.diagonal(view, dim1=1, dim2=2)
	return diag.reshape((bs,*img_shape)) if not flatten else diag
