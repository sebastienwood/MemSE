import torch
import math
from typing import Optional, List

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

def avgPool2d(module, data):
	x, gamma, gamma_shape = avgPool2d_layer_vec_batched(data['mu'], data['gamma'], module.kernel_size, module.stride, module.padding, data['gamma_shape'])
	data['current_type'] = 'AvgPool2D'
	data['mu'] = x
	data['gamma'] = gamma
	data['gamma_shape'] = gamma_shape
