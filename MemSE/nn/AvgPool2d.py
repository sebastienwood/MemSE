import torch
import math
from typing import Optional, List, Tuple, Union
from MemSE.misc.has_lib import has_triton

# if has_triton():
#     from .op.conv2d_triton import conv as conv
# else:
#     conv = torch.nn.functional.conv2d
    
# TODO using triton conv2d, reduce filter size to 1
# TODO use float16
# TODO see for merge in double_conv

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
def avgPool2d_gamma_avgpool(tensor: torch.Tensor, kernel_size:Union[int, Tuple[int, int]]=2, stride:Union[int, Tuple[int, int]]=2, padding:Union[int, Tuple[int, int]]=0) -> torch.Tensor:
	if isinstance(kernel_size, int):
		kernel_size = (kernel_size, kernel_size)
	if isinstance(stride, int):
		stride = (stride, stride)
	if isinstance(padding, int):
		padding = (padding, padding)

	bs = tensor.shape[0]
	img_shape = tensor.shape[1:4]

	nice_view = tensor.reshape(-1, img_shape[0], img_shape[1], img_shape[2])#.contiguous(memory_format=torch.channels_last)
	first_res = torch.nn.functional.avg_pool2d(input=nice_view, kernel_size=kernel_size, stride=stride, padding=padding)

	first_res_shape = first_res.shape
	nice_view_res = first_res.view(bs, img_shape[0], img_shape[1], img_shape[2], img_shape[0], first_res_shape[2], first_res_shape[3])

	permuted = nice_view_res.permute(0, 4, 5, 6, 1, 2, 3)
	another_nice_view = permuted.reshape(-1, img_shape[0], img_shape[1], img_shape[2])#.contiguous(memory_format=torch.channels_last)
	second_res = torch.nn.functional.avg_pool2d(input=another_nice_view, kernel_size=kernel_size, stride=stride, padding=padding)

	second_res_shape = second_res.shape
	anv_res = second_res.view(bs, img_shape[0], first_res_shape[2], first_res_shape[3], img_shape[0], second_res_shape[2], second_res_shape[3])

	result = anv_res.permute(0, 4, 5, 6, 1, 2, 3)#.to(memory_format=torch.contiguous_format)
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


@torch.jit.script
def avgPool2d_avgpool(mu, gamma:torch.Tensor, kernel_size:int=2, stride:int=2, padding:int=0, gamma_shape:Optional[List[int]]=None) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[int]]]:
	Hin = mu.shape[2]
	Hout = Wout = math.floor(1+(Hin+2*padding-kernel_size)/stride)
	nc = mu.shape[1]

	if gamma_shape is not None:
		gamma_shape = [mu.shape[0],nc,Hout,Wout,nc,Hout,Wout]
		new_gamma = gamma
	else:
		new_gamma = avgPool2d_gamma_avgpool(gamma, kernel_size, stride, padding)

	new_mu = torch.nn.functional.avg_pool2d(mu, kernel_size=kernel_size,stride=stride, padding=padding)
	return new_mu, new_gamma, gamma_shape


def avgPool2d(module, data):
	x, gamma, gamma_shape = avgPool2d_avgpool(data['mu'], data['gamma'], module.kernel_size, module.stride, module.padding, data['gamma_shape'])
	data['current_type'] = 'AvgPool2D'
	data['mu'] = x
	data['gamma'] = gamma
	data['gamma_shape'] = gamma_shape


if __name__ == '__main__':
    import timeit
    dtype=torch.float32
    mu_test = torch.rand(16,3,6,6, dtype=dtype)
    gamma_test = torch.rand(16,3,6,6,3,6,6, dtype=dtype)
    res_legacy, res_g_l, _ = avgPool2d_layer_vec_batched(mu_test, gamma_test)
    res_pool, res_g_p, _ = avgPool2d_avgpool(mu_test, gamma_test)
    assert torch.allclose(res_legacy, res_pool.to(res_legacy)), torch.mean((res_legacy - res_pool) ** 2)
    assert torch.allclose(res_g_l, res_g_p.to(res_legacy)), torch.mean((res_legacy - res_pool) ** 2)
    mu_test = mu_test.to('cuda', dtype=torch.float16)
    gamma_test = gamma_test.to('cuda', dtype=torch.float16)
    print('Starting profile')
    print(timeit.timeit('avgPool2d_layer_vec_batched(mu_test, gamma_test)', globals=globals()))
    print(timeit.timeit('avgPool2d_avgpool(mu_test, gamma_test)', globals=globals()))