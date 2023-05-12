from functools import partial
import torch
import torch.nn as nn
import math
from typing import Callable, Optional, List, Tuple, Union

from MemSE.nn.map import register_memse_mapping
from MemSE.nn.base_layer import MemSELayer, MemSEReturn
from MemSE.utils import realize_tuple

__all__ = ['AdaptiveAvgPool2d', 'AvgPool2d', 'MaxPool2d']


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


#@torch.jit.script
def avgPool2d_gamma_avgpool(tensor: torch.Tensor, partial_avgpool_cb: Callable) -> torch.Tensor:
	bs = tensor.shape[0]
	img_shape = tensor.shape[1:4]

	nice_view = tensor.reshape(-1, img_shape[0], img_shape[1], img_shape[2])#.contiguous(memory_format=torch.channels_last)
	first_res = partial_avgpool_cb(input=nice_view)

	first_res_shape = first_res.shape
	nice_view_res = first_res.view(bs, img_shape[0], img_shape[1], img_shape[2], img_shape[0], first_res_shape[2], first_res_shape[3])

	permuted = nice_view_res.permute(0, 4, 5, 6, 1, 2, 3)
	another_nice_view = permuted.reshape(-1, img_shape[0], img_shape[1], img_shape[2])#.contiguous(memory_format=torch.channels_last)
	second_res = partial_avgpool_cb(input=another_nice_view)

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


#@torch.jit.script
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


def avgPool2d_montecarlo(module, data):
	data['current_type'] = 'AvgPool2D'
	data['mu'] = torch.nn.functional.avg_pool2d(data['mu'], kernel_size=module.kernel_size,stride=module.stride, padding=module.padding)
 

@register_memse_mapping()
class AvgPool2d(MemSELayer):
	def initialize_from_module(self, avgpool: nn.AvgPool2d):
		assert isinstance(avgpool, nn.AvgPool2d)
		self.kernel_size = avgpool.kernel_size
		self.stride = avgpool.stride
		self.padding = avgpool.padding
		self.ceil_mode = avgpool.ceil_mode
		assert avgpool.count_include_pad is True
		assert avgpool.divisor_override is None

	@property
	def op_properties(self):
		return {
			'kernel_size': self.kernel_size,
			'stride': self.stride,
			'padding': self.padding,
			'ceil_mode': self.ceil_mode,
		}

	def functional_base(self, x, *args, **kwargs):
		return self.partial_functional_base(x)

	@classmethod
	@property
	def dropin_for(cls):
		return set([nn.AvgPool2d])

	@property
	def partial_functional_base(self) -> Callable:
		return partial(nn.functional.avg_pool2d, **self.op_properties)

	@staticmethod
	def theoretical_output_shape(f, Hin, padding, kernel_size, stride):
		return f(((Hin + 2 * padding[0] - kernel_size[0]) / stride[0]) + 1)

	def theoretical_gamma_shape(self, x: torch.Tensor):
		Hin = x.shape[2]
		Win = x.shape[3]
		padding = realize_tuple(self.padding, 2)
		kernel_size = realize_tuple(self.kernel_size, 2)
		stride = realize_tuple(self.stride, 2)
		f = math.ceil if self.ceil_mode else math.floor

		Hout = self.theoretical_output_shape(f, Hin, padding[0], kernel_size[0], stride[0])
		Wout = self.theoretical_output_shape(f, Win, padding[1], kernel_size[1], stride[1])
		return [x.shape[0],x.shape[1],Hout,Wout,x.shape[1],Hout,Wout]

	def memse(self, previous_layer:MemSEReturn, *args, **kwargs):
		x = previous_layer.out
		gamma = previous_layer.gamma
		gamma_shape = previous_layer.gamma_shape
		power = previous_layer.power

		if gamma_shape is not None:
			gamma_shape = self.theoretical_gamma_shape(x)
		else:
			gamma = avgPool2d_gamma_avgpool(gamma, self.partial_functional_base)

		x = self.functional_base(x)
		return MemSEReturn(x, gamma, gamma_shape, power)


@register_memse_mapping()
class MaxPool2d(AvgPool2d):
	def initialize_from_module(self, avgpool: nn.MaxPool2d):
		assert isinstance(avgpool, nn.MaxPool2d)
		self.kernel_size = avgpool.kernel_size
		self.stride = avgpool.stride
		self.padding = avgpool.padding
		self.ceil_mode = avgpool.ceil_mode
		self.dilation = avgpool.dilation
		assert avgpool.return_indices is False

	@property
	def op_properties(self):
		return {
			'kernel_size': self.kernel_size,
			'stride': self.stride,
			'padding': self.padding,
			'ceil_mode': self.ceil_mode,
			'dilation': self.dilation,
		}

	def theoretical_gamma_shape(self, x: torch.Tensor):
		Hin = x.shape[2]
		Win = x.shape[3]
		padding = realize_tuple(self.padding, 2)
		kernel_size = realize_tuple(self.kernel_size, 2)
		stride = realize_tuple(self.stride, 2)
		dilation = realize_tuple(self.dilation, 2)
		f = math.ceil if self.ceil_mode else math.floor

		Hout = self.theoretical_output_shape(f, Hin, padding[0], dilation[0], kernel_size[0], stride[0])
		Wout = self.theoretical_output_shape(f, Win, padding[1], dilation[1], kernel_size[1], stride[1])
		return [x.shape[0],x.shape[1],Hout,Wout,x.shape[1],Hout,Wout]

	@classmethod
	@property
	def dropin_for(cls):
		return set([nn.MaxPool2d])

	@staticmethod
	def theoretical_output_shape(f, Hin, padding, dilation, kernel_size, stride):
		return f(((Hin + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0]) + 1)

	@property
	def partial_functional_base(self) -> Callable:
		return partial(nn.functional.max_pool2d, **self.op_properties)


@register_memse_mapping()
class AdaptiveAvgPool2d(AvgPool2d):
	def initialize_from_module(self, avgpool: nn.AdaptiveAvgPool2d):
		assert isinstance(avgpool, nn.AdaptiveAvgPool2d)
		self.output_size = avgpool.output_size

	@classmethod
	@property
	def dropin_for(cls):
		return set([nn.AdaptiveAvgPool2d])

	@property
	def partial_functional_base(self) -> Callable:
		return partial(nn.functional.adaptive_avg_pool2d, output_size=self.output_size)

	def theoretical_gamma_shape(self, x: torch.Tensor):
		output_size = realize_tuple(self.output_size, 2)
		return [x.shape[0],x.shape[1],output_size[0],output_size[1],x.shape[1],output_size[0],output_size[1]]


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