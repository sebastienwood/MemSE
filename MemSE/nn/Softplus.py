from functools import partial
import torch
import numpy as np

import opt_einsum as oe

from typing import Callable, Optional, List

from MemSE.nn.map import register_memse_mapping
from MemSE.nn.base_layer.MemSEAct import MemSEAct
from .utils import diagonal_replace

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

	gamma_view = gamma.view(gamma.shape[0], gamma.shape[1]*gamma.shape[2]*gamma.shape[3], -1)
	sigma_2 = gamma_view.diagonal(dim1=1, dim2=2)
	sigma_2 = sigma_2.view(*gamma.shape[:4])

	if gamma_shape is not None:
		gamma = torch.zeros(gamma_shape[0],gamma_shape[1],gamma_shape[2],gamma_shape[3],gamma_shape[4],gamma_shape[5],gamma_shape[6], dtype=mu.dtype, device=mu.device)
		gamma_shape = None
		# TODO check if gamma init cannot be delegated further if its zero
		# TODO check if gamma is only diagonal elements to simplify first pass

	ga_r = oe.contract('bcij,bklm,bcijklm->bcijklm',d_mu,d_mu,gamma)
	second_comp = 0.25 * oe.contract('bcij,bklm->bcijklm', dd_mu, dd_mu)
	ga_r -= oe.contract('bcijklm,bcijcij,bklmklm->bcijklm', second_comp, gamma, gamma)

	ga_view = ga_r.view(ga_r.shape[0], ga_r.shape[1]*ga_r.shape[2]*ga_r.shape[3], -1)
	gg = torch.zeros(*ga_r.shape[:4], device=ga_r.device, dtype=ga_r.dtype)

	if degree_taylor >= 2:
		mu_r += 0.5 * oe.contract('bcij,bcij->bcij', dd_mu, sigma_2)

		gg += oe.contract('bcij,bcij->bcij', softplus_d[1] ** 2, sigma_2)
	if degree_taylor >= 4:
		mu_r += 0.125 * oe.contract('bcij,bcij->bcij', softplus_d[4], sigma_2 ** 2)

		fourth_comp = 2 * (softplus_d[2] / 2) ** 2 + oe.contract('bcij,bcij->bcij', softplus_d[1], softplus_d[3])
		gg += oe.contract('bcij,bcij->bcij', fourth_comp, sigma_2 ** 2)
	if degree_taylor >= 6:
		mu_r += (15/720) * oe.contract('bcij,bcij->bcij', softplus_d[6], sigma_2 ** 3)

		six_comp = (softplus_d[3] / 6) ** 2 * 15
		six_comp += (1/4) * oe.contract('bcij,bcij->bcij', softplus_d[1], softplus_d[5])
		six_comp += (1/2) * oe.contract('bcij,bcij->bcij', softplus_d[2], softplus_d[4])
		gg += oe.contract('bcij,bcij->bcij', six_comp, sigma_2 ** 3)

	#ga_view.diagonal(dim1=1, dim2=2).view(*ga_r.shape[:4]).copy_(gg)
	ga_r = diagonal_replace(ga_view, gg.view(*ga_view.diagonal(dim1=1, dim2=2).shape)).view(*ga_r.shape)

	#print(ga_view.diagonal(dim1=1, dim2=2).mean())
	return mu_r, ga_r, gamma_shape


def softplus(module, data):
	x, gamma, gamma_shape = softplus_vec_batched(data['mu'], data['gamma'], data['gamma_shape'], degree_taylor=data['taylor_order'])
	data['current_type'] = 'Softplus'
	data['mu'] = x
	data['gamma'] = gamma
	data['gamma_shape'] = gamma_shape


@register_memse_mapping()
class Softplus_(MemSEAct):
	__type__ = 'Softplus'
	__min_taylor__ = 2
	__max_taylor__ = 6
	def main(self, x, sigma_2, derivatives: dict):
		import warnings
		mu_r = self.functional_base(x)
		warnings.Warn("Softplus may not work properly in presence of 2dim inputs yet")

		if self.degree_taylor >= 2:
			mu_r += 0.5 * oe.contract('bcij,bcij->bcij', derivatives[2], sigma_2)  # type: ignore
			gg = oe.contract('bcij,bcij->bcij', derivatives[1] ** 2, sigma_2)
		else:
			raise ValueError('Degree taylor for Softplus should be >= 2')
		if self.degree_taylor >= 4:
			mu_r += 0.125 * oe.contract('bcij,bcij->bcij', derivatives[4], sigma_2 ** 2)  # type: ignore

			fourth_comp = 2 * (derivatives[2] / 2) ** 2 + oe.contract('bcij,bcij->bcij', derivatives[1], derivatives[3])
			gg += oe.contract('bcij,bcij->bcij', fourth_comp, sigma_2 ** 2)  # type: ignore
		if self.degree_taylor >= 6:
			mu_r += (15 / 720) * oe.contract('bcij,bcij->bcij', derivatives[6], sigma_2 ** 3)  # type: ignore

			six_comp = (derivatives[3] / 6) ** 2 * 15
			six_comp += (1 / 4) * oe.contract('bcij,bcij->bcij', derivatives[1], derivatives[5])  # type: ignore
			six_comp += (1 / 2) * oe.contract('bcij,bcij->bcij', derivatives[2], derivatives[4])  # type: ignore
			gg += oe.contract('bcij,bcij->bcij', six_comp, sigma_2 ** 3)  # type: ignore
		return mu_r, gg

	def derivatives(self, mu):
		return dd_softplus(mu, max(min(self.__max_taylor__, self.degree_taylor), self.__min_taylor__))[2]

	def gamma_extra_updates(self, gamma, x, old_gamma, derivatives) -> None:
		second_comp = 0.25 * oe.contract('bcij,bklm->bcijklm', derivatives[1], derivatives[1])  # type: ignore
		gamma -= oe.contract('bcijklm,bcijcij,bklmklm->bcijklm', second_comp, old_gamma, old_gamma)

	@classmethod
	@property
	def dropin_for(cls):
		return set([torch.nn.Softplus])

	def functional_base(self, x: torch.Tensor) -> torch.Tensor:
		return torch.nn.functional.softplus(x, beta=self.beta, threshold=self.threshold)


Softplus = Softplus_()
