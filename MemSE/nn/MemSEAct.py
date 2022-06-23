from typing import Any
from MemSE.nn.utils import diagonal_replace

import opt_einsum as oe
import torch


class MemSEAct:
    __type__ = 'UnknownAct'
    __min_taylor__ = 1
    __max_taylor__ = 1
    @classmethod
    def __call__(cls, module, data) -> Any:
        mu, gamma, gamma_shape = data['mu'], data['gamma'], data['gamma_shape']
        degree_taylor = data['taylor_order']

        gamma_view = gamma.view(gamma.shape[0], gamma.shape[1]*gamma.shape[2]*gamma.shape[3], -1)
        sigma_2 = gamma_view.diagonal(dim1=1, dim2=2)
        sigma_2 = sigma_2.view(*gamma.shape[:4])
        assert sigma_2.numel() == mu.numel()
        if gamma_shape is not None:
            gamma = torch.zeros(gamma_shape[0],gamma_shape[1],gamma_shape[2],gamma_shape[3],gamma_shape[4],gamma_shape[5],gamma_shape[6], dtype=mu.dtype, device=mu.device)
            gamma_shape = None

        #TODO update gamma, prepare derivatives, store and pass them around as needed
        d_mu = cls.derivatives(module, data, mu)
        ga_r = oe.contract('bcij,bklm,bcijklm->bcijklm',d_mu[1],d_mu[1],gamma)
        cls.gamma_extra_update(module, data, ga_r, d_mu)
        ga_view = ga_r.view(ga_r.shape[0], ga_r.shape[1]*ga_r.shape[2]*ga_r.shape[3], -1)

        mu_p, gamma_p = cls.main(module, data, mu, sigma_2)
        ga_r = diagonal_replace(ga_view, gamma_p.view(*ga_view.diagonal(dim1=1, dim2=2).shape)).view(*ga_r.shape)

        data['current_type'] = cls.__type__
        data['mu'] = mu_p
        data['gamma'] = ga_r
        data['gamma_shape'] = None

    @staticmethod
    def main(module, data, mu, sigma_2, derivatives=None):
        pass

    @classmethod
    def derivatives(cls, module, data, mu) -> dict:
        '''Return a dict with keys 1..n the derivatives of mu'''
        pass

    @classmethod
    def gamma_extra_update(cls, module, data, gamma, derivatives):
        pass

