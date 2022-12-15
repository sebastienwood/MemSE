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
        # TODO work with any input shape
        mu, gamma, gamma_shape = data['mu'], data['gamma'], data['gamma_shape']
        original_mu_shape = mu.shape
        if gamma_shape is not None:
            gamma = torch.zeros(*gamma_shape, dtype=gamma.dtype, device=mu.device)
            gamma_shape = None

        if len(gamma.shape) == 7: # TODO could simplify by flattening, all subcalls should be elementwise
            gamma_square_view_shape = (gamma.shape[0], gamma.shape[1]*gamma.shape[2]*gamma.shape[3], -1)
            einsum_expr = 'bcij,bklm,bcijklm->bcijklm'
        elif len(gamma.shape) == 3:
            gamma_square_view_shape = gamma.shape
            einsum_expr = 'bi,bj,bij->bij'
        else:
            raise ValueError('Unsupported shape for activation inputs')

        gamma_view = gamma.reshape(gamma_square_view_shape)
        sigma_2 = gamma_view.diagonal(dim1=1, dim2=2)
        sigma_2 = sigma_2.view(*mu.shape)
        assert sigma_2.numel() == mu.numel()

        d_mu = cls.derivatives(module, data, mu)
        ga_r = oe.contract(einsum_expr,d_mu[1],d_mu[1],gamma)
        cls.gamma_extra_update(module, data, ga_r, d_mu)
        ga_view = ga_r.reshape(gamma_square_view_shape)

        mu_p, gamma_p = cls.main(module, data, mu, sigma_2, d_mu)
        ga_r = diagonal_replace(ga_view, gamma_p.reshape(*ga_view.diagonal(dim1=1, dim2=2).shape)).view(*ga_r.shape)
        ga_r.extra_info = 'ga_r in memseact'
        data['current_type'] = cls.__type__
        data['mu'] = mu_p
        data['gamma'] = ga_r
        data['gamma_shape'] = gamma_shape

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

