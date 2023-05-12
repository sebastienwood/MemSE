from typing import Any, Tuple
from MemSE.nn.base_layer import MemSELayer, MemSEReturn, MontecarloReturn
from MemSE.nn.utils import diagonal_replace

import opt_einsum as oe
import torch


class MemSEAct(MemSELayer):
    __type__ = 'UnknownAct'
    __min_taylor__ = 1
    __max_taylor__ = 1
    def memse(self, previous_layer:MemSEReturn, *args, **kwargs) -> MemSEReturn:
        # TODO work with any input shape
        x = previous_layer.out
        gamma = previous_layer.gamma
        gamma_shape = previous_layer.gamma_shape

        original_mu_shape = x.shape
        if gamma_shape is not None:
            gamma = torch.zeros(*gamma_shape, dtype=gamma.dtype, device=x.device)
            gamma_shape = None

        if len(gamma.shape) == 7: # TODO could simplify by flattening, all subcalls should be elementwise
            gamma_square_view_shape = (gamma.shape[0], gamma.shape[1] * gamma.shape[2] * gamma.shape[3], -1)
            einsum_expr = 'bcij,bklm,bcijklm->bcijklm'
        elif len(gamma.shape) == 3:
            gamma_square_view_shape = gamma.shape
            einsum_expr = 'bi,bj,bij->bij'
        else:
            raise ValueError('Unsupported shape for activation inputs')

        gamma_view = gamma.reshape(gamma_square_view_shape)
        sigma_2 = gamma_view.diagonal(dim1=1, dim2=2)
        sigma_2 = sigma_2.view(*x.shape)
        assert sigma_2.numel() == x.numel()

        d_mu = self.derivatives(x)
        ga_r: torch.Tensor = oe.contract(einsum_expr,d_mu[1],d_mu[1],gamma)  # type: ignore
        self.gamma_extra_update(ga_r, x, gamma, d_mu)
        ga_view = ga_r.reshape(gamma_square_view_shape)

        mu_p, gamma_p = self.main(x, sigma_2, d_mu)
        ga_r = diagonal_replace(ga_view, gamma_p.reshape(*ga_view.diagonal(dim1=1, dim2=2).shape)).view(*ga_r.shape)
        ga_r.extra_info = 'ga_r in memseact'
        return MemSEReturn(mu_p, ga_r, gamma_shape, power=previous_layer.power)

    def main(self, x, sigma_2, derivatives=None) -> Tuple[torch.Tensor, torch.Tensor]:
        raise ValueError('Should be implemented in children class')

    def derivatives(self, x: torch.Tensor) -> dict:
        '''Return a dict with keys 1..n the derivatives of mu'''
        pass

    def gamma_extra_update(self, gamma, x, old_gamma, derivatives) -> None:
        pass
