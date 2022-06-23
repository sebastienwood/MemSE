import math
import torch
from MemSE.nn.MemSEAct import MemSEAct

from MemSE.nn.utils import diagonal_replace


def relu(module, data):
    mu, gamma, gamma_shape = data['mu'], data['gamma'], data['gamma_shape']
    gamma_view = gamma.view(gamma.shape[0], gamma.shape[1]*gamma.shape[2]*gamma.shape[3], -1)
    sigma_2 = gamma_view.diagonal(dim1=1, dim2=2)
    sigma_2 = sigma_2.view(*gamma.shape[:4])
    assert sigma_2.numel() == mu.numel()

    if gamma_shape is not None:
        gamma = torch.zeros(gamma_shape[0],gamma_shape[1],gamma_shape[2],gamma_shape[3],gamma_shape[4],gamma_shape[5],gamma_shape[6], dtype=mu.dtype, device=mu.device)
        gamma_shape = None

    DENOM = math.sqrt(2 * math.pi)
    SQRT_2 = math.sqrt(2)

    first_m = (sigma_2 / DENOM) * torch.exp(-torch.square(mu / sigma_2) / 2)
    second_m = 0.5 * (1-torch.erf(-mu / (sigma_2 * SQRT_2)))
    mu_p = first_m + mu * second_m

    first_g = mu * first_m
    second_g = (torch.square(sigma_2)+torch.square(mu)) * second_m
    gamma_p = first_g + second_g
    ga_r = diagonal_replace(gamma_view, gamma_p.view(*gamma_view.diagonal(dim1=1, dim2=2).shape)).view(*gamma.shape)

    data['current_type'] = 'ReLU'
    data['mu'] = mu_p
    data['gamma'] = ga_r
    data['gamma_shape'] = gamma_shape


class ReLU(MemSEAct):
    __type__ = 'ReLU'
    def main(module, data, mu, sigma_2, *args, **kwargs):
        DENOM = math.sqrt(2 * math.pi)
        SQRT_2 = math.sqrt(2)

        first_m = (sigma_2 / DENOM) * torch.exp(-torch.square(mu / sigma_2) / 2)
        second_m = 0.5 * (1-torch.erf(-mu / (sigma_2 * SQRT_2)))
        mu_p = first_m + mu * second_m

        first_g = mu * first_m
        second_g = (torch.square(sigma_2)+torch.square(mu)) * second_m
        gamma_p = first_g + second_g - mu_p ** 2
        return mu_p, gamma_p

    def derivatives(cls, module, data, mu):
        return {1: torch.hardtanh(torch.relu(mu), max_val=0.)}