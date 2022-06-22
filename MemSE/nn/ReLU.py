import math
import torch


def relu(module, data):
    mu, gamma, gamma_shape = data['mu'], data['gamma'], data['gamma_shape']
    gamma_view = gamma.view(gamma.shape[0], gamma.shape[1]*gamma.shape[2]*gamma.shape[3], -1)
    sigma_2 = gamma_view.diagonal(dim1=1, dim2=2)
    sigma_2 = sigma_2.view(*gamma.shape[:4])

    DENOM = math.sqrt(2 * math.pi)
    SQRT_2 = math.sqrt(2)

    if gamma_shape is not None:
        gamma = torch.zeros(gamma_shape[0],gamma_shape[1],gamma_shape[2],gamma_shape[3],gamma_shape[4],gamma_shape[5],gamma_shape[6], dtype=mu.dtype, device=mu.device)
        gamma_shape = None

    first_m = (sigma_2 / DENOM) * torch.exp(-torch.square(mu / sigma_2) / 2)
    second_m = 0.5 * (1-torch.erf(-mu / (sigma_2 * SQRT_2)))
    mu_p = first_m + mu * second_m

    first_g = mu * first_m
    second_g = (torch.square(sigma_2)+torch.square(mu)) * second_m
    gamma_p = first_g + second_g

    data['current_type'] = 'ReLU'
    data['mu'] = mu_p
    data['gamma'] = gamma_p
    data['gamma_shape'] = gamma_shape
