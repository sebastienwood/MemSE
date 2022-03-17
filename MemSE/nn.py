import torch

def mse_gamma(tar, mu, gamma):
    return torch.diagonal(gamma, dim1=1, dim2=2) + torch.square(mu - tar)
