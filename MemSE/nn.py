import torch

def mse(tar, mu, gamma):
    return torch.diagonal(gamma, dim1=1, dim2=2) + torch.square(mu - tar)