from functools import partial
from torchquad import set_up_backend, Trapezoid
import torch

set_up_backend("torch", data_type="float32")

mu = torch.rand((4,8,8))
sigma = torch.rand((4,8,8))

def order_1(mu, sigma, x):
    return torch.relu(x) * ((1/(sigma*torch.sqrt(2*3.14159265)))*torch.exp(-0.5*((x-mu)/sigma)**2))

def order_2(mu, sigma, x):
    return torch.square(torch.relu(x)) * ((1/(sigma*torch.sqrt(2*3.14159265)))*torch.exp(-0.5*((x-mu)/sigma)**2))

def integration_foreach(mu, sigma):
    '''For each element of mu/sigma perform numerical integration of order_1 and order_2'''
    o1, o2 = [], []
    for m, s in zip(mu.flatten(), sigma.flatten()):
        tp = Trapezoid()
        p_o1 = partial(order_1, m, s)
        p_o2 = partial(order_2, m, s)
        r1 = tp.integrate(
            p_o1,
            dim=1,
            N=1e6,
            integration_domain=[torch.max((0,m-s*10)),torch.max((0,m+s*10))+s],
            backend="torch",
        )
        r2 = tp.integrate(
            p_o2,
            dim=1,
            N=1e6,
            integration_domain=[torch.max((0,m-s*10)),torch.max((0,m+s*10))+s],
            backend="torch",
        )
        o1.append(r1)
        o2.append(r2)
    return o1, o2
