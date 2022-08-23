import torch
import torch.nn as nn
import numpy as np
import pytest

from MemSE.nn import mse_gamma

DEBUG = True
torch.manual_seed(0)
#inp = torch.rand(BATCH_SIZE,3,32,32)

from MemSE.test_utils import INP as inp, DEVICES, MODELS, METHODS, SIGMA, get_net_transformed, nn2memse


@pytest.mark.parametrize("method", METHODS.values())
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("net", MODELS.keys())
@pytest.mark.parametrize("sigma", SIGMA)
def test_memristor_manips(method, device, net, sigma):
    net = MODELS[net]
    conv2duf, o = get_net_transformed(net, method)
    conv2duf, o = conv2duf.to(device), o.to(device)
    memse = nn2memse(conv2duf, sigma)
    m, g, _ = memse.no_power_forward(inp.to(device))
    assert not torch.any(torch.isnan(m))
    assert not torch.any(torch.isnan(g))
    mse_th = mse_gamma(o, m, g)
    mse_sim = memse.mse_sim(inp, o, reps=1e3)
    assert torch.allclose(mse_th.to(mse_sim), mse_sim, rtol=0.05)  # TODO method/net wise rtol/atol

    if DEBUG:
        print('DEBUG STARTS')
        mses, means, varis, covs = memse.mse_forward(inp, compute_power=False, reps=1e3)
        for t in [means, varis, covs]:
            for type in t['sim'].keys():
                for idx in type.keys():
                    assert torch.allclose(t['sim'][type][idx], t['us'][type][idx], rtol=0.05)
        print(means)
        print(varis)
        print(covs)
        print('----------')
        print(mse_th.mean())
        print(mse_sim.mean())
        print('MSE', ((mse_th - mse_sim) ** 2).mean())
        assert False # enforce output to stdout
