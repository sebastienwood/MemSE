import torch
import torch.nn as nn
import pytest
import time
from MemSE.network_manipulations import conv_to_fc
from MemSE.nn import *
from MemSE import MemSE, MemristorQuant
from MemSE.nn.utils import mse_gamma

torch.manual_seed(0)
inp = torch.rand(2, 3, 9, 9)
conv = nn.Conv2d(3, 3, 2, bias=False)
out = conv(inp)
SIGMA = 0.1
R = 1
memse_dict = {
			'mu': inp,
			'gamma_shape': None,
			'gamma': torch.rand([*inp.shape, *inp.shape[1:]]),
			'P_tot': torch.zeros(inp.shape[0]),
			'current_type': None,
			'compute_power': False,
			'taylor_order': 1,
			'sigma': SIGMA,
			'r': R
}

def nn2memse(nn):
    quanter = MemristorQuant(nn)
    memse = MemSE(nn, quanter, input_bias=False)
    return memse

def get_mses(memse):
    mse_th_mu, mse_th_gamma, p_tot = memse(inp)
    mse_th = mse_gamma(out, mse_th_mu, mse_th_gamma)
    mse_sim = memse.mse_sim(inp, out)
    return mse_th, mse_sim


@pytest.mark.parametrize("slow", [True, False])
def test_conv2duf(slow):
    conv2duf = Conv2DUF(conv, inp.shape, out.shape[1:], slow=slow)
    quanter = MemristorQuant(conv2duf, std_noise=0.1)
    memse = MemSE(conv2duf, quanter, input_bias=False)
    mu, gamma, p_tot = memse.no_power_forward(inp)
    mse_th = mse_gamma(out, mu, gamma)
    mse_sim = memse.mse_sim(inp, out, reps=1e5)
    #mses, means, varis = memse.mse_forward(inp, compute_power=False, reps=1e4)
    #print(means)
    #print(varis)
    #print('----------')
    print(mse_th)
    print(mse_sim)
    assert torch.allclose(mse_th.to(mse_sim), mse_sim, rtol=0.05)


def test_conv2duf_mse_var():
    conv2duf = Conv2DUF(conv, inp.shape, out.shape[1:])
    quanter = MemristorQuant(conv2duf, std_noise=SIGMA)
    _ = MemSE.init_learnt_gmax(quanter)
    quanter.quant()
    ct = conv2duf.weight.learnt_Gmax / conv2duf.weight.Wmax
    start = time.time()
    mu, gamma, _ = conv2duf.mse_var(conv2duf, memse_dict, ct, conv2duf.original_weight, SIGMA)
    print(time.time()- start)
    start = time.time()
    mu_slow, gamma_slow, _ = conv2duf.slow_mse_var(conv2duf, memse_dict, ct, conv2duf.original_weight, SIGMA)
    print(time.time()- start)
    print('Reporting fast')
    print('MEAN', gamma.mean())
    print('SPARSITY', torch.count_nonzero(gamma)/torch.numel(gamma))
    print('SHAPE', gamma.shape)
    print('Reporting slow')
    print('MEAN', gamma_slow.mean())
    print('SPARSITY', torch.count_nonzero(gamma_slow)/torch.numel(gamma_slow))
    print('SHAPE', gamma_slow.shape)
    print('Reporting error')
    assert gamma.shape == gamma_slow.shape
    print('MSE', ((gamma - gamma_slow) ** 2).mean())
    assert False # so that logs are printed


def test_conv2d():
    conv2duf = conv_to_fc(conv, inp.shape[1:], verbose=True)
    quanter = MemristorQuant(conv2duf)
    memse = MemSE(conv2duf, quanter, input_bias=False)
    mse_th, mse_sim = get_mses(memse)
    print(mse_th.mean())
    print(mse_sim.mean())
    assert torch.allclose(mse_th, mse_sim, rtol=0.1, atol=10)


def test_relu():
    relu = nn.ReLU()
    memse = nn2memse(relu)
    mse_th, mse_sim = get_mses(memse)
    assert torch.allclose(mse_th, mse_sim, rtol=0.1, atol=10)
