import torch
import torch.nn as nn
from MemSE.network_manipulations import conv_to_fc
from MemSE.nn import *
from MemSE import MemSE, MemristorQuant
from MemSE.nn.utils import mse_gamma


inp = torch.rand(2, 3, 3, 3)
conv = nn.Conv2d(3, 3, 2, bias=False)
out = conv(inp)

def nn2memse(nn):
    quanter = MemristorQuant(nn)
    memse = MemSE(nn, quanter, input_bias=False)
    return memse

def get_mses(memse):
    mse_th_mu, mse_th_gamma, p_tot = memse(inp)
    mse_th = mse_gamma(out, mse_th_mu, mse_th_gamma)
    mse_sim = memse.mse_sim(inp, out)
    return mse_th, mse_sim

def test_conv2duf():
    conv2duf = Conv2DUF(conv, inp.shape, out.shape[1:])
    quanter = MemristorQuant(conv2duf)
    memse = MemSE(conv2duf, quanter, input_bias=False)
    mu, gamma, p_tot = memse.no_power_forward(inp)
    mse_th = mse_gamma(out, mu, gamma)
    mse_sim = memse.mse_sim(inp, out)
    mses, means, varis = memse.mse_forward(inp, compute_power=False, reps=1e5)
    print(means)
    print(varis)
    print('----------')
    print(mse_th.mean())
    print(mse_sim.mean())
    assert torch.allclose(mse_th, mse_sim)


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
