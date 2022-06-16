import torch
import torch.nn as nn
from MemSE.network_manipulations import conv_to_fc
from MemSE.nn import Conv2DUF
from MemSE import MemSE, MemristorQuant
from MemSE.nn.utils import mse_gamma


inp = torch.rand(2, 3, 3, 3)
conv = nn.Conv2d(3, 3, 2, bias=False)
out = conv(inp)


def test_conv2duf():
    conv2duf = Conv2DUF(conv, inp.shape, out.shape[1:])
    quanter = MemristorQuant(conv2duf)
    memse = MemSE(conv2duf, quanter, input_bias=False)
    memse.quant()
    mu, gamma, p_tot = memse.no_power_forward(inp)
    mse_th = mse_gamma(out, mu, gamma)
    mse_sim = memse.mse_sim(inp, out)
    print(mse_th.mean())
    print(mse_sim.mean())
    assert torch.allclose(mse_th, mse_sim)


def test_conv2d():
    conv2duf = conv_to_fc(conv, inp.shape[1:], verbose=True)
    quanter = MemristorQuant(conv2duf)
    memse = MemSE(conv2duf, quanter, input_bias=False)
    memse.quant()
    mse_th_mu, mse_th_gamma, p_tot = memse(inp)
    mse_th = mse_gamma(out, mse_th_mu, mse_th_gamma)
    mse_sim = memse.mse_sim(inp, out)
    print(mse_th.mean())
    print(mse_sim.mean())
    assert torch.allclose(mse_th, mse_sim)
