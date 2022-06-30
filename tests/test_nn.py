import torch
import torch.nn as nn
import pytest
import time
from MemSE.network_manipulations import conv_to_fc, conv_to_unfolded
from MemSE.nn import *
from MemSE import MemSE, MemristorQuant
from MemSE.nn.utils import mse_gamma

torch.manual_seed(0)
inp = torch.rand(2, 3, 9, 9)
conv = nn.Conv2d(3, 3, 2, bias=False)
def conv_factory():
    return nn.Conv2d(3, 3, 2, bias=False)

seq = nn.Sequential(conv,*[conv_factory() for _ in range(5)])
seq_conv2duf = conv_to_unfolded(seq, inp.shape[1:])

out = conv(inp)
conv2duf = Conv2DUF(conv, inp.shape, out.shape[1:])
out_seq = seq(inp)
out_seq_conv2duf = seq_conv2duf(inp)
def test_impl_correctness():
    assert torch.allclose(out_seq, out_seq_conv2duf)
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

def get_mses(memse, inp, out, reps=1e5):
    mse_th_mu, mse_th_gamma, p_tot = memse(inp)
    mse_th = mse_gamma(out, mse_th_mu, mse_th_gamma)
    mse_sim = memse.mse_sim(inp, out, reps=reps)
    return mse_th, mse_sim

def switch_conv2duf_impl(m, slow):
    if type(m) == Conv2DUF:
        m.change_impl(slow)


@pytest.mark.parametrize("net", [conv2duf, seq_conv2duf])
@pytest.mark.parametrize("slow", [True, False])
def test_conv2duf(net, slow):
    net.apply(lambda m: switch_conv2duf_impl(m, slow))
    quanter = MemristorQuant(net, std_noise=0.1)
    memse = MemSE(net, quanter, input_bias=False)
    mu, gamma, p_tot = memse.no_power_forward(inp)
    o = out if f'{net=}'.split('=')[0] == "conv2duf" else out_seq
    mse_th = mse_gamma(o, mu, gamma)
    mse_sim = memse.mse_sim(inp, o, reps=1e6)
    #mses, means, varis = memse.mse_forward(inp, compute_power=False, reps=1e4)
    #print(means)
    #print(varis)
    #print('----------')
    print(mse_th.mean())
    print(mse_sim.mean())
    print('MSE', ((mse_th - mse_sim) ** 2).mean())
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
    out = conv(inp)
    conv2duf = conv_to_fc(conv, inp.shape[1:], verbose=True)
    out_conv2duf = conv2duf(inp)
    assert out.shape == out_conv2duf.shape
    print(out.mean())
    print(out_conv2duf.mean())
    print(((out - out_conv2duf)**2).max())
    assert torch.allclose(out, out_conv2duf, atol=1e-5)
    quanter = MemristorQuant(conv2duf)
    memse = MemSE(conv2duf, quanter, input_bias=False)
    mse_th, mse_sim = get_mses(memse, inp, out)
    print(mse_th.mean())
    print(mse_sim.mean())
    assert torch.allclose(mse_th, mse_sim, rtol=0.1, atol=10)


def test_relu():
    relu = nn.ReLU()
    memse = nn2memse(relu)
    mse_th, mse_sim = get_mses(memse, inp, relu(inp))
    assert torch.allclose(mse_th, mse_sim, rtol=0.1, atol=10)
