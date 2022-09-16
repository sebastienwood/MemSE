import torch

from MemSE.network_manipulations import conv_to_unfolded
from MemSE.nn import Conv2DUF
from MemSE import MemristorQuant, MemSE
torch.manual_seed(0)

def switch_conv2duf_impl(m, slow):
    if type(m) == Conv2DUF:
        m.change_impl(slow)

device = torch.device('cpu')
inp = torch.rand(2, 3, 5, 5)

SIGMA = 0.1
memse_dict = {
    'mu': inp,
    'gamma_shape': None,
    'gamma': torch.rand([*inp.shape, *inp.shape[1:]]),
    'P_tot': torch.zeros(inp.shape[0]),
    'current_type': None,
    'compute_power': False,
    'taylor_order': 1,
    'sigma': SIGMA,
    'r': 1
}

conv2d = torch.nn.Conv2d(3, 3, 3)
conv2duf = conv_to_unfolded(conv2d, inp.shape)
quanter = MemristorQuant(conv2duf, std_noise=SIGMA)
_ = MemSE.init_learnt_gmax(quanter)
quanter.quant()
ct = conv2duf.weight.learnt_Gmax / conv2duf.weight.Wmax
with torch.no_grad():
    mu, gamma, _ = Conv2DUF.mse_var(conv2duf, inp, memse_dict['gamma'], memse_dict['gamma_shape'], memse_dict['r'], ct, conv2duf.original_weight)
    mu_slow, gamma_slow, _ = Conv2DUF.slow_mse_var(conv2duf, memse_dict, ct, conv2duf.original_weight, SIGMA)

def test_eq_impl():
    assert torch.allclose(mu, mu_slow.to(mu))
    assert torch.allclose(gamma, gamma_slow.to(gamma))
