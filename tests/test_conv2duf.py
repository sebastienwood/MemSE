import torch

from MemSE.network_manipulations import conv_to_unfolded
from MemSE.nn import Conv2DUF
from MemSE import MemristorQuant, MemSE
torch.manual_seed(0)
SIGMA = 0.1
inp = torch.rand(2, 3, 5, 5)
conv2d = torch.nn.Conv2d(3, 3, 3)

def test_eq_impl():
    if torch.cuda.is_available():
        mus, gammas = [], []
        for device in [torch.device('cpu'), torch.device('cuda:0')]:
            inp_p = inp.clone().to(device)

            memse_dict = {
                'mu': inp_p,
                'gamma_shape': None,
                'gamma': torch.rand([*inp.shape, *inp.shape[1:]]).to(device),
                'P_tot': torch.zeros(inp.shape[0]).to(device),
                'current_type': None,
                'compute_power': True,
                'sigma': SIGMA,
                'r': 1
            }

            conv2duf = conv_to_unfolded(conv2d, inp.shape).to(device)
            quanter = MemristorQuant(conv2duf, std_noise=SIGMA)
            _ = MemSE.init_learnt_gmax(quanter)
            quanter.quant()
            with torch.no_grad():
                Conv2DUF.memse(conv2duf, memse_dict)
                mus.append(memse_dict['mu'])
                gammas.append(memse_dict['gamma'])
    assert torch.allclose(mus[0], mus[1].to(mus[0]))
    assert torch.allclose(gammas[0], gammas[1].to(gammas[0]))
