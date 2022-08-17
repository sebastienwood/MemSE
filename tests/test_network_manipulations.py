import torch
import torch.nn as nn
import numpy as np
import pytest

from MemSE.network_manipulations import build_sequential_linear, conv_to_fc, conv_to_unfolded, record_shapes, fuse_conv_bn
from MemSE.nn import Conv2DUF
from MemSE.models import smallest_vgg, resnet18, smallest_vgg_ReLU
from MemSE import MemristorQuant, MemSE

torch.manual_seed(0)
inp = torch.rand(2,3,32,32)
conv = nn.Conv2d(3,3,2)
smallest_vgg_ = smallest_vgg()
smallest_vgg_ReLU_ = smallest_vgg_ReLU()
resnet18_ = fuse_conv_bn(resnet18().eval(), 'resnet18')

MODELS = [
    conv,
    smallest_vgg_,
    resnet18_,
    smallest_vgg_ReLU_
]

devices = ['cpu']
if torch.cuda.is_available():
    devices.append('cuda')

METHODS = [conv_to_unfolded, conv_to_fc]

SIGMA = np.logspace(-1,-3,3).tolist()


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("net", MODELS)
@pytest.mark.parametrize("sigma", SIGMA)
def test_memristor_manips(method, device, net, sigma):
    y = record_shapes(net, inp).to(device)
    conv2duf = method(net, inp.shape).to(device)
    y_hat = conv2duf(inp.to(device))
    assert y.shape == y_hat.shape
    assert torch.allclose(y, y_hat, rtol=1e-3, atol=1e-6)
    quanter = MemristorQuant(conv2duf, std_noise=sigma, Gmax = 3.268, N=128)
    memse = MemSE(conv2duf, quanter, input_bias=False)#.to(device)
    m, g, p = memse.no_power_forward(inp.to(device))
    assert not torch.any(torch.isnan(m))
    assert not torch.any(torch.isnan(g))
