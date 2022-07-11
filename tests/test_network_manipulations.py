import torch
import torch.nn as nn
import pytest

from MemSE.network_manipulations import build_sequential_linear, conv_to_fc, conv_to_unfolded, record_shapes, fuse_conv_bn
from MemSE.nn import Conv2DUF
from MemSE.models import smallest_vgg, resnet18

torch.manual_seed(0)
inp = torch.rand(1,3,32,32)
conv = nn.Conv2d(3,3,2)
smallest_vgg_ = smallest_vgg()
resnet18_ = fuse_conv_bn(resnet18().eval(), 'resnet18')

MODELS = [
    conv,
    smallest_vgg_,
    resnet18_
]


@pytest.mark.parametrize("net", MODELS)
def test_memristor_unfolded(net):
    y = record_shapes(net, inp)
    conv2duf = conv_to_unfolded(net, inp.shape)
    y_hat = conv2duf(inp)
    assert y.shape == y_hat.shape
    assert torch.allclose(y, y_hat, rtol=1e-3)


@pytest.mark.parametrize("net", MODELS)
def test_memristor_large(net):
    y = record_shapes(net, inp)
    conv2duf = conv_to_fc(net, inp.shape)
    y_hat = conv2duf(inp)
    assert y.shape == y_hat.shape
    assert torch.allclose(y, y_hat, rtol=1e-3)
