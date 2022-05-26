import torch
import torch.nn as nn

from MemSE.network_manipulations import build_sequential_linear, record_shapes
from MemSE.nn import Conv2DUF


def test_memristor_unfolded():
    inp = torch.rand(1,3,10,12)
    conv = nn.Conv2d(3,3,2)
    y = record_shapes(conv, inp)
    conv2duf = Conv2DUF(conv, inp.shape, conv.__output_shape)
    y_hat = conv2duf(inp)
    assert torch.allclose(y, y_hat)

def test_memristor_large():
    inp = torch.rand(1,3,10,12)
    conv = nn.Conv2d(3,3,2)
    y = record_shapes(conv, inp)
    conv2duf = build_sequential_linear(conv)
    y_hat = conv2duf(inp)
    assert torch.allclose(y, y_hat)
