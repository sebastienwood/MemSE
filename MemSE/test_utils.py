import torch
import torch.nn as nn
import numpy as np


from MemSE import MemristorQuant, MemSE
from MemSE.nn import Flattener
from MemSE.dataset import get_dataloader
from MemSE.definitions import WMAX_MODE, ROOT
from MemSE.models import smallest_vgg, resnet18, smallest_vgg_ReLU, make_small_JohNet
from MemSE.fx import record_shapes, fuse_conv_bn

DEVICES = ['cpu']
if torch.cuda.is_available():
    DEVICES.append('cuda')

BATCH_SIZE = 2
_, _, test_loader, _, _ = get_dataloader('cifar10', root=f'{ROOT}/data', bs=BATCH_SIZE)
INP = next(iter(test_loader))[0]


def conv_factory():
    return nn.Conv2d(3, 3, 2, bias=False)


MODELS = {
    'relu': nn.ReLU(),
    'conv': conv_factory(),
    'seq': nn.Sequential(*[nn.Sequential(conv_factory(), nn.ReLU()) for _ in range(3)]),
    'fc': nn.Sequential(Flattener(), nn.Linear(32*32*3, 32*32, bias=True)),
    'vgg': smallest_vgg(),
    'vgg_relu': smallest_vgg_ReLU(),
    'vgg_features': smallest_vgg_ReLU(features_only=True),
    'resnet': fuse_conv_bn(resnet18().eval(), 'resnet18'),
    'johnet': fuse_conv_bn(make_small_JohNet().eval(), 'make_johnet'),
}


def get_net_transformed(net, method, inp: torch.Tensor=INP):
    y = record_shapes(net, inp)
    conv2duf = method(net, inp.shape)
    y_hat = conv2duf(inp)
    assert y.shape == y_hat.shape
    #assert torch.allclose(y, y_hat, atol=1e-6, rtol=1e-4), torch.mean((y_hat - y)**2)
    return conv2duf, y.detach()


SIGMA = np.logspace(-1, -3, 3).tolist()


def nn2memse(nn, sigma: float=SIGMA[0], mode=WMAX_MODE.ALL):
    quanter = MemristorQuant(nn, std_noise=sigma, Gmax = 3.268, N=128, wmax_mode=mode)
    memse = MemSE(nn, quanter, input_bias=False)
    return memse
