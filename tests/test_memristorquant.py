from MemSE import MemristorQuant
from MemSE.definitions import WMAX_MODE
from MemSE.test_utils import INP as inp, DEVICES, MODELS, METHODS, SIGMA, get_net_transformed, nn2memse

import torch.nn as nn
import pytest

@pytest.mark.parametrize("method", METHODS.values())
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("sigma", SIGMA)
def test_columnwise(method, device, sigma):
    net = nn.Conv2d(3, 3, 3)
    net = MODELS[net]
    conv2duf, o = get_net_transformed(net, method, inp)
    conv2duf, o = conv2duf.to(device), o.to(device)
    memse = nn2memse(conv2duf, sigma, mode=WMAX_MODE.COLUMNWISE)
    # TODO: only testing init right now, should test further

