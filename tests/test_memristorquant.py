from MemSE import MemristorQuant
from MemSE.definitions import WMAX_MODE
from MemSE.test_utils import INP as inp, DEVICES, MODELS, METHODS, SIGMA, get_net_transformed, nn2memse

import torch.nn as nn
import pytest

@pytest.mark.parametrize("method", METHODS.values())
@pytest.mark.parametrize("device", [DEVICES[1]])
@pytest.mark.parametrize("mode", [WMAX_MODE.ALL, WMAX_MODE.COLUMNWISE, WMAX_MODE.LAYERWISE])
def test_columnwise(method, device, mode):
    net = MODELS['johnet']
    conv2duf, o = get_net_transformed(net, method, inp)
    conv2duf, o = conv2duf, o.to(device)
    print(conv2duf)
    memse = nn2memse(conv2duf, mode=mode).to(device)
    memse.quanter.quant()
    memse.quanter.unquant()
    memse.forward(inp.to(device))
    # TODO: only testing init right now, should test further

