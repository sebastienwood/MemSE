from MemSE import METHODS
from MemSE.definitions import WMAX_MODE
from MemSE.test_utils import INP as inp, DEVICES, MODELS, SIGMA, get_net_transformed, nn2memse
import torch
import pytest

@pytest.mark.parametrize("method", METHODS.values())
@pytest.mark.parametrize("device", [DEVICES[1]])
@pytest.mark.parametrize("mode", [WMAX_MODE.ALL, WMAX_MODE.COLUMNWISE, WMAX_MODE.LAYERWISE])
def test_columnwise(method, device, mode):
    net = MODELS['vgg_features']
    conv2duf, o = get_net_transformed(net, method, inp)
    conv2duf, o = conv2duf, o.to(device)
    memse = nn2memse(conv2duf, mode=mode).to(device)
    memse.quanter.quant()
    print(memse.quanter.Wmax)
    memse.quanter.unquant()
    memse.forward(inp.to(device))
    

@pytest.mark.parametrize("device", DEVICES)
def test_change(device):
    net = MODELS['vgg_features']
    conv2duf, o = get_net_transformed(net, METHODS['unfolded'], inp)
    conv2duf, o = conv2duf, o.to(device)
    memse = nn2memse(conv2duf, mode=WMAX_MODE.ALL).to(device)
    all_wmax = memse.quanter.Wmax
    memse.quanter.init_wmax('layerwise')
    layerwise_wmax = memse.quanter.Wmax
    assert len(torch.unique(layerwise_wmax)) > 1
    memse.quanter.init_wmax('columnwise')
    columnwise_wmax = memse.quanter.Wmax
    for columnwise_wmax_i in columnwise_wmax:
        assert len(torch.unique(columnwise_wmax_i)) > 1
