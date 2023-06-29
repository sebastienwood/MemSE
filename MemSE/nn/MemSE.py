import copy
from typing import Optional
import torch
import torch.nn as nn
from MemSE.fx import cast_to_memse
from MemSE.nn import MontecarloReturn, MemSEReturn, MEMSE_MAP
from MemSE import MemristorQuant
from ofa.imagenet_classification.elastic_nn.modules.dynamic_layers import (
    DynamicConvLayer,
    DynamicLinearLayer,
    DynamicConv2d
)
from ofa.imagenet_classification.elastic_nn.utils import set_running_statistics


class MemSE(nn.Module):
    def __init__(self, model:nn.Module, opmap:dict = MEMSE_MAP, std_noise:float = 0.001, N:int = 1e6, gu=10., *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = cast_to_memse(model, opmap)
        self.quanter = MemristorQuant(self.model, std_noise=std_noise, N=N, Gmax=gu)

    def forward(self, x):
        return self.model(x)

    def quant(self, c_one:bool = False):
        self.quanter.quant(c_one=c_one)

    def unquant(self):
        self.quanter.unquant()

    def montecarlo_forward(self, x):
        assert self.quanter.quanted and not self.quanter.noised, 'Need quanted and denoised'
        x = MontecarloReturn(out=x, power=torch.zeros(x.shape[0], device=x.device))
        return self.model(x)

    def memse_forward(self, x):
        assert self.quanter.quanted and not self.quanter.noised, 'Need quanted and denoised'
        x = MemSEReturn(out=x, gamma=torch.zeros(0, device=x.device, dtype=x.dtype), gamma_shape=[*x.shape, *x.shape[1:]], power=torch.zeros(x.shape[0], device=x.device))
        return self.model(x)


def count_layers(model):
    size = 0
    for n, m in model.named_modules():
        if isinstance(m, (DynamicLinearLayer, DynamicConv2d)):
            size += 1
    return size


class OFAxMemSE(nn.Module):
    def __init__(self, model: nn.Module, opmap:dict = MEMSE_MAP, std_noise: float = 0.001, N: int = 1000000, *args, **kwargs) -> None:
        nn.Module.__init__(self)
        self.model = model
        self.opmap = opmap
        self.std_noise = std_noise
        self.N = N
        assert hasattr(self.model, 'sample_active_subnet')

        # Crude Gmax computation: count the nb of DynamicConvLayer and DynamicLinearLayer
        self.gmax_size = count_layers(model)
        # Cache block index to crossbars index
        self.block_to_crossbar_map = {}
        for stage_id, block_idx in enumerate(self.model.grouped_block_index):
            for idx in block_idx:
                if idx - 1 in self.block_to_crossbar_map:
                    offset = max(self.block_to_crossbar_map[idx-1]) + 1
                else:
                    offset = 3
                self.block_to_crossbar_map[idx] = [offset + v for v in range(count_layers(self.model.blocks[idx]))]
        print('OFAxMemSE initialized with ', self.gmax_size, ' crossbars')

    def to(self, *args, **kwargs):
        self._device = (args, kwargs)
        return super().to(*args, **kwargs)

    def quant(self, c_one:bool = False):
        self._model.quanter.quant(c_one=c_one)

    def unquant(self):
        self._model.quanter.unquant()

    def forward(self, inp):
        return self._model.forward(inp)

    def montecarlo_forward(self, inp):
        return self._model.montecarlo_forward(inp)

    def memse_forward(self, inp):
        return self._model.memse_forward(inp)

    def sample_active_subnet(self, data_loader, skip_adaptation: bool = False, noisy: bool = True, arch=None):
        # TODO this static cast may be inefficient, but we'd need to rewrite OFA's (conv, bn) to dynamically fuse them
        # its also not very flexible as it only works for resnet
        if arch is None:
            arch_config = self.model.sample_active_subnet()  # type: ignore
        else:
            self.model.set_active_subnet(**arch)
            arch_config = arch

        # get currently active crossbars as mask
        # https://github.com/mit-han-lab/once-for-all/blob/a5381c1924d93e582e4a321b3432579507bf3d22/ofa/imagenet_classification/elastic_nn/networks/ofa_resnets.py#LL288C9-L291C35
        # (only valid for resnet)
        active_crossbars = [0, 2, self.gmax_size - 1]
        if self.model.input_stem_skipping <= 0:
            active_crossbars.append(1)
        for stage_id, block_idx in enumerate(self.model.grouped_block_index):
            depth_param = self.model.runtime_depth[stage_id]
            active_idx = block_idx[: len(block_idx) - depth_param]
            for idx in active_idx:
                active_crossbars.extend(self.block_to_crossbar_map[idx])
        active_crossbars.sort()

        model = self.model.get_active_subnet()
        if hasattr(self, '_device'):
            model = model.to(*self._device[0], **self._device[1])
        if not skip_adaptation:
            set_running_statistics(model, data_loader)
        self._model = MemSE(model, self.opmap, self.std_noise, self.N)
        assert self._model.quanter.Wmax.numel() == len(active_crossbars)

        if noisy:
            gmax = torch.einsum("a,a->a", torch.zeros(len(active_crossbars)).normal_(1, 0.3), self._model.quanter.Wmax)
        else:
            gmax = self._model.quanter.Wmax.clone()
        gmax.clamp_(1e-6)
        self._model.quanter.init_gmax(gmax)
        if hasattr(self, '_device'):
            self._model = self._model.to(*self._device[0], **self._device[1])
        gmax_clean = torch.zeros(self.gmax_size).scatter_(0, torch.LongTensor(active_crossbars), gmax).tolist()
        state = arch_config | {'gmax': gmax_clean}
        self._state = state

        if hasattr(self, '_device'):
            self._model = self._model.to(*self._device[0], **self._device[1])
        return state

    def set_active_subnet(self, arch, data_loader):
        self._state = arch
        arch = copy.deepcopy(arch)
        gmax = arch.pop('gmax')
        self.model.set_active_subnet(**arch)
        model = self.model.get_active_subnet()
        set_running_statistics(model, data_loader)
        self._model = MemSE(model, self.opmap, self.std_noise, self.N, gu=[g for g in gmax if g > 0])
        if hasattr(self, '_device'):
            self._model = self._model.to(*self._device[0], **self._device[1])
