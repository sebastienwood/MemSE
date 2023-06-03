from typing import Optional
import torch
import torch.nn as nn
from MemSE.fx import cast_to_memse
from MemSE.nn import MontecarloReturn, MemSEReturn, MEMSE_MAP
from MemSE import MemristorQuant
from ofa.imagenet_classification.elastic_nn.modules.dynamic_layers import (
    DynamicConvLayer,
    DynamicLinearLayer,
)


class MemSE(nn.Module):
    def __init__(self, model:nn.Module, opmap:dict = MEMSE_MAP, std_noise:float = 0.001, N:int = 1e6, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = cast_to_memse(model, opmap)
        self.quanter = MemristorQuant(self.model, std_noise=std_noise, N=N, Gmax=10.)

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


class OFAxMemSE(nn.Module):
    def __init__(self, model: nn.Module, opmap: dict, std_noise: float = 0.001, N: int = 1000000, *args, **kwargs) -> None:
        nn.Module.__init__(self)
        self.model = model
        self.opmap = opmap
        self.std_noise = std_noise
        self.N = N
        assert hasattr(self.model, 'sample_active_subnet')

        # Crude Gmax computation: count the nb of DynamicConvLayer  and DynamicLinearLayer
        self.gmax_size = 0
        for m in model.modules():
            if isinstance(m , (DynamicLinearLayer, DynamicConvLayer)):
                self.gmax_size += 1

    def sample_active_subnet(self):
        # TODO this static cast may be inefficient, but we'd need to rewrite OFA's (conv, bn) to dynamically fuse them
        # its also not very flexible as it only works for resnet
        arch_config = self.model.sample_active_subnet()  # type: ignore
        # get currently active crossbars as mask
        # https://github.com/mit-han-lab/once-for-all/blob/a5381c1924d93e582e4a321b3432579507bf3d22/ofa/imagenet_classification/elastic_nn/networks/ofa_resnets.py#LL288C9-L291C35
        # (only valid for resnet)
        active_crossbars = [0, 2]
        if self.model.input_stem_skipping <= 0:
            active_crossbars.append(1)
        current = 2
        for stage_id, block_idx in enumerate(self.model.grouped_block_index):
            depth_param = self.model.runtime_depth[stage_id]
            active_idx = block_idx[: len(block_idx) - depth_param]
            for idx in active_idx:
                active_crossbars.append(current + idx + 1) # TODO check if there arent 3 convs that should be accounted here
            current += len(block_idx)
        active_crossbars.append(current + 1)  # linear classifier
        active_crossbars.sort()

        # sample on these crossbars Gmax

        # return intialized memse
        model = self.model.get_active_subnet()
        model = cast_to_memse(model, self.opmap)
        quanter = MemristorQuant(self.model, std_noise=self.std_noise, N=self.N, Gmax=gmax)