from typing import Optional
import torch
import torch.nn as nn
from MemSE.fx import cast_to_memse
from MemSE.nn import MontecarloReturn, MemSEReturn
from MemSE import MemristorQuant


class MemSE(nn.Module):
    def __init__(self, model:nn.Module, opmap:dict, std_noise:float = 0.001, N:int = 1e6, *args, **kwargs) -> None:
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