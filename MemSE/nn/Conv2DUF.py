from __future__ import annotations
import math
import torch
import torch.nn as nn
import opt_einsum as oe
from torchtyping import TensorType

from MemSE.nn.base_layer.MemSELayer import MemSELayer, MemSEReturn
from MemSE.nn.map import register_memse_mapping

from .op import conv2duf_op, double_conv
from .utils import gamma_to_diag


@register_memse_mapping()
class Conv2DUF(MemSELayer):
    def initialize_from_module(self, conv: nn.Conv2d):
        assert isinstance(conv, nn.Conv2d)
        self.original_conv_weight_shape = conv.weight.shape
        for k in ['kernel_size', 'padding', 'dilation', 'groups', 'stride', 'output_padding', 'padding_mode', 'in_channels', 'out_channels']:
            setattr(self, k, getattr(conv, k))

        self.register_parameter('weight', nn.Parameter(conv.weight.detach().clone().view(conv.weight.shape[0], -1)))
        if conv.bias is not None:
            self.register_parameter('bias', nn.Parameter(conv.bias.detach().clone()))
        else:
            self.bias = None
            
    def extra_repr(self) -> str:
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    def functional_base(self, x, weight=None, bias=None):
        return torch.nn.functional.conv2d(x, self.original_weight if weight is None else weight, bias=self.bias if bias is None else bias, **self.conv_property_dict)
    
    @classmethod
    @property
    def dropin_for(cls):
        return set([nn.Conv2d])

    @property
    def original_weight(self) -> torch.Tensor:
        return self.weight.reshape(self.original_conv_weight_shape)

    @property
    def out_features(self):
        return self.original_weight.shape[0]

    @property
    def conv_property_dict(self):
        return {
            'stride': self.stride,
            'padding': self.padding,
            'dilation': self.dilation,
            'groups': self.groups
        }

    @property
    def memristored(self):
        return {'weight': self.weight, 'bias': self.bias}
    
    @property
    def memristored_einsum(self) -> dict:
        return {
            'weight': 'coij,c->coij',
            'bias': 'c,c->c',
            'out': 'coij,c->coij'
        }

    def unfold_input(self, x):
        return nn.functional.unfold(x, self.kernel_size, self.dilation, self.padding, self.stride)

    def memse(self, previous_layer:MemSEReturn, *args, **kwargs):
        x = previous_layer.out
        gamma = previous_layer.gamma
        gamma_shape = previous_layer.gamma_shape
        power = previous_layer.power
        
        ct = self.Gmax / self.Wmax  # Only one column at most (vector of weights)
        ct_sq = ct ** 2
        if ct.dim() == 0:
            ct = ct.repeat(self.original_weight.shape[0]).to(x)
            ct_sq = ct_sq.repeat(self.original_weight.shape[0]).to(x)

        # TODO if compute power
        # dominated by memory accesses
        #  - fused kernel for all convolutions (+, -, n) using triton template (good perf)
        #  - fused kernel for mse_var (same same)
        if power:
            P_tot = self.energy(x=x,
                                gamma=gamma,
                                gamma_shape=gamma_shape,
                                c=ct,
                                w=self.original_weight,
                                b=self.bias,
                                r=self.tia_resistance,
                                sigma=self.std_noise)
            power.add_(P_tot)

        mu, gamma, gamma_shape = self.mse_var(x,
                                              gamma,
                                              gamma_shape,
                                              self.tia_resistance,
                                              (self.std_noise * math.sqrt(2)) ** 2 / ct_sq,
                                              weights=self.original_weight,
                                              bias=self.bias
                                              )

        return MemSEReturn(mu, gamma, gamma_shape, power)

    def mse_var(self, input, gamma, gamma_shape, r, c0, weights, bias):
        mu = self.functional_base(input, weights, bias) * r

        # TODO
        # gamma only diag kernels at the end
        if gamma_shape is None:
            gamma_n = double_conv(gamma, weights, **self.conv_property_dict)
        else:
            gamma_n = torch.zeros(mu.shape + mu.shape[1:], device=mu.device, dtype=gamma.dtype)
        gamma_n = conv2duf_op(gamma_n, gamma, gamma_shape, input, c0, weight_shape=weights.shape, bias=bias, **self.conv_property_dict) * r ** 2
        return mu, gamma_n, None

    def energy(self,
               x: TensorType["batch", "channel_in", "width", "height"],
               gamma: TensorType["batch", "channel_in", "width", "height", "channel_in", "width", "height"],
               gamma_shape,
               c: TensorType["channel_out"],
               w: TensorType["channel_out", "channel_in", "width", "height"],
               b: TensorType["channel_out"],
               r,
               sigma):
        sum_x_gd: torch.Tensor = x ** 2
        if gamma_shape is None:
            sum_x_gd += gamma_to_diag(gamma)
        abs_w: torch.Tensor = oe.contract('coij,c->coij', torch.abs(w), c).to(sum_x_gd)
        if b is not None:
            abs_b = oe.contract('c,c->c', torch.abs(b), c).to(sum_x_gd) 
            Bpos = torch.clip(b, min=0)
            Bneg = torch.clip(-b, min=0)
        else:
            abs_b = Bpos = Bneg = None
        e_p_mem: torch.Tensor = torch.sum(torch.nn.functional.conv2d(sum_x_gd, abs_w, bias=abs_b, **self.conv_property_dict), dim=(1,2,3))

        Gpos = torch.clip(w, min=0)
        Gneg = torch.clip(-w, min=0)
        zp_mu, zp_g, _ = self.mse_var(x, gamma, gamma_shape, r, (sigma/c) ** 2, Gpos, Bpos)
        zm_mu, zm_g, _ = self.mse_var(x, gamma, gamma_shape, r, (sigma/c) ** 2, Gneg, Bneg)

        e_p_tiap = torch.sum((zp_mu ** 2 + gamma_to_diag(zp_g)) / r, dim=(1,2,3))
        e_p_tiam = torch.sum((zm_mu ** 2 + gamma_to_diag(zm_g)) / r, dim=(1,2,3))
        return e_p_mem + e_p_tiap + e_p_tiam
