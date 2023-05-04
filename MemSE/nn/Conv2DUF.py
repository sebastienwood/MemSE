from __future__ import annotations
import math
import torch
import torch.nn as nn
import numpy as np
import opt_einsum as oe
from torchtyping import TensorType

from MemSE.nn import MemSELayer

from .op import conv2duf_op
from .utils import double_conv, gamma_to_diag

class Conv2DUF(MemSELayer):
    def initialize_from_module(self, conv: nn.Conv2d):
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
        return [self.weight, self.bias]
    
    @property
    def memristored_einsum(self) -> dict:
        return {
            self.weight: 'coij,c->coij',
            self.bias: 'c,c->c',
            'out': 'coij,c->coij'
        }

    def unfold_input(self, x):
        return nn.functional.unfold(x, self.kernel_size, self.dilation, self.padding, self.stride)

    @staticmethod
    def memse(conv2duf: Conv2DUF, memse_dict):
        mu, gamma, gamma_shape = memse_dict['mu'], memse_dict['gamma'], memse_dict['gamma_shape']
        ct = conv2duf.Gmax / conv2duf.Wmax  # Only one column at most (vector of weights)
        ct_sq = ct ** 2
        if ct.dim() == 0:
            ct = ct.repeat(conv2duf.original_weight.shape[0]).to(mu)
            ct_sq = ct_sq.repeat(conv2duf.original_weight.shape[0]).to(mu)

        # TODO if compute power
        # dominated by memory accesses
        #  - fused kernel for all convolutions (+, -, n) using triton template (good perf)
        #  - fused kernel for mse_var (same same)
        if memse_dict['compute_power']:
            P_tot = conv2duf.energy(conv2duf,
                                    x=mu,
                                    gamma=gamma,
                                    gamma_shape=gamma_shape,
                                    c=ct,
                                    w=conv2duf.original_weight,
                                    b=conv2duf.bias,
                                    r=memse_dict['r'],
                                    sigma=memse_dict['sigma'])
        else:
            P_tot = 0.

        mu, gamma, gamma_shape = conv2duf.mse_var(conv2duf,
                                                  mu,
                                                  gamma,
                                                  gamma_shape,
                                                  memse_dict['r'],
                                                  (memse_dict['sigma'] * math.sqrt(2)) ** 2 / ct_sq,
                                                  weights=conv2duf.original_weight,
                                                  bias=conv2duf.bias
                                                  )

        memse_dict['P_tot'] += P_tot
        memse_dict['current_type'] = 'Conv2DUF'
        memse_dict['mu'] = mu
        memse_dict['gamma'] = gamma
        memse_dict['gamma_shape'] = gamma_shape

    @staticmethod
    def mse_var(conv2duf: Conv2DUF, input, gamma, gamma_shape, r, c0, weights, bias):
        mu = conv2duf.forward(input, weights, bias) * r

        # TODO
        # gamma only diag kernels at the end
        if gamma_shape is None:
            gamma_n = double_conv(gamma, weights, **conv2duf.conv_property_dict)
        else:
            gamma_n = torch.zeros(mu.shape + mu.shape[1:], device=mu.device, dtype=gamma.dtype)
        gamma_n = conv2duf_op(gamma_n, gamma, gamma_shape, input, c0, weight_shape=weights.shape, bias=bias, **conv2duf.conv_property_dict) * r ** 2
        return mu, gamma_n, None

    @staticmethod
    def energy(conv2duf: Conv2DUF,
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
        e_p_mem: torch.Tensor = torch.sum(torch.nn.functional.conv2d(sum_x_gd, abs_w, bias=abs_b, **conv2duf.conv_property_dict), dim=(1,2,3))

        Gpos = torch.clip(w, min=0)
        Gneg = torch.clip(-w, min=0)
        zp_mu, zp_g, _ = conv2duf.mse_var(conv2duf, x, gamma, gamma_shape, r, (sigma/c) ** 2, Gpos, Bpos)
        zm_mu, zm_g, _ = conv2duf.mse_var(conv2duf, x, gamma, gamma_shape, r, (sigma/c) ** 2, Gneg, Bneg)

        e_p_tiap = torch.sum((zp_mu ** 2 + gamma_to_diag(zp_g)) /  r, dim=(1,2,3))
        e_p_tiam = torch.sum((zm_mu ** 2 + gamma_to_diag(zm_g)) /  r, dim=(1,2,3))
        return e_p_mem + e_p_tiap + e_p_tiam
    
    @staticmethod
    def memse_montecarlo(conv2duf: Conv2DUF, memse_dict):
        ct = conv2duf.Gmax / conv2duf.Wmax
        
        w_noise = torch.normal(mean=0., std=conv2duf._crossbar.manager.std_noise, size=conv2duf.original_conv_weight_shape, device=conv2duf.weight.device)
        w_noise_n = torch.normal(mean=0., std=conv2duf._crossbar.manager.std_noise, size=conv2duf.original_conv_weight_shape, device=conv2duf.weight.device)
        sign_w = torch.sign(conv2duf.original_weight)
        abs_w: torch.Tensor = oe.contract('coij,c->coij', torch.abs(conv2duf.original_weight), ct).to(conv2duf.weight)
        Gpos = torch.clip(torch.where(sign_w > 0, abs_w, 0.) + w_noise, min=0)
        Gneg = torch.clip(torch.where(sign_w < 0, abs_w, 0.) + w_noise_n, min=0)
        
        if conv2duf.bias is not None:
            b_noise = torch.normal(mean=0., std=conv2duf._crossbar.manager.std_noise, size=conv2duf.bias.shape, device=conv2duf.bias.device)
            b_noise_n = torch.normal(mean=0., std=conv2duf._crossbar.manager.std_noise, size=conv2duf.bias.shape, device=conv2duf.bias.device)
            sign_b = torch.sign(conv2duf.bias)
            abs_b = oe.contract('c,c->c', torch.abs(conv2duf.bias), ct).to(conv2duf.weight) 
            Bpos = torch.clip(torch.where(sign_b > 0, abs_b, 0.) + b_noise, min=0)
            Bneg = torch.clip(torch.where(sign_b < 0, abs_b, 0.) + b_noise_n, min=0)
        else:
            abs_b = Bpos = Bneg = None
        
        # PRECOMPUTE CONVS
        zp_mu = torch.nn.functional.conv2d(memse_dict['mu'], Gpos, Bpos, **conv2duf.conv_property_dict)
        zm_mu = torch.nn.functional.conv2d(memse_dict['mu'], Gneg, Bneg, **conv2duf.conv_property_dict)
        
        # ENERGY
        sum_x_gd: torch.Tensor = memse_dict['mu'] ** 2
        e_p_mem: torch.Tensor = torch.sum(torch.nn.functional.conv2d(sum_x_gd, abs_w, bias=abs_b, **conv2duf.conv_property_dict), dim=(1,2,3))
        e_p_tiap = torch.sum(((zp_mu * memse_dict['r']) ** 2) /  memse_dict['r'], dim=(1,2,3))
        e_p_tiam = torch.sum(((zm_mu * memse_dict['r']) ** 2) /  memse_dict['r'], dim=(1,2,3))
        
        memse_dict['P_tot'] += e_p_mem + e_p_tiap + e_p_tiam
        memse_dict['current_type'] = 'Conv2DUF'
        memse_dict['mu'] = memse_dict['r'] * (torch.einsum(f'coij,c -> coij', zp_mu, 1/ct) - torch.einsum(f'coij,c -> coij', zm_mu, 1/ct))