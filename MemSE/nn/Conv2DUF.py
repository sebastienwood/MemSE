from __future__ import annotations
import math
import torch
import torch.nn as nn
import numpy as np
import opt_einsum as oe
from torchtyping import TensorType

from .op import conv2duf_op
from .utils import double_conv, gamma_to_diag

class Conv2DUF(nn.Module):
    def __init__(self, conv, input_shape, output_shape):
        super().__init__()
        assert len(output_shape) == 3, f'chw or cwh with no batch dim ({output_shape})'
        self.c = [conv]
        self.output_shape = output_shape

        self.register_parameter('weight', nn.Parameter(conv.weight.detach().clone().view(conv.weight.shape[0], -1)))
        assert conv.bias is None

    def forward(self, x, weight=None):
        # inp_unf = self.unfold_input(x)
        # out_unf = inp_unf.transpose(1, 2).matmul(self.weight.to(x)).transpose(1, 2)
        # out = out_unf.view(x.shape[0], *self.output_shape)
        # if self.bias is not None:
        # 	out += self.bias[:, None, None]
        # return out
        return torch.nn.functional.conv2d(x, self.original_weight if weight is None else weight, **self.conv_property_dict)

    @property
    def original_weight(self):
        return self.weight.reshape(self.c[0].weight.shape)

    @property
    def out_features(self):
        return self.original_weight.shape[0]

    @property
    def kernel_size(self):
        return self.c[0].kernel_size

    @property
    def padding(self):
        return self.c[0].padding

    @property
    def dilation(self):
        return self.c[0].dilation

    @property
    def groups(self):
        return 'adaptive' if self.c[0].groups == 'adaptive' else int(self.c[0].groups)

    @property
    def stride(self):
        return self.c[0].stride

    @property
    def out_features_numel(self):
        return self.output_shape.numel()

    @property
    def conv_property_dict(self):
        return {
            'stride': self.stride,
            'padding': self.padding,
            'dilation': self.dilation,
            'groups': self.groups
        }

    def unfold_input(self, x):
        return nn.functional.unfold(x, self.c[0].kernel_size, self.c[0].dilation, self.c[0].padding, self.c[0].stride)

    @staticmethod
    def memse(conv2duf: Conv2DUF, memse_dict):
        mu, gamma, gamma_shape = memse_dict['mu'], memse_dict['gamma'], memse_dict['gamma_shape']
        ct = conv2duf.weight.learnt_Gmax / conv2duf.weight.Wmax  # Only one column at most (vector of weights)
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
                                                  conv2duf.original_weight
                                                  )

        memse_dict['P_tot'] += P_tot
        memse_dict['current_type'] = 'Conv2DUF'
        memse_dict['mu'] = mu
        memse_dict['gamma'] = gamma
        memse_dict['gamma_shape'] = gamma_shape

    @staticmethod
    def mse_var(conv2duf: Conv2DUF, input, gamma, gamma_shape, r, c0, weights):
        mu = conv2duf(input, weights) * r

        # TODO
        # gamma only diag kernels at the end
        if gamma_shape is None:
            gamma_n = double_conv(gamma, weights, **conv2duf.conv_property_dict)
        else:
            gamma_n = torch.zeros(mu.shape + mu.shape[1:], device=mu.device, dtype=mu.dtype)
        gamma_n = conv2duf_op(gamma_n, gamma, gamma_shape, input, c0, weight_shape=weights.shape, **conv2duf.conv_property_dict) * r ** 2
        return mu, gamma_n, None

    @staticmethod
    def energy(conv2duf: Conv2DUF,
               x: TensorType["batch", "channel_in", "width", "height"],
               gamma: TensorType["batch", "channel_in", "width", "height", "channel_in", "width", "height"],
               gamma_shape,
               c: TensorType["channel_out"],
               w: TensorType["channel_out", "channel_in", "width", "height"],
               r,
               sigma):
        sum_x_gd: torch.TensorType = x ** 2
        if gamma_shape is None:
            sum_x_gd += gamma_to_diag(gamma)
        abs_w: torch.TensorType = oe.contract('coij,c->coij', torch.abs(w), c).to(sum_x_gd)
        e_p_mem: torch.TensorType = torch.sum(torch.nn.functional.conv2d(sum_x_gd, abs_w, **conv2duf.conv_property_dict), dim=(1,2,3))

        Gpos = torch.clip(w, min=0)
        Gneg = torch.clip(-w, min=0)
        zp_mu, zp_g, _ = conv2duf.mse_var(conv2duf, x, gamma, gamma_shape, r, (sigma/c) ** 2, Gpos)
        zm_mu, zm_g, _ = conv2duf.mse_var(conv2duf, x, gamma, gamma_shape, r, (sigma/c) ** 2, Gneg)

        e_p_tiap = torch.sum((zp_mu ** 2 + gamma_to_diag(zp_g)) /  r, dim=(1,2,3))
        e_p_tiam = torch.sum((zm_mu ** 2 + gamma_to_diag(zm_g)) /  r, dim=(1,2,3))
        return e_p_mem + e_p_tiap + e_p_tiam
