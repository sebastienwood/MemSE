from typing import Tuple, Optional
from torchtyping import TensorType
from itertools import product

import torch
import torch.nn as nn
import math

from MemSE.nn import Conv2DUF
from MemSE import MemristorQuant, MemSE

SQRT_2 = math.sqrt(2)


def ref_fused_conv2duf(out_gamma: TensorType["batch", "channel_out", "width_out", "height_out", "channel_out", "width_out", "height_out"],
                   out_p: TensorType["batch"],
                   in_mu: TensorType["batch", "channel_in", "width_in", "height_in"],
                   in_gamma: TensorType["batch", "channel_in", "width_in", "height_in", "channel_in", "width_in", "height_in"],
                   weights: TensorType["channel_out", "channel_in", "width", "height"],
                   c: TensorType["channel_out"],
                   sigma: float,
                   r: float,
                   ):
    ''' A kernel that perform double_conv + conv2duf_op at once, for each case of MSE+energy loops
    '''
    for bi in range(out_gamma.shape[0]):
        for ci in range(out_gamma.shape[1]):
            ratio = sigma / c[ci]
            for ii in range(out_gamma.shape[2]):
                for ji in range(out_gamma.shape[3]):
                    for cj in range(out_gamma.shape[4]):
                        for ij in range(out_gamma.shape[5]):
                            for jj in range(out_gamma.shape[6]): 
                                for kc in range(weights.shape[1]):
                                    for ki in range(weights.shape[2]):
                                        for kj in range(weights.shape[3]):
                                            if ci == cj:
                                                out_gamma[bi, ci, ii, ji, cj, ij, jj] += ratio * (in_mu[bi, kc, ii+ki, ji+kj] * in_mu[bi, kc, ij+ki, jj+kj] + in_gamma[bi, kc, ii+ki, ji+kj, kc, ij+ki, jj+kj])
												
                                            for kl in range(weights.shape[1]):
                                                for km in range(weights.shape[2]):
                                                    for kn in range(weights.shape[3]):
                                                        out_gamma[bi, ci, ii, ji, cj, ij, jj] += weights[ci,kc,ki,kj] * weights[cj, kl, km, kn] * in_gamma[bi, kc, ii+ki, ji+kj, kl, ij+km, jj+kn]
									
                    # DIAGONALE == VAR
                    out_gamma[bi, ci, ii, ji, cj, ij, jj] = 0
                    for kc in range(weights.shape[1]):
                        for ki in range(weights.shape[2]):
                            for kj in range(weights.shape[3]):
                                g_2 = in_gamma[bi, kc, ii+ki, ji+kj, kc, ii+ki, ji+kj]
                                out_gamma[bi, ci, ii, ji, ci, ii, ji] += ratio * (in_mu[bi, kc, ii+ki, ji+kj]**2 + g_2) + g_2 * weights[ci, kc, ki, kj] ** 2
                                for kl in range(weights.shape[1]):
                                    for km in range(weights.shape[2]):
                                        for kn in range(weights.shape[3]):
                                            if kc != kl or ki != km or kj != kn:
                                                out_gamma[bi, ci, ii, ji, ci, ii, ji] += weights[ci,kc,ki,kj] * weights[ci, kl, km, kn] * in_gamma[bi, ci, ii+ki, ji+kj, cj, ii+km, ji+kn]
    out_gamma *= r ** 2


def fused_conv2duf(out_gamma: TensorType["batch", "channel_out", "width_out", "height_out", "channel_out", "width_out", "height_out"],
                   out_p: TensorType["batch"],
                   out_mu: TensorType["batch", "channel_out", "width_out", "height_out"],
                   in_mu: TensorType["batch", "channel_in", "width_in", "height_in"],
                   in_gamma: TensorType["batch", "channel_in", "width_in", "height_in", "channel_in", "width_in", "height_in"],
                   weights: TensorType["channel_out", "channel_in", "width", "height"],
                   bias: Optional[TensorType["channel_out"]],
                   c: TensorType["channel_out"],
                   sigma: float,
                   r: float,
                   padding: Tuple[int, int], 
                   stride: Tuple[int, int]
                   ):
    ''' A kernel that perform double_conv + conv2duf_op at once, for each case of MSE+energy loops
    '''
    # TODO init out_gamma here, shape full
    # TODO reduce by symmetry, store twice if needed
    # TODO fused with relu
    r_2 = r ** 2
    for bi in range(out_gamma.shape[0]):
        pi = 0
        for ci in range(out_gamma.shape[1]):
            ratio_mse = (sigma * SQRT_2) ** 2 / (c[ci] ** 2)
            ratio_p = (sigma / c[ci]) ** 2
            if bias is not None:
                b = bias[ci]
                abs_b = c[ci] * b
                b_p, b_n = min(0, b), min(0, -b)
            for ii in range(out_gamma.shape[2]):
                strided_ii = ii * stride[0]
                for ji in range(out_gamma.shape[3]):
                    strided_ji = ji * stride[1]
                    for cj in range(out_gamma.shape[4]):
                        for ij in range(out_gamma.shape[5]):
                            strided_ij = ij * stride[0]
                            for jj in range(out_gamma.shape[6]):
                                flag_diag = ci == cj and ii == ij and ji == jj  # stride invariant
                                strided_jj = jj * stride[1]
                                og = 0.
                                om = 0.
                                pi_patch_p = 0.
                                pi_patch_n = 0.
                                for kc in range(weights.shape[1]): # TODO the loop could intertwine as mkmk instead of mmkk currently
                                    for ki in range(weights.shape[2]):
                                        iiki = strided_ii + ki
                                        oob_0 = iiki < padding[0] or iiki >= in_gamma.shape[2] + padding[0]
                                        if oob_0:
                                            continue

                                        for kj in range(weights.shape[3]):
                                            w_l = weights[ci,kc,ki,kj]
                                            w_l_p, w_l_n = min(0, w_l), min(0, -w_l)
                                            jikj = strided_ji + kj
                                            oob_0j = jikj < padding[1] or jikj >= in_gamma.shape[3] + padding[1]
                                            if oob_0j:
                                                continue

                                            iiki_padded = iiki - padding[0]
                                            jikj_padded = jikj - padding[1]
                                            ml = in_mu[bi, kc, iiki_padded, jikj_padded]
                                            if flag_diag:  # to perform it once only
                                                pi_patch_p += (ml * w_l_p)  # the 2 convolutions with G+ and G-
                                                pi_patch_n += (ml * w_l_n)
                                                om += ml * w_l  # the regular convolution

                                            for kl in range(weights.shape[1]):
                                                for km in range(weights.shape[2]):
                                                    ijkm = strided_ij + km
                                                    oob_0p = ijkm < padding[0] or ijkm >= in_gamma.shape[2] + padding[0]
                                                    if oob_0p:
                                                        continue
                                                    for kn in range(weights.shape[3]):
                                                        jjkn = strided_jj + kn
                                                        oob_0pj = jjkn < padding[1] or jjkn >= in_gamma.shape[3] + padding[1]
                                                        if oob_0pj:
                                                            continue

                                                        flag_conv_delta = ki == km and kj == kn and kc == kl
                                                        ijkm_padded = ijkm - padding[0]
                                                        jjkn_padded = jjkn - padding[1]

                                                        g = in_gamma[bi, kc, iiki_padded, jikj_padded, kl, ijkm_padded, jjkn_padded]

                                                        if ci == cj and flag_conv_delta:
                                                            mmg = (ml * in_mu[bi, kc, ijkm_padded, jjkn_padded] + g)
                                                            og += ratio_mse * mmg  # ~ what conv2dufop does

                                                            if flag_diag:  # doing the convolution at beginning of energy f(mu ** 2 + diag(gamma) == mmg, abs(w)*c) +  conv2dufop for power ratio_p * mmg
                                                                pi += (ratio_p + abs(w_l) * c[ci]) * mmg

                                                        w_r = weights[cj, kl, km, kn]
                                                        
                                                        og += w_l * w_r * g  # double conv
                                                        if flag_diag:
                                                            # TODO: if same sign pi += w_l * w_r * g
                                                            # ww = w_l * w_r
                                                            # if ww > 0:
                                                            # pi += ww * g
                                                            w_r_p = min(0, w_r)
                                                            w_r_n = min(0, -w_r)
                                                            pi += w_l_p * w_r_p * g + w_l_n * w_r_n * g  # double conv for energy
                                if bias is not None and ci == cj:
                                    og += 1.
                                out_gamma[bi, ci, ii, ji, cj, ij, jj] += og * r_2
                                if flag_diag:
                                    if bias is not None:
                                        pi_patch_n += 1 + b_n.
                                        pi_patch_p += 1 + b_p.
                                        om += b
                                    out_mu[bi, ci, ii, ji] += om * r
                                    pi += pi_patch_p ** 2 + pi_patch_n ** 2 
        out_p[bi] += pi * r
    return out_mu, out_gamma, out_p


def fused_conv2duf_low_storage():
    ''' Perform the operation on out_gamma of reduced size by exploiting symmetry
    '''
    pass


def fromMatrixToVector(i: int, j: int, N: int):
   if (i <= j):
      return i * N - (i - 1) * i / 2 + j - i
   else:
      return j * N - (j - 1) * j / 2 + i - j


if __name__ == '__main__':
    wh = 6
    ch = 2
    bs = 2
    padding = 0
    stride = 1
    kernel = 3
    who = int((wh - kernel + 2 * padding) / stride + 1)

    device = torch.device('cuda:0')

    out_gamma = torch.zeros(bs, ch, who, who, ch, who, who).to(device)
    out_mu = torch.zeros(bs, ch, who, who).to(device)
    out_p = torch.zeros(bs).to(device)
    in_gamma = torch.rand(bs, ch, wh, wh, ch, wh, wh).to(device)
    in_mu = torch.ones(bs, ch, wh, wh).to(device)
    
    sigma = 0.01
    r = 1.2
    conv = nn.Conv2d(ch, ch, kernel_size=kernel, padding=padding, stride=stride, bias=False).to(device)
    conv.weight.data.fill_(1.)
    assert who == conv(in_mu).shape[2]
    conv2duf = Conv2DUF(conv, in_mu.shape, torch.Size([ch, who, who]))
    quanter = MemristorQuant(conv2duf, std_noise=sigma)
    # _ = MemSE.init_learnt_gmax(quanter)
    quanter.quant()
    c = conv2duf.weight.learnt_Gmax / conv2duf.weight.Wmax
    if c.dim() == 0:
        c = c.repeat(out_gamma.shape[1])
    memse_dict = {
        'mu': in_mu,
        'gamma': in_gamma,
        'gamma_shape': None,
        'P_tot': out_p.clone(),
        'r': r,
        'sigma': sigma,
        'compute_power': True
    }
    weights = conv.weight

    conv2duf.memse(conv2duf, memse_dict)
    m, g, p = memse_dict['mu'], memse_dict['gamma'], memse_dict['P_tot']
    print('Done ref')
    assert torch.all(out_gamma == 0.) and torch.all(out_p == 0.) and torch.all(out_mu == 0.)
    m_, g_, p_ = fused_conv2duf(out_gamma.clone(), out_p.clone(), out_mu.clone(), in_mu, in_gamma, weights, c, sigma, r, (padding, padding), (stride, stride))

    print(g[0])
    print('*'*10)
    print(g_[0])

    print(p[0])
    print('*'*10)
    print(p_[0])
    assert torch.allclose(m, m_.to(m)), torch.max((m - m_) ** 2)
    assert torch.allclose(g, g_.to(g)), torch.max((g - g_) ** 2)
    assert torch.allclose(p, p_.to(p)), torch.max((p - p_) ** 2)
