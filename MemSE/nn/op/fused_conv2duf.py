from typing import Tuple
from torchtyping import TensorType
from itertools import product

import torch
import torch.nn as nn
import math

from MemSE.nn import Conv2DUF

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
                   c: TensorType["channel_out"],
                   sigma: float,
                   r: float,
                   padding: Tuple[int, int], 
                   stride: Tuple[int, int]
                   ):
    ''' A kernel that perform double_conv + conv2duf_op at once, for each case of MSE+energy loops
    '''
    # TODO init out_gamma here, shape full
    # TODO stride, virtual padding
    r_2 = r ** 2
    for bi in range(out_gamma.shape[0]):
        pi = 0
        for ci in range(out_gamma.shape[1]):
            ratio_mse = (sigma * SQRT_2) ** 2 / (c[ci] ** 2)
            ratio_p = (sigma / c[ci]) ** 2
            for ii in range(out_gamma.shape[2]):
                strided_ii = ii * stride[0]
                for ji in range(out_gamma.shape[3]):
                    strided_ij = ij * stride[0]
                    for cj in range(out_gamma.shape[4]):
                        for ij in range(out_gamma.shape[5]):
                            strided_ji = ji * stride[1]
                            for jj in range(out_gamma.shape[6]):
                                flag_diag = ci == cj and ii == ij and ji == jj  # stride invariant
                                strided_jj = jj * stride[1]
                                og = 0.
                                om = 0.
                                for kc in range(weights.shape[1]): # TODO the loop could intertwine as mkmk instead of mmkk currently
                                    for ki in range(weights.shape[2]):
                                        iiki = strided_ii + ki
                                        ijki = strided_ij + ki
                                        oob_0 = iiki < padding[0] or iiki >= in_gamma.shape[2] + padding[0]
                                        oob_0p = ijki < padding[0] or ijki >= in_gamma.shape[2] + padding[0]
                                        if oob_0 or oob_0p:
                                            continue
                                        for kj in range(weights.shape[3]):
                                            jikj = strided_ji + kj
                                            jjkj = strided_jj + kj
                                            oob_0j = oob_0 or jikj < padding[1] or jikj >= in_gamma.shape[3] + padding[1]
                                            oob_0pj = oob_0p or jjkj < padding[1] or jjkj >= in_gamma.shape[3] + padding[1]
                                            if not oob_0j and not oob_0pj:
                                                # Recenter on actual coords
                                                iiki_padded = iiki - padding[0]
                                                ijki_padded = ijki - padding[0]
                                                jikj_padded = jikj - padding[1]
                                                jjkj_padded = jjkj - padding[1]
                                            w_l = weights[ci,kc,ki,kj]
                                            if ci == cj:
                                                ml = in_mu[bi, kc, ii+ki, ji+kj]
                                                mmg = (ml * in_mu[bi, kc, ij+ki, jj+kj] + in_gamma[bi, kc, ii+ki, ji+kj, kc, ij+ki, jj+kj])
                                                og += ratio_mse * mmg
                                                if flag_diag:
                                                    pi += (ratio_p + abs(w_l) * c[ci]) * mmg + (ml * abs(w_l)) ** 2
                                                    om += ml * w_l
                                            for kl in range(weights.shape[1]):
                                                for km in range(weights.shape[2]):
                                                    for kn in range(weights.shape[3]):
                                                        w_r = weights[cj, kl, km, kn]
                                                        g = in_gamma[bi, kc, ii+ki, ji+kj, kl, ij+km, jj+kn]
                                                        og += w_l * w_r * g
                                                        if flag_diag:
                                                            # TODO: if same sign pi += w_l * w_r * g
                                                            # ww = w_l * w_r
                                                            # if ww > 0:
                                                            # pi += ww * g
                                                            w_l_p, w_r_p = min(0, w_l), min(0, w_r)
                                                            w_l_n, w_r_n = min(0, -w_l), min(0, -w_r)
                                                            pi += w_l_p * w_r_p * g + w_l_n * w_r_n * g
                                out_gamma[bi, ci, ii, ji, cj, ij, jj] += og * r_2
                                out_mu[bi, ci, ii, ji] += om * r
        out_p[bi] += pi * r
    return out_gamma


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
    out_p = torch.zeros(bs).to(device)
    in_gamma = torch.rand(bs, ch, wh, wh, ch, wh, wh).to(device)
    in_mu = torch.rand(bs, ch, wh, wh).to(device)
    c = torch.tensor(1.).to(device)
    if c.dim() == 0:
        c = c.repeat(out_gamma.shape[1])
    sigma = 0.01
    r = 1.2
    conv = nn.Conv2d(ch, ch, kernel_size=kernel, padding=padding, stride=stride).to(device)
    assert who == conv(in_mu).shape[2]
    conv2duf = Conv2DUF(conv, in_mu.shape, torch.Size([ch, who, who]))
    weight_shape = conv.weight.shape
    weights = conv.weight

    res = []

    res.append(conv2duf.mse_var(conv2duf, in_mu, in_gamma, None, r, sigma/c, weights)[1])
    print('Done ref')
    res.append(fused_conv2duf(out_gamma.clone(), out_p.clone(), in_mu, in_gamma, weights, c, sigma, r))
    print(res[0].shape)
    print(res[1].shape)
    assert torch.allclose(res[0], res[1].to(res[0])), torch.max((res[0] - res[1]) ** 2)
