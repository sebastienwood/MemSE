CUDASIM = False

if CUDASIM:
    import os
    os.environ['NUMBA_ENABLE_CUDASIM'] = '1'

import os
from typing import List, Tuple

import math
import numba
import torch
import opt_einsum as oe
from torch.autograd import Function
from numba import njit, prange, cuda

from MemSE.nn.op.conv2duf_opt import op_numba_w

module_path = os.path.dirname(__file__)


@njit(parallel=True, nogil=True, boundscheck=False, fastmath=True)
def op_numba(input, gamma, mu, c, weight_shape_1, weight_shape_2, weight_shape_3, bias: bool, padding, stride):
    for bi in prange(input.shape[0]):
        for i0 in prange(input.shape[2]):
            for j0 in prange(input.shape[3]):
                for i0p in range(input.shape[5]):
                    for j0p in range(input.shape[6]):
                        strided_i0 = i0 * stride[0]
                        strided_i0p = i0p * stride[0]
                        strided_j0 = j0 * stride[1]
                        strided_j0p = j0p * stride[1]
                        v = numba.float32(0.)
                        for ii in range(weight_shape_2):
                            # Virtual padding
                            i0ii = strided_i0 + ii
                            i0pii = strided_i0p + ii
                            oob_0 = i0ii < padding[0] or i0ii >= mu.shape[2] + padding[0]
                            oob_0p = i0pii < padding[0] or i0pii >= mu.shape[2] + padding[0]
                            if oob_0 or oob_0p:
                                continue
                            for ji in range(weight_shape_3):
                                j0ji = strided_j0 + ji
                                j0pji = strided_j0p + ji
                                oob_0j = oob_0 or j0ji < padding[1] or j0ji >= mu.shape[3] + padding[1]
                                oob_0pj = oob_0p or j0pji < padding[1] or j0pji >= mu.shape[3] + padding[1]
                                if not oob_0j and not oob_0pj:
                                    # Recenter on actual coords
                                    i0ii_padded = i0ii - padding[0]
                                    j0ji_padded = j0ji - padding[1]
                                    i0pii_padded = i0pii - padding[0]
                                    j0pji_padded = j0pji - padding[1]
                                    for ci in range(weight_shape_1):
                                        v += (mu[bi, ci, i0ii_padded, j0ji_padded] * mu[bi, ci, i0pii_padded, j0pji_padded] + gamma[bi, ci, i0ii_padded, j0ji_padded, ci, i0pii_padded, j0pji_padded])
                        if bias:
                            v += 1.
                        for c0 in range(input.shape[1]):
                            input[bi, c0, i0, j0, c0, i0p, j0p] += c[c0] * v
    return input


@cuda.jit
def op_numba_c_f(input, gamma, c, weight_shape_1, weight_shape_2, weight_shape_3, bias: bool, padding, stride):
    sharedC = cuda.shared.array(shape=512, dtype=numba.float32)
    bi, i0, j0 = cuda.grid(3)
    x_gridsize, y_gridsize, z_gridsize = cuda.gridsize(3)

    if cuda.threadIdx.y == 0 and cuda.threadIdx.z == 0:
        tB = cuda.threadIdx.x
        while tB < c.shape[0]:
            sharedC[tB] = c[tB]
            tB += cuda.blockDim.x
    cuda.syncthreads()

    if bi > input.shape[0] or i0 > input.shape[2] or j0 > input.shape[3]:
        return

    while bi < input.shape[0]:
        while i0 < input.shape[2]:
            while j0 < input.shape[3]:
                # Iterate on io j0 i0p j0p
                for i0p in range(input.shape[5]):
                    for j0p in range(input.shape[6]):
                        strided_i0 = i0 * stride[0]
                        strided_i0p = i0p * stride[0]
                        strided_j0 = j0 * stride[1]
                        strided_j0p = j0p * stride[1]
                        v = numba.float32(0.)
                        for ii in range(weight_shape_2):
                            # Virtual padding
                            i0ii = strided_i0 + ii
                            i0pii = strided_i0p + ii
                            oob_0 = i0ii < padding[0] or i0ii >= gamma.shape[1] + padding[0]
                            oob_0p = i0pii < padding[0] or i0pii >= gamma.shape[1] + padding[0]
                            if oob_0 or oob_0p:
                                continue
                            for ji in range(weight_shape_3):
                                j0ji = strided_j0 + ji
                                j0pji = strided_j0p + ji
                                oob_0j = oob_0 or j0ji < padding[1] or j0ji >= gamma.shape[2] + padding[1]
                                oob_0pj = oob_0p or j0pji < padding[1] or j0pji >= gamma.shape[2] + padding[1]
                                if not oob_0j and not oob_0pj:
                                    # Recenter on actual coords
                                    i0ii_padded = i0ii - padding[0]
                                    j0ji_padded = j0ji - padding[1]
                                    i0pii_padded = i0pii - padding[0]
                                    j0pji_padded = j0pji - padding[1]
                                    gamma_cache = gamma[bi, i0ii_padded, j0ji_padded, i0pii_padded, j0pji_padded]
                                    for ci in range(weight_shape_1):
                                        v += gamma_cache[ci]
                        # For each c0 update input
                        if bias:
                            v += 1.
                        for c0 in range(input.shape[1]):
                            input[bi, c0, i0, j0, c0, i0p, j0p] += v * sharedC[c0]
                            #cuda.atomic.add(input, (bi, c0, i0, j0, c0, i0p, j0p), v * sharedC[c0])
                j0 += z_gridsize
            i0 += y_gridsize
        bi += x_gridsize


class Conv2DUF_op(Function):
    @staticmethod
    def forward(ctx, input, gamma, mu, c, weight_shape, bias: bool, padding, stride):
        return torch.from_numpy(op_numba(input.numpy(), gamma.numpy(), mu.numpy(), c.numpy(), weight_shape[1], weight_shape[2], weight_shape[3], bias, padding=padding, stride=stride))

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out


TEST = False
class Conv2DUF_op_CUDA(Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, gamma: torch.Tensor, gamma_shape, mu: torch.Tensor, c, weight_shape, bias: bool, padding, stride):
        gamma_permute = torch.diagonal(oe.contract('bcij,bklm->bijlmck', mu, mu), dim1=-2, dim2=-1)
        if gamma_shape is None:
            gamma_permute += torch.diagonal(torch.permute(gamma, (0, 2, 3, 5, 6, 1, 4)), dim1=-2, dim2=-1)
        cuda.select_device(int(str(input.device).split(':')[1]))
        if TEST:
            threadsperblock= (8, 128)
            grid_bi = input.shape[0] * input.shape[2] * input.shape[3] * input.shape[5] * input.shape[6]
            blockspergrid_x = min(math.ceil((weight_shape[1] + threadsperblock[0] - 1) / threadsperblock[0]), 1024)
            blockspergrid_y = min(math.ceil((grid_bi + threadsperblock[1] - 1)/ threadsperblock[1]), 1024)
            blockspergrid = (blockspergrid_x, blockspergrid_y)
            op_numba_w[blockspergrid, threadsperblock](input.detach(), gamma_permute.detach(), c.detach(), weight_shape[1], weight_shape[2], weight_shape[3], bias, padding, stride)
        else:
            threadsperblock= (8,8,8)# 4)
            blockspergrid_x = math.ceil(input.shape[0] / threadsperblock[0])
            blockspergrid_y = math.ceil(input.shape[2] / threadsperblock[1])
            blockspergrid_z = math.ceil(input.shape[3] / threadsperblock[2])
            blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
            op_numba_c_f[blockspergrid, threadsperblock](input.detach(), gamma_permute.detach(), c.detach(), weight_shape[1], weight_shape[2], weight_shape[3], bias, padding, stride)
        return input

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out


def next_power_of_2(x):
    return 1<<(x-1).bit_length()


def conv2duf_op(input, gamma, gamma_shape, mu, c, weight_shape, bias, stride: Tuple[int]=(1,1), padding: int = 0, dilation: int = 1, groups: int = 1):
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    assert all(d == 1 for d in dilation), 'Dilation != 1 not supported yet'
    if isinstance(padding, int):
        padding = (padding, padding)
    assert groups == 1, 'Groups != 1 not supported yet'

    if c.dim() == 0:
        c = c.repeat(input.shape[1])
    assert input.shape[1] == c.shape[0]
    assert weight_shape[1] == mu.shape[1]
    assert weight_shape[0] == input.shape[1]
    bias = True if bias is not None else False

    if input.device.type == "cpu":
        # NOTE @sebastienwood "cpu" version is kept intact as a 100% sure version of the code
        # it could be easily optimized using the latests ideas
        # major drawback is DRAM usage: gamma is always initialized 
        gamma = gamma if gamma_shape is None else torch.zeros(gamma_shape, device=mu.device, dtype=mu.dtype)
        Conv2DUF_op.apply(input, gamma, mu, c, weight_shape, bias, padding, stride)
        return input
    else:
        input = Conv2DUF_op_CUDA.apply(input, gamma, gamma_shape, mu, c.to(input), weight_shape, bias, padding, stride)
        return input


if __name__ == '__main__':
    from time import time
    import numpy as np
    wh = 32
    ch = 3
    bs = 16
    padding = 0
    stride = 1
    kernel = 3
    who = (wh - kernel + 2 * padding) / stride + 1

    input = torch.rand(bs, ch, who, who, ch, who, who)
    gamma = torch.rand(bs, ch, wh, wh, ch, wh, wh)
    mu = torch.rand(bs, ch, wh, wh)
    c = torch.tensor(1.)
    if c.dim() == 0:
        c = c.repeat(input.shape[1])
    weight_shape = [ch, ch, 3, 3]
    res = []

    profile = False
    dd = [torch.device('cuda:0')]
    if not profile:
        dd.insert(0, torch.device('cpu'))
    for d in dd:
        timings = []
        input_, gamma, mu, c = input.clone().to(d), gamma.to(d), mu.to(d), c.to(d)
        res.append(conv2duf_op(input_, gamma, mu, c, weight_shape).to('cpu'))
        # for _ in range(100):
        #     start = time()
        #     conv2duf_op(input_, gamma, mu, c, weight_shape)
        #     timings.append(time() - start)
                
        # median_time = np.median(timings)
        # print(f'Median time is {median_time}')
    assert torch.allclose(res[0], res[1]), torch.mean((res[0] - res[1]) ** 2)
