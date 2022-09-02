CUDASIM = False

if CUDASIM:
    import os
    os.environ['NUMBA_ENABLE_CUDASIM'] = '1'

import os
import timeit
from typing import List, Tuple

import math
import numba
import torch
from torch.autograd import Function
from numba import njit, prange, cuda


TUBE = None

module_path = os.path.dirname(__file__)


@njit(parallel=True, nogil=True, boundscheck=False, fastmath=True)
def op_numba(input, gamma, mu, c, weight_shape_1, weight_shape_2, weight_shape_3, padding, stride):
    for bi in prange(input.shape[0]):
        for i0 in prange(input.shape[2]):
            for j0 in prange(input.shape[3]):
                for i0p in range(input.shape[5]):
                    for j0p in range(input.shape[6]):
                        strided_i0 = i0 * stride[0]
                        strided_i0p = i0p * stride[0]
                        strided_j0 = j0 * stride[1]
                        strided_j0p = j0p * stride[1]
                        v = 0.
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
                        for c0 in range(input.shape[1]):
                            input[bi, c0, i0, j0, c0, i0p, j0p] += c[c0] * v
    return input


@cuda.jit
def op_numba_c(input, gamma, mu, c, weight_shape_1, weight_shape_2, weight_shape_3, padding, stride):
    sharedC = cuda.shared.array(shape=1024, dtype=numba.float32)
    bi = cuda.grid(1)
    x_gridsize = cuda.gridsize(1)
    threadB = cuda.threadIdx.x

    if threadB < c.shape[0]:
        sharedC[threadB] = c[threadB]
    if cuda.blockDim.x < c.shape[0] and threadB == 0:  # not covering all values
        for i in range(cuda.blockDim.x, c.shape[0]):
            sharedC[i] = c[i]

    cuda.syncthreads()

    if bi > input.shape[0]:
        return

    while bi < input.shape[0]:
        # Iterate on io j0 i0p j0p
        for i0 in range(input.shape[2]):
            for j0 in range(input.shape[3]):
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
                        for c0 in range(input.shape[1]):
                            cuda.atomic.add(input, (bi, c0, i0, j0, c0, i0p, j0p), v * sharedC[c0])
        bi += x_gridsize


@cuda.jit
def op_numba_c_f(input, gamma, mu, c, weight_shape_1, weight_shape_2, weight_shape_3, padding, stride):
    sharedC = cuda.shared.array(shape=1024, dtype=numba.float32)
    bi, i0, j0 = cuda.grid(3)
    x_gridsize, y_gridsize, z_gridsize = cuda.gridsize(3)
    threadB = cuda.threadIdx.x

    if threadB < c.shape[0]:
        sharedC[threadB] = c[threadB]
    if cuda.blockDim.x < c.shape[0] and threadB == 0:  # not covering all values
        for i in range(cuda.blockDim.x, c.shape[0]):
            sharedC[i] = c[i]

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
                                    mu_cache = mu[bi]
                                    gamma_cache = gamma[bi, i0ii_padded, j0ji_padded, i0pii_padded, j0pji_padded]
                                    # TODO idea: ci last format for coalesced accesses
                                    # TODO shared memory for block at least on bi
                                    for ci in range(weight_shape_1):
                                        #v += (mu_cache[ci, i0ii_padded, j0ji_padded] * mu_cache[ci, i0pii_padded, j0pji_padded] + gamma[bi, ci, i0ii_padded, j0ji_padded, ci, i0pii_padded, j0pji_padded])
                                        v += (mu_cache[ci, i0ii_padded, j0ji_padded] * mu_cache[ci, i0pii_padded, j0pji_padded] + gamma_cache[ci])
                        # For each c0 update input
                        for c0 in range(input.shape[1]):
                            cuda.atomic.add(input, (bi, c0, i0, j0, c0, i0p, j0p), v * sharedC[c0])
                j0 += z_gridsize
            i0 += y_gridsize
        bi += x_gridsize


class Conv2DUF_op(Function):
    @staticmethod
    def forward(ctx, input, gamma, mu, c, weight_shape, padding, stride):
        return torch.from_numpy(op_numba(input.numpy(), gamma.numpy(), mu.numpy(), c.numpy(), weight_shape[1], weight_shape[2], weight_shape[3], padding=padding, stride=stride))

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out


class Conv2DUF_op_CUDA(Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, gamma: torch.Tensor, mu: torch.Tensor, c, weight_shape, padding, stride):
        cuda.select_device(int(str(input.device).split(':')[1]))
        # TODO batch heavy/small filters is longer in any case
        if False and input.shape[2] * input.shape[3] < input.shape[0]:
            threadsperblock= (min(1024,next_power_of_2(input.shape[0])),)# 4)
            blockspergrid_x = math.ceil(input.shape[0] / threadsperblock[0])

            blockspergrid = (blockspergrid_x,)# blockspergrid_y)
            op_numba_c[blockspergrid, threadsperblock](input, gamma, mu, c, weight_shape[1], weight_shape[2], weight_shape[3], padding, stride)
        else:
            threadsperblock= (8,8,8)# 4)
            blockspergrid_x = math.ceil(input.shape[0] / threadsperblock[0])
            blockspergrid_y = math.ceil(input.shape[2] / threadsperblock[1])
            blockspergrid_z = math.ceil(input.shape[3] / threadsperblock[2])

            blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

            with torch.no_grad():
                gamma_permute = torch.permute(gamma.detach(), (0, 2, 3, 5, 6, 1, 4))
                gamma_permute = torch.diagonal(gamma_permute, dim1=-2, dim2=-1)

            op_numba_c_f[blockspergrid, threadsperblock](input.detach(), gamma_permute.detach(), mu.detach(), c.detach(), weight_shape[1], weight_shape[2], weight_shape[3], padding, stride)
        return input

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out


def next_power_of_2(x):
    return 1<<(x-1).bit_length()


def conv2duf_op(input, gamma, mu, c, weight_shape, stride: Tuple[int]=(1,1), padding: int = 0, dilation: int = 1, groups: int = 1, **kwargs):
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

    if input.device.type == "cpu":
        Conv2DUF_op.apply(input, gamma, mu, c, weight_shape, padding, stride)
        return input
    else:
        input = Conv2DUF_op_CUDA.apply(input, gamma, mu, c.to(input), weight_shape, padding, stride)
        return input


if __name__ == '__main__':
    from time import time
    import numpy as np
    wh = 32
    who = wh - 2
    ch = 3
    bs = 32

    input = torch.rand(bs, ch, who, who, ch, who, who)
    gamma = torch.rand(bs, ch, wh, wh, ch, wh, wh)
    mu = torch.rand(bs, ch, wh, wh)
    c = torch.tensor(1.)
    if c.dim() == 0:
        c = c.repeat(input.shape[1])
    weight_shape = [ch, ch, 3, 3]

    for d in [torch.device('cpu'), torch.device('cuda:0')]:
        timings = []
        input, gamma, mu, c = input.to(d), gamma.to(d), mu.to(d), c.to(d)
        for _ in range(100):
            start = time()
            conv2duf_op(input, gamma, mu, c, weight_shape)
            timings.append(time() - start)
                
        median_time = np.median(timings)
        print(f'Median time is {median_time}')
