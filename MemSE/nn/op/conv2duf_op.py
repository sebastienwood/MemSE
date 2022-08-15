CUDASIM = False

if CUDASIM:
    import os
    os.environ['NUMBA_ENABLE_CUDASIM'] = '1'

import os
import timeit
from typing import List

import math
import numba
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function
from torch.utils.cpp_extension import load
from torch.profiler import profile, record_function, ProfilerActivity
from numba import njit, prange, cuda


TUBE = None

module_path = os.path.dirname(__file__)


def op_slow(input, gamma, mu, c, weight_shape):
    if c.dim() == 0:
        c = c.repeat(input.shape[1])
    for bi in range(input.shape[0]):
        for c0 in range(input.shape[1]):
            for i0 in range(input.shape[2]):
                for j0 in range(input.shape[3]):
                    for i0p in range(input.shape[5]):
                        for j0p in range(input.shape[6]):
                            for ci in range(weight_shape[1]):
                                for ii in range(weight_shape[2]):
                                    for ji in range(weight_shape[3]):
                                        input[bi, c0, i0, j0, c0, i0p, j0p] += c[c0] * (mu[bi, ci, i0+ii, j0+ji] * mu[bi, ci, i0p+ii, j0p+ji] + gamma[bi, ci, i0+ii, j0+ji, ci, i0p+ii, j0p+ji])
    return input


@njit(parallel=True, nogil=True, boundscheck=False, fastmath=True)
def op_numba(input, gamma, mu, c, weight_shape_1, weight_shape_2, weight_shape_3):
    for bi in prange(input.shape[0]):
        for i0 in prange(input.shape[2]):
            for j0 in prange(input.shape[3]):
                for i0p in range(input.shape[5]):
                    for j0p in range(input.shape[6]):
                        v = 0.
                        for ci in range(weight_shape_1):
                            for ii in range(weight_shape_2):
                                for ji in range(weight_shape_3):
                                    v += (mu[bi, ci, i0+ii, j0+ji] * mu[bi, ci, i0p+ii, j0p+ji] + gamma[bi, ci, i0+ii, j0+ji, ci, i0p+ii, j0p+ji])
                        for c0 in range(input.shape[1]):
                            input[bi, c0, i0, j0, c0, i0p, j0p] += c[c0] * v
    return input


@cuda.jit
def op_numba_c(input, gamma, mu, c, weight_shape_1, weight_shape_2, weight_shape_3):
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
                        v = numba.float32(0.)
                        for ci in range(weight_shape_1):
                            # threadM = mu[bi, ci]
                            # threadG = gamma[bi, ci]
                            for ii in range(weight_shape_2):
                                for ji in range(weight_shape_3):
                                    v += (mu[bi, ci, i0+ii, j0+ji] * mu[bi, ci, i0p+ii, j0p+ji] + gamma[bi, ci, i0+ii, j0+ji, ci, i0p+ii, j0p+ji])
                        # For each c0 update input
                        for c0 in range(input.shape[1]):
                            #input[bi, c0, i0, j0, c0, i0p, j0p] += v * sharedC[c0]
                            cuda.atomic.add(input, (bi, c0, i0, j0, c0, i0p, j0p), v * sharedC[c0])
        bi += x_gridsize


@cuda.jit
def op_numba_c_f(input, gamma, mu, c, weight_shape_1, weight_shape_2, weight_shape_3):
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

    if bi > input.shape[0] or i0 > input.shape[1] or j0 > input.shape[2]:
        return

    while bi < input.shape[0]:
        while i0 < input.shape[1]:
            while j0 < input.shape[2]:
                # Iterate on io j0 i0p j0p
                for i0p in range(input.shape[5]):
                    for j0p in range(input.shape[6]):
                        v = numba.float32(0.)
                        for ci in range(weight_shape_1):
                            for ii in range(weight_shape_2):
                                for ji in range(weight_shape_3):
                                    v += (mu[bi, ci, i0+ii, j0+ji] * mu[bi, ci, i0p+ii, j0p+ji] + gamma[bi, ci, i0+ii, j0+ji, ci, i0p+ii, j0p+ji])
                        # For each c0 update input
                        for c0 in range(input.shape[1]):
                            #input[bi, c0, i0, j0, c0, i0p, j0p] += v * sharedC[c0]
                            cuda.atomic.add(input, (bi, c0, i0, j0, c0, i0p, j0p), v * sharedC[c0])
                j0 += z_gridsize
            i0 += y_gridsize
        bi += x_gridsize


class Conv2DUF_op(Function):
    @staticmethod
    def forward(ctx, input, gamma, mu, c, weight_shape):
        return torch.from_numpy(op_numba(input.numpy(), gamma.numpy(), mu.numpy(), c.numpy(), weight_shape[1], weight_shape[2], weight_shape[3]))

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out


def next_power_of_2(x):
    return 1<<(x-1).bit_length()


def conv2duf_op(input, gamma, mu, c, weight_shape, stride: int=1):
    assert stride == 1, 'Stride != 1 not supported yet'
    if c.dim() == 0:
        c = c.repeat(input.shape[1])
    assert weight_shape[1] == mu.shape[1]
    if input.device.type == "cpu":
        input = Conv2DUF_op.apply(input, gamma, mu, c, weight_shape)
        return input
    else:
        # TODO fidling with threads
        if True or input.shape[2] * input.shape[1] < input.shape[0]:
            threadsperblock= (min(1024,next_power_of_2(input.shape[0])),)# 4)
            blockspergrid_x = math.ceil(input.shape[0] / threadsperblock[0])

            blockspergrid = (blockspergrid_x,)# blockspergrid_y)
            op_numba_c[blockspergrid, threadsperblock](input, gamma, mu, c, weight_shape[1], weight_shape[2], weight_shape[3])
        else:
            threadsperblock= (8,8,8)# 4)
            blockspergrid_x = math.ceil(input.shape[0] / threadsperblock[0])
            blockspergrid_y = math.ceil(input.shape[1] / threadsperblock[1])
            blockspergrid_z = math.ceil(input.shape[2] / threadsperblock[2])

            blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
            op_numba_c_f[blockspergrid, threadsperblock](input, gamma, mu, c, weight_shape[1], weight_shape[2], weight_shape[3])
        return input


if __name__ == '__main__':
    slow_compare = False
    wh = 4
    who = wh - 2
    ch = 2
    bs = 2

    input = torch.rand(bs, ch, who, who, ch, who, who)
    gamma = torch.rand(bs, ch, wh, wh, ch, wh, wh)
    mu = torch.rand(bs, ch, wh, wh)
    c = torch.tensor(1.)
    if c.dim() == 0:
        c = c.repeat(input.shape[1])
    weight_shape = [ch, ch, 3, 3]
    
    ref = conv2duf_op(input.clone(), gamma, mu, c, weight_shape)
    print(input.mean())
    print('REF MEAN IS')
    print(ref.mean())

    if slow_compare:
        slow = op_slow(input.clone(), gamma, mu, c, weight_shape)
        assert torch.allclose(slow, ref)
    print('WITH TIMING')
    print(timeit.timeit(lambda: conv2duf_op(input.clone(), gamma, mu, c, weight_shape), number=5))
    for device in [torch.device('cuda:0')]:
        print('#'*10)
        print(device)
        print('#'*10)
    
        n = input.clone()#.fill_(0)
        nb = conv2duf_op(n.to(device), gamma.to(device), mu.to(device), c.to(device), weight_shape)
        print('CANDIDATE MEAN IS')
        print(nb.mean())
        print(ref[0])
        print(nb[0])
        assert torch.allclose(ref, nb.to(ref))
        print('WITH TIMING')
        print(timeit.timeit(lambda: conv2duf_op(n.to(device), gamma.to(device), mu.to(device), c.to(device), weight_shape), number=5))

        # print('Here is taichi')
        # tai = conv2duf_taichi(input.clone(), gamma, mu, c, weight_shape)
        # print(tai.mean())
        # assert torch.allclose(ref, tai)
        # print(timeit.timeit(lambda: conv2duf_taichi(input.clone(), gamma, mu, c, weight_shape), number=5))

    device = torch.device('cuda:0')
    with profile(activities=[ProfilerActivity.CUDA], profile_memory=True) as prof:
        with record_function("op"):
            conv2duf_op(n.to(device), gamma.to(device), mu.to(device), c.to(device), weight_shape)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    prof.export_chrome_trace("trace.json")
