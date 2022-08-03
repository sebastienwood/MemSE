import os
import timeit
from typing import List

import math
import numba
import torch
import taichi as ti
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function
from torch.utils.cpp_extension import load
from numba import njit, prange, cuda
from stannum import Tube

TUBE = None

module_path = os.path.dirname(__file__)
#conv2duf_cuda = load(
#    "conv2duf",
#    sources=[
#        os.path.join(module_path, "conv2duf_op.cpp"),
#        os.path.join(module_path, "conv2duf_op_kernel.cu"),
#    ],
#)


# class FusedLeakyReLUFunctionBackward(Function):
#     @staticmethod
#     def forward(ctx, grad_output, out, bias, negative_slope, scale):
#         ctx.save_for_backward(out)
#         ctx.negative_slope = negative_slope
#         ctx.scale = scale

#         empty = grad_output.new_empty(0)

#         grad_input = fused.fused_bias_act(
#             grad_output.contiguous(), empty, out, 3, 1, negative_slope, scale
#         )

#         dim = [0]

#         if grad_input.ndim > 2:
#             dim += list(range(2, grad_input.ndim))

#         if bias:
#             grad_bias = grad_input.sum(dim).detach()

#         else:
#             grad_bias = empty

#         return grad_input, grad_bias

#     @staticmethod
#     def backward(ctx, gradgrad_input, gradgrad_bias):
#         out, = ctx.saved_tensors
#         gradgrad_out = fused.fused_bias_act(
#             gradgrad_input.contiguous(),
#             gradgrad_bias,
#             out,
#             3,
#             1,
#             ctx.negative_slope,
#             ctx.scale,
#         )

#         return gradgrad_out, None, None, None, None


# class FusedLeakyReLUFunction(Function):
#     @staticmethod
#     def forward(ctx, input, bias, negative_slope, scale):
#         empty = input.new_empty(0)

#         ctx.bias = bias is not None

#         if bias is None:
#             bias = empty

#         out = fused.fused_bias_act(input, bias, empty, 3, 0, negative_slope, scale)
#         ctx.save_for_backward(out)
#         ctx.negative_slope = negative_slope
#         ctx.scale = scale

#         return out

#     @staticmethod
#     def backward(ctx, grad_output):
#         out, = ctx.saved_tensors

#         grad_input, grad_bias = FusedLeakyReLUFunctionBackward.apply(
#             grad_output, out, ctx.bias, ctx.negative_slope, ctx.scale
#         )

#         if not ctx.bias:
#             grad_bias = None

#         return grad_input, grad_bias, None, None


# class FusedLeakyReLU(nn.Module):
#     def __init__(self, channel, bias=True, negative_slope=0.2, scale=2 ** 0.5):
#         super().__init__()

#         if bias:
#             self.bias = nn.Parameter(torch.zeros(channel))

#         else:
#             self.bias = None

#         self.negative_slope = negative_slope
#         self.scale = scale

#     def forward(self, input):
#         return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)


# @triton.autotune(
#     configs=[
#         triton.Config({'BLOCK_SIZE': 128}),
#         triton.Config({'BLOCK_SIZE': 256})
#     ],
#     key=['B', 'C']
# )
# @triton.jit
# def conv2duf_kernel(
#     input_ptr, gamma_ptr,
#     # Tensor dimensions
#     B, C, I, J, D, K, L,
#     # Stride
#     stride_ib, stride_ic, stride_ii, stride_ij, stride_ik, stride_il, stride_im,
#     stride_gb, stride_gc, stride_gi, stride_gj, stride_gk, stride_gl, stride_gm,
#     # Meta parameters
#     BLOCK_SIZE: tl.constexpr  # BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
#     #GROUP_SIZE_M: tl.constexpr
# ):
#     """Kernel for computing sliding sum on 7D gamma tensor.
#     """
#     # ------------
#     # Map program ids pid to the block of input it should compute
#     pid = tl.program_id(axis=0)
#     channel = tl.program_id(axis=1)
#     off_in = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
#     print(off_in)
#     input_ptrs = input_ptr + (off_in[:, None] * stride_ib)
#     print(input_ptrs)
#     inp = tl.load(input_ptrs, mask=input_ptrs< B, other=0)
#     print(inp)
#     # ------------
#     # Create pointers for data
    

#     # Need to load gamma patch
    

#     # Need to load input item

#     # Need to store input updated
    


# class Conv2DUFWeird(Function):
#     @staticmethod
#     def forward(ctx, input, gamma, weight_shape, BLOCK_SIZE=256):
#         input = input.contiguous()
#         gamma = gamma.contiguous()
#         assert len(input.shape) == 7 and len(gamma.shape) == 7
#         assert input.is_contiguous()
#         assert gamma.is_contiguous()
#         M = input.shape[0]
#         grid = lambda META: (
#             triton.cdiv(M, META['BLOCK_SIZE'])
#         )
#         conv2duf_kernel[grid](
#             input,
#             gamma,
#             input.shape[0], input.shape[1], input.shape[2], input.shape[3], input.shape[4], input.shape[5], input.shape[6],
#             input.stride(0), input.stride(1), input.stride(2), input.stride(3), input.stride(4), input.stride(5), input.stride(6),
#             gamma.stride(0), gamma.stride(1), gamma.stride(2), gamma.stride(3), gamma.stride(4), gamma.stride(5), gamma.stride(6)
#             BLOCK_SIZE=BLOCK_SIZE
#         )
#         return input

#     def backward(ctx, grad_output):
#         pass

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


@njit(parallel=True, nogil=True)
def op_numba(input, gamma, mu, c, weight_shape_1, weight_shape_2, weight_shape_3):
    for bi in prange(input.shape[0]):
        for c0 in prange(input.shape[1]):
            for i0 in range(input.shape[2]):
                for j0 in range(input.shape[3]):
                    for i0p in range(input.shape[5]):
                        for j0p in range(input.shape[6]):
                            v = 0.
                            for ci in range(weight_shape_1):
                                for ii in range(weight_shape_2):
                                    for ji in range(weight_shape_3):
                                        v += (mu[bi, ci, i0+ii, j0+ji] * mu[bi, ci, i0p+ii, j0p+ji] + gamma[bi, ci, i0+ii, j0+ji, ci, i0p+ii, j0p+ji])
                            input[bi, c0, i0, j0, c0, i0p, j0p] += c[c0] * v
    return input


@cuda.jit
def op_numba_c(input, gamma, mu, c, weight_shape_1, weight_shape_2, weight_shape_3):
    sharedI = cuda.shared.array((0, 0, 0, 0, 0, 0, 0), dtype=numba.float32)
    sharedM = cuda.shared.array((0, 0, 0, 0), dtype=numba.float32)
    sharedG = cuda.shared.array((0, 0, 0, 0, 0, 0, 0), dtype=numba.float32)
    sharedC = cuda.shared.array(0, dtype=numba.float32)
    bi, c0 = cuda.grid(2)
    threadB, threadC = cuda.threadIdx.x, cuda.threadIdx.y
    x_gridsize = cuda.gridDim.x * cuda.blockDim.x
    y_gridsize = cuda.gridDim.y * cuda.blockDim.y

    while bi < input.shape[0]:
        while c0 < input.shape[1]:
            if bi < input.shape[0] and c0 < input.shape[1]:
                sharedI[threadB, threadC, :, :, 0] = input[bi, c0, :, :, c0]
                sharedM[threadB] = mu[bi]
                sharedG[threadB] = gamma[bi]
                sharedC[threadC] = c[c0]
            cuda.syncthreads()
            for i0 in range(input.shape[2]):
                for j0 in range(input.shape[3]):
                    for i0p in range(input.shape[5]):
                        for j0p in range(input.shape[6]):
                            v = numba.float32(0.)
                            for ci in range(weight_shape_1):
                                for ii in range(weight_shape_2):
                                    for ji in range(weight_shape_3):
                                        v += (sharedM[threadB, ci, i0+ii, j0+ji] * sharedM[threadB, ci, i0p+ii, j0p+ji] + sharedG[threadB, ci, i0+ii, j0+ji, ci, i0p+ii, j0p+ji])
                            sharedI[threadB, threadC, i0, j0, 0, i0p, j0p] += sharedC[threadC] * v
            
            cuda.syncthreads()
            if bi < input.shape[0] and c0 < input.shape[1]:
                input[bi, c0, :, :, c0, :, :] = sharedI[bi, c0, :, :, 0, :, :]

            c0 += y_gridsize
        bi += x_gridsize


@ti.kernel
def op_taichi(gamma: ti.template(), mu: ti.template(), c: ti.template(), input: ti.template(), weight_shape_1: int, weight_shape_2: int, weight_shape_3:int):
    ti.block_local(c, mu, gamma)
    for bi in range(input.shape[0]):
        for c0 in range(input.shape[1]):
            for i0 in range(input.shape[2]):
                for j0 in range(input.shape[3]):
                    for i0p in range(input.shape[5]):
                        for j0p in range(input.shape[6]):
                            v = 0.
                            for ci in ti.static(range(weight_shape_1)):
                                for ii in ti.static(range(weight_shape_2)):
                                    for ji in ti.static(range(weight_shape_3)):
                                        v += (mu[bi, ci, i0+ii, j0+ji] * mu[bi, ci, i0p+ii, j0p+ji] + gamma[bi, ci, i0+ii, j0+ji, ci, i0p+ii, j0p+ji])
                            input[bi, c0, i0, j0, c0, i0p, j0p] += c[c0] * v
    return input


def conv2duf_taichi(input, gamma, mu, c, weight_shape):
    if c.dim() == 0:
        c = c.repeat(input.shape[1])
    global TUBE
    if TUBE is None:
        device = input.device # TODO dim alignment with -2, ...
        b = input.shape[0]
        tube = Tube(device) \
            .register_input_tensor((-1,)*7, input.dtype, "gamma", True) \
            .register_input_tensor((-1,)*4, input.dtype, "mu", True) \
            .register_input_tensor((-1,), input.dtype, "c", True) \
            .register_output_tensor((-1,)*7, input.dtype, "input", True) \
            .register_kernel(op_taichi, ["gamma", "mu", "c", "input"]) \
            .finish()
        TUBE = tube
    return TUBE(gamma, mu, c, input, weight_shape[1], weight_shape[2], weight_shape[3])


class Conv2DUF_op(Function):
    @staticmethod
    def forward(ctx, input, gamma, mu, c, weight_shape):
        return torch.from_numpy(op_numba(input.numpy(), gamma.numpy(), mu.numpy(), c.numpy(), weight_shape[1], weight_shape[2], weight_shape[3]))

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out


def conv2duf_op(input, gamma, mu, c, weight_shape):
    if c.dim() == 0:
        c = c.repeat(input.shape[1])
    if input.device.type == "cpu":
        input = Conv2DUF_op.apply(input, gamma, mu, c, weight_shape)
        return input
    else:
        # TODO fidling with threads
        threadsperblock= (16, 16)
        blockspergrid_x = math.ceil(input.shape[0] / threadsperblock[0])
        blockspergrid_y = math.ceil(input.shape[1] / threadsperblock[1])

        blockspergrid = (blockspergrid_x, blockspergrid_y)
        op_numba_c[blockspergrid, threadsperblock](input, gamma, mu, c, weight_shape[1], weight_shape[2], weight_shape[3])
        return input


if __name__ == '__main__':
    device = torch.device('cuda:0')
    input = torch.rand(2, 3, 7, 7, 3, 7, 7).to(device)
    gamma = torch.rand(2, 3, 9, 9, 3, 9, 9).to(device)
    mu = torch.rand(2, 3, 9, 9).to(device)
    c = torch.tensor(1.).to(device)
    weight_shape = [3, 3, 3, 3]
    ref = op_slow(input.clone(), gamma, mu, c, weight_shape)
    print(input.mean())
    print(ref.mean())
    #print(timeit.timeit(lambda: op_slow(input.clone(), gamma, mu, c, weight_shape), number=5), ' s')

    print('Here is numba')
    nb = conv2duf_op(input.clone(), gamma, mu, c, weight_shape)
    print(nb.mean())
    assert torch.allclose(ref, nb)
    print(timeit.timeit(lambda: conv2duf_op(input.clone(), gamma, mu, c, weight_shape), number=5))

    # print('Here is taichi')
    # tai = conv2duf_taichi(input.clone(), gamma, mu, c, weight_shape)
    # print(tai.mean())
    # assert torch.allclose(ref, tai)
    # print(timeit.timeit(lambda: conv2duf_taichi(input.clone(), gamma, mu, c, weight_shape), number=5))
