import pytest
import numpy as np
import torch
import torch.nn as nn
import timeit

from MemSE.mse_functions import linear_layer_vec_batched
from MemSE.keops_mse_functions import k_linear_layer
from MemSE.network_manipulations import build_sequential_linear, record_shapes
from MemSE.nn import Conv2DUF

use_cuda = False

def test_new_method():
    def maybe_cuda_from_numpy(tensor):
        if type(tensor) is np.ndarray:
            tensor = torch.from_numpy(tensor)
        return tensor.cuda() if use_cuda else tensor
    G_1 = 16384
    MU_1 = 3468
    bs = 4
    mu = np.random.rand(bs, MU_1).astype(np.float32)
    gamma = np.random.rand(bs, MU_1, MU_1).astype(np.float32)
    G =  np.random.rand(G_1, MU_1).astype(np.float32)
    sigma = 0.1
    r = 1
    c = np.ones(G_1).astype(np.float32)

    mu_t = maybe_cuda_from_numpy(mu)
    ga_t = maybe_cuda_from_numpy(gamma)
    G_t = maybe_cuda_from_numpy(G)
    c_t = maybe_cuda_from_numpy(c)
    sigma_c = sigma / c_t

    ref = linear_layer_vec_batched(mu_t, ga_t, G_t, sigma_c, r)
    print(timeit.timeit('linear_layer_vec_batched(mu_t, ga_t, G_t, sigma_c, r)'))

    res = k_linear_layer(mu_t, ga_t, G_t, sigma_c, r)
    assert torch.allclose(ref, res)
    print(timeit.timeit('k_linear_layer(mu_t, ga_t, G_t, sigma_c, r)'))

def test_memristor_unfolded():
    inp = torch.rand(1,3,10,12)
    conv = nn.Conv2d(3,3,2)
    record_shapes(conv, inp.shape[1:])
    y = conv(inp)
    conv2duf = Conv2DUF(conv, inp.shape, conv.__output_shape)
    y_hat = conv2duf(inp)
    assert torch.allclose(y, y_hat)

def test_memristor_large():
    inp = torch.rand(1,3,10,12)
    conv = nn.Conv2d(3,3,2)
    y = record_shapes(conv, inp.shape)
    conv2duf = build_sequential_linear(conv)
    y_hat = conv2duf(inp)
    assert torch.allclose(y, y_hat)
