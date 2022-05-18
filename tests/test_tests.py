import numpy as np
import torch
import timeit
from MemSE.nn.Linear import linear_layer_vec_batched, k_linear_layer

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