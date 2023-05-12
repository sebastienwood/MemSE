import torch
import timeit
from .conv2duf_op import conv2duf_op

def op_slow(input: torch.Tensor, gamma: torch.Tensor, mu: torch.Tensor, c: torch.Tensor, weight_shape: list) -> torch.Tensor:
    for bi in range(input.shape[0]):
        for i0 in range(input.shape[2]):
            for j0 in range(input.shape[3]):
                for i0p in range(input.shape[5]):
                    for j0p in range(input.shape[6]):
                        v = 0.
                        for ci in range(weight_shape[1]):
                            for ii in range(weight_shape[2]):
                                for ji in range(weight_shape[3]):
                                    v += (mu[bi, ci, i0+ii, j0+ji] * mu[bi, ci, i0p+ii, j0p+ji] + gamma[bi, ci, i0+ii, j0+ji, ci, i0p+ii, j0p+ji])
                        for c0 in range(input.shape[1]):
                            input[bi, c0, i0, j0, c0, i0p, j0p] += c[c0] * v
    return input


def op_challenger_kamran(input: torch.Tensor, gamma: torch.Tensor, mu: torch.Tensor, c: torch.Tensor, weight_shape: list) -> torch.Tensor:
    who = input.shape[2]
    assert input.shape[2] == input.shape[3]
    assert input.shape[3] == input.shape[5]
    assert input.shape[5] == input.shape[6]
    num_ch = weight_shape[1]
    k = weight_shape[2]
    assert weight_shape[2] == weight_shape[3]

    for bi in range(input.shape[0]):
        for i0 in range(who):
            for j0 in range(who):
                for i0p in range(who):
                    for j0p in range(who):
                        mu1 = mu[bi, :, i0:i0+k, j0:j0+k]
                        mu2 = mu[bi, :, i0p:i0p+k, j0p:j0p+k]
                        gamma1 = gamma[bi, torch.arange(num_ch), i0:i0+k, j0:j0+k,
                                       torch.arange(num_ch), i0p:i0p+k, j0p:j0p+k]
                        gamma1 = gamma1.reshape(num_ch, k*k, k*k)
                        v = gamma1[:, torch.arange(k*k), torch.arange(k*k)].sum(dim=-1)
                        v += (mu1*mu2).sum(dim=[1, 2])
                        v = v.sum()
                        input[bi, torch.arange(num_ch), i0, j0, torch.arange(num_ch), i0p, j0p] += c * v
    return input



if __name__ == '__main__':
    device = torch.device('cpu')
    wh = 32
    who = wh - 2
    ch = 3
    bs = 16

    input = torch.rand(bs, ch, who, who, ch, who, who).to(device)
    gamma = torch.rand(bs, ch, wh, wh, ch, wh, wh).to(device)
    mu = torch.rand(bs, ch, wh, wh).to(device)
    c = torch.tensor(1.).to(device)
    if c.dim() == 0:
        c = c.repeat(input.shape[1])
    weight_shape = [ch, ch, 3, 3]
    slow = conv2duf_op(input.clone(), gamma, mu, c, weight_shape)
    print('Ref done')

    ###
    # Challenger
    ###
    challenger = op_challenger(input.clone(), gamma, mu, c, weight_shape)
    assert torch.allclose(challenger, slow)
    t_n = timeit.timeit(lambda: conv2duf_op(input.clone(), gamma, mu, c, weight_shape), number=5)
    t_c = timeit.timeit(lambda: op_challenger(input.clone(), gamma, mu, c, weight_shape), number=5)
    print(f'Challenger took {t_c} secs')
    print(f'Numba optimized took {t_n} secs')
    print(f'C_s / N_s = {t_c/t_n}')
