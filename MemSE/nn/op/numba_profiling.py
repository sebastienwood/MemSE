import torch
from numba import cuda
from MemSE.nn.op import conv2duf_op

device = torch.device('cuda:0')
wh = 32
who = wh - 2
ch = 3
bs = 128

if __name__ == '__main__':
    input = torch.rand(bs, ch, who, who, ch, who, who).to(device)
    gamma = torch.rand(bs, ch, wh, wh, ch, wh, wh).to(device)
    mu = torch.rand(bs, ch, wh, wh).to(device)
    c = torch.tensor(1.).to(device)
    if c.dim() == 0:
        c = c.repeat(input.shape[1])
    weight_shape = [ch, ch, 3, 3]
    with cuda.profiling():
        conv2duf_op(input, gamma, mu, c, weight_shape)
