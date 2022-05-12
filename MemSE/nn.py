from numpy import dtype
import torch
import torch.nn as nn

def mse_gamma(tar, mu, gamma, verbose: bool = False):
    vari = torch.diagonal(gamma, dim1=1, dim2=2)
    exp = torch.square(mu - tar)
    if verbose:
        res_v = vari.mean(dim=1).abs()
        res_e = exp.mean(dim=1).abs()
        tot = res_v + res_e
        print(f'VAR IMPORTANCE {res_v / tot}')
        print(f'EXP IMPORTANCE {res_e / tot}')
    return exp + vari


def diagonal_replace(tensor, diagonal):
    '''Backprop compatible diagonal replacement
    '''
    mask = torch.diag(torch.ones(diagonal.shape[1:], device=tensor.device)).unsqueeze_(0)
    out = mask * torch.diag_embed(diagonal) + (1 - mask) * tensor
    return out


def zero_but_diag_(tensor):
    diag = tensor.diagonal(dim1=1, dim2=2).data.clone()
    tensor.data.zero_()
    tensor.diagonal(dim1=1, dim2=2).data.copy_(diag)


def quant_but_diag_(tensor, quant_scheme):
    pass


class Conv2DUF(nn.Module):
    def __init__(self, conv, input_shape, output_shape):
        super().__init__()
        self.c = conv
        self.output_shape = output_shape
        self.w = conv.weight.detach().clone().view(conv.weight.size(0), -1).t()
        #self.linear_equiv = nn.Linear(input_shape[1:].numel(),
        #                              output_shape[1:].numel(),
        #                              bias=True if conv.bias is not None else False,
        #                              device=conv.weight.device,
        #                              dtype=conv.weight.dtype)
        #self.linear_equiv.weight.data.copy_(w)
        #self.linear_equiv.bias.data.copy_(conv.bias)

    def forward(self, x):
        inp_unf = torch.nn.functional.unfold(x, self.c.kernel_size, self.c.dilation, self.c.padding, self.c.stride)
        out_unf = inp_unf.transpose(1, 2).matmul(self.w).transpose(1, 2)
        out = out_unf.view(*self.output_shape)
        return out
