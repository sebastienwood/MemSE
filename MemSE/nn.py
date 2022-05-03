import torch

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
