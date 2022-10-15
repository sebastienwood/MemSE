import math
import torch
import torch.nn as nn
import opt_einsum as oe

from typing import Optional, List, Tuple, Union


__all__ = ['mse_gamma', 'diagonal_replace', 'zero_but_diag_', 'quant_but_diag_', 'zero_diag', 'LambdaLayer', 'InspectorLayer']


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)
    

class InspectorLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, x):
        print(x.shape)
        return x


def mse_gamma(tar, mu, gamma, verbose: bool = False):
    if len(tar.shape) != len(mu.shape):
        tar = tar.reshape_as(mu)
    vari = torch.diagonal(gamma, dim1=1, dim2=2)
    exp = torch.square(mu - tar)
    if verbose:
        res_v = vari.mean(dim=1).abs()
        res_e = exp.mean(dim=1).abs()
        tot = res_v + res_e
        print(f'VAR IMPORTANCE {res_v / tot}')
        print(f'EXP IMPORTANCE {res_e / tot}')
    return exp + vari


def diagonal_replace(tensor, diagonal, backprop:bool=False):
    """Backprop compatible diagonal replacement

    Args:
        tensor (_type_): _description_
        diagonal (_type_): a 1D `torch.Tensor` replacing `tensor` diagonal elements
        backprop (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    assert len(diagonal.shape) == 2
    assert len(tensor.shape) == 3
    assert tensor.shape[1] == tensor.shape[2]
    tensor[:, range(tensor.shape[1]), range(tensor.shape[2])] = diagonal
    # diag_ones = torch.ones(diagonal.shape[1:], device=tensor.device)
    # mask = torch.diag(diag_ones).unsqueeze_(0)
    # if backprop:
    #     mask *= .99
    # out = mask * torch.diag_embed(diagonal) + (1 - mask) * tensor
    return tensor


def zero_diag(tensor):
    if len(tensor.shape) > 2:
        original_shape = tensor.shape
        half = int(math.sqrt(tensor.numel()/tensor.shape[0]))
        tensor = tensor.view(tensor.shape[0], half, half)
    diag = torch.zeros(torch.diagonal(tensor, dim1=1, dim2=2).shape, device=tensor.device)
    res = diagonal_replace(tensor, diag)
    return res.view(original_shape)


def zero_but_diag_(tensor):
    diag = tensor.diagonal(dim1=1, dim2=2).data.clone()
    tensor.data.zero_()
    tensor.diagonal(dim1=1, dim2=2).data.copy_(diag)


def quant_but_diag_(tensor, quant_scheme):
    pass


#@torch.jit.script # see https://github.com/pytorch/pytorch/issues/49372
def padded_mu_gamma(mu, gamma: torch.Tensor, padding:Union[int, Tuple]=1, gamma_shape:Optional[List[int]]=None, square_reshape: bool = True):
    if isinstance(padding, int):
        padding = (padding, padding, padding, padding)
    if isinstance(padding, tuple) and len(padding) == 2:
        padding = (padding[1], padding[1], padding[0], padding[0])
    assert len(padding) == 4
    batch_len = mu.shape[0]
    #padded_size = [mu.shape[1], mu.shape[2]+padding*2, mu.shape[3]+padding*2]

    pad_mu = torch.nn.functional.pad(mu, padding)
    numel_image = pad_mu.shape[1:].numel()
    if square_reshape:
        pad_mu = torch.reshape(pad_mu, (batch_len,numel_image))

    if gamma_shape is not None:# gamma == 0 store only size
        pad_gamma = gamma
        if square_reshape:
            gamma_shape = [batch_len,numel_image,numel_image]
        else:
            gamma_shape = pad_mu.shape + pad_mu.shape[1:]
    else:
        pad_gamma = torch.nn.functional.pad(gamma, (padding + (0, 0) + padding))
        if square_reshape:
            pad_gamma = torch.reshape(pad_gamma, (batch_len,numel_image,numel_image))

    return pad_mu, pad_gamma, gamma_shape


#@torch.jit.script
def energy_vec_batched(c, G, gamma:torch.Tensor, mu, new_gamma_pos_diag:torch.Tensor, new_mu_pos, new_gamma_neg_diag:torch.Tensor, new_mu_neg, r:float, gamma_shape:Optional[List[int]]=None):
    if gamma_shape is not None:
        mu_r = oe.contract('i,ij,bj->b', c, torch.abs(G), mu)
    else:
        diag_gamma = torch.diagonal(gamma, dim1=1, dim2=2)
        mu_r = oe.contract('i,ij,bj->b', c, torch.abs(G), diag_gamma+mu)

    #diag_ngp = torch.diagonal(new_gamma_pos_diag, dim1=1, dim2=2)
    #diag_ngn = torch.diagonal(new_gamma_neg_diag, dim1=1, dim2=2)
    diags = (new_gamma_pos_diag + torch.square(new_mu_pos) + new_gamma_neg_diag + torch.square(new_mu_neg)) # (diag_ngp + torch.square(new_mu_pos) + diag_ngn + torch.square(new_mu_neg))
    return (mu_r + oe.contract('i,bi->b',torch.square(c), diags)) / r


@torch.jit.script
def double_conv(tensor: torch.Tensor,
                weight: torch.Tensor,
                stride: List[int] = (1, 1),
                padding: List[int] = (1, 1),
                dilation: List[int] = (1, 1),
                groups: int = 1,
                ):
    '''A doubly convolution for tensor of shape [bijkijk]'''
    if not weight.is_cuda:
        dtype = torch.float32
    else:
        dtype = torch.float16
    weight = weight.to(dtype=dtype, memory_format=torch.channels_last)
    tensor = tensor.to(dtype=dtype)

    # TODO not so sure it works for grouped convolutions
    bs = tensor.shape[0]
    img_shape = tensor.shape[1:4]

    nice_view = tensor.reshape(-1, img_shape[0], img_shape[1], img_shape[2]).contiguous(memory_format=torch.channels_last)
    first_res = torch.nn.functional.conv2d(input=nice_view, weight=weight, stride=stride, padding=padding, dilation=dilation, groups=groups)

    first_res_shape = first_res.shape
    nice_view_res = first_res.view(bs, img_shape[0], img_shape[1], img_shape[2], first_res_shape[1], first_res_shape[2], first_res_shape[3])

    permuted = nice_view_res.permute(0, 4, 5, 6, 1, 2, 3)
    another_nice_view = permuted.reshape(-1, img_shape[0], img_shape[1], img_shape[2]).contiguous(memory_format=torch.channels_last)
    second_res = torch.nn.functional.conv2d(input=another_nice_view, weight=weight, stride=stride, padding=padding, dilation=dilation, groups=groups)

    second_res_shape = second_res.shape
    anv_res = second_res.view(bs, first_res_shape[1], first_res_shape[2], first_res_shape[3], second_res_shape[1], second_res_shape[2], second_res_shape[3])

    return anv_res.permute(0, 4, 5, 6, 1, 2, 3).to(memory_format=torch.contiguous_format)


def gamma_to_diag(tensor, flatten=False):
    bs = tensor.shape[0]
    nc = tensor.shape[1]
    img_shape = tensor.shape[1:4]
    numel_image = img_shape.numel()
    view = torch.reshape(tensor, (bs,numel_image,numel_image))
    diag = torch.diagonal(view, dim1=1, dim2=2)
    return diag.reshape((bs,*img_shape)) if not flatten else diag


def gamma_add_diag(tensor, added):
    diag = gamma_to_diag(tensor)
    diag += added


def squeeze_group(nc, win_nc, groups):
    '''Given number of channels, size of weights input channel and nb of groups of unsqueezed conv, return number of groups for squeezed conv'''
    print(nc/win_nc)
    return int(nc / win_nc)


if __name__ == '__main__':
    from time import time
    import numpy as np
    device = torch.device('cuda:0')
    unscripted = double_conv
    inp = torch.rand(16, 3, 32, 32, 3, 32, 32, device=device)
    w = torch.rand(3, 3, 3, 3, device=device)
    for n, m in {'us': unscripted, 's': torch.jit.script(double_conv)}.items(): # 
        #for dt in [torch.float16, torch.float32]:
            #for memf in [torch.channels_last, torch.contiguous_format]:
        timings = []
        for _ in range(100):
            start = time()
            m(inp, w)
            timings.append(time() - start)
                
        median_time = np.median(timings)
        print(f'Median time is {median_time} ({n})')
