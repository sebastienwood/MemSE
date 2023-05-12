import torch
from typing import Tuple


@torch.jit.script
def double_conv(tensor: torch.Tensor,
                weight: torch.Tensor,
                stride: Tuple[int, int] = (1, 1),
                padding: Tuple[int, int] = (1, 1),
                dilation: Tuple[int, int] = (1, 1),
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
