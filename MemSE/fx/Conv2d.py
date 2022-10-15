import torch
import torch.nn as nn

from MemSE.nn import Conv2DUF, Flattener, Padder, Reshaper

__all__ = ['build_sequential_linear', 'build_sequential_unfolded']

#adapted from https://github.com/pytorch/pytorch/issues/26781#issuecomment-821054668
def convmatrix2d(kernel, image_shape, padding, stride):
    # kernel: (out_channels, in_channels, kernel_height, kernel_width, ...)
    # image: (in_channels, image_height, image_width, ...)

    assert padding[0] == padding[1]
    assert stride[0] == stride[1]
    
    padding = padding[0]
    old_shape = image_shape
    pads = (padding, padding, padding, padding)
    image_shape = (image_shape[0], image_shape[1] + padding*2, image_shape[2]
                   + padding*2)

    assert image_shape[0] == kernel.shape[1]
    assert len(image_shape[1:]) == len(kernel.shape[2:])
    
    result_dims = torch.tensor(image_shape[1:]) - torch.tensor(kernel.shape[2:]) + 1
    m = torch.zeros((
        kernel.shape[0],
        *result_dims,
        *image_shape
    ))

    for i in range(0, m.shape[1], stride[0]):
        for j in range(0, m.shape[2], stride[0]):
            m[:,i,j,:,i:i+kernel.shape[2],j:j+kernel.shape[3]] = kernel
    
    if stride[0] > 1:
        result_dims_stride = ((torch.tensor(image_shape[1:]) - torch.tensor(kernel.shape[2:]))//stride[0]) + 1
        m_stride = torch.zeros((
            kernel.shape[0],
            *result_dims_stride,
            *image_shape
        ))
        i_tmp = 0
        for i in range(0, result_dims[0]):
            j_tmp = 0
            if i % stride[0] != 0:
                continue
            for j in range(0, result_dims[1]):
                if j % stride[0] == 0:
                    m_stride[:,i_tmp,j_tmp,:,:,:] = m[:,i,j,:,:,:]
                    j_tmp += 1
            i_tmp += 1
 
        return m_stride.flatten(0, len(kernel.shape[2:])).flatten(1)

    return m.flatten(0, len(kernel.shape[2:])).flatten(1)


@torch.no_grad()
def build_sequential_linear(conv: nn.Conv2d):
    assert isinstance(conv, nn.Conv2d)
    current_input_shape = conv.__input_shape
    current_output_shape = conv.__output_shape
    rand_x = torch.rand(current_input_shape)
    rand_y = conv(rand_x)
    conv_fced = convmatrix2d(conv.weight, current_input_shape[1:], conv.padding, conv.stride)
    linear = nn.Linear(conv_fced.shape[1], conv_fced.shape[0], bias=conv.bias is not None)
    linear.weight.data = conv_fced
    if conv.bias is not None:
        linear.bias.data = conv.bias.repeat_interleave((linear.weight.shape[0]//conv.bias.shape[0]))
    seq = nn.Sequential(
        Padder((conv.padding[1], conv.padding[1], conv.padding[0], conv.padding[0])),
        Flattener(),
        linear,
        Reshaper(current_output_shape[1:]),
    )
    rand_y_repl = seq(rand_x)
    assert torch.allclose(rand_y, rand_y_repl, atol=1e-5), f'Linear did not cast to a satisfying solution ({torch.mean((rand_y - rand_y_repl)**2)})'
    return seq

@torch.no_grad()
def build_sequential_unfolded(conv: nn.Conv2d):
    assert isinstance(conv, nn.Conv2d)
    current_input_shape = conv.__input_shape
    current_output_shape = conv.__output_shape
    if len(current_output_shape) == 4:
        current_output_shape = current_output_shape[1:]
    return Conv2DUF(conv, current_input_shape, current_output_shape)