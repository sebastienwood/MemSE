import torch
import torch.nn as nn
from MemSE.utils import count_parameters
from MemSE.conv_decompositions import tucker_decomposition_conv_layer

#from https://github.com/pytorch/pytorch/issues/26781#issuecomment-821054668
def convmatrix2d(kernel, image_shape, padding=None):
    # kernel: (out_channels, in_channels, kernel_height, kernel_width, ...)
    # image: (in_channels, image_height, image_width, ...)

    if padding:
        assert padding[0] == padding[1]
        padding = padding[0]
        old_shape = image_shape
        pads = (padding, padding, padding, padding)
        image_shape = (image_shape[0], image_shape[1] + padding*2, image_shape[2]
                       + padding*2)
    else:
        image_shape = tuple(image_shape)

    assert image_shape[0] == kernel.shape[1]
    assert len(image_shape[1:]) == len(kernel.shape[2:])
    result_dims = torch.tensor(image_shape[1:]) - torch.tensor(kernel.shape[2:]) + 1
    m = torch.zeros((
        kernel.shape[0], 
        *result_dims, 
        *image_shape
    ))
    for i in range(m.shape[1]):
        for j in range(m.shape[2]):
            m[:,i,j,:,i:i+kernel.shape[2],j:j+kernel.shape[3]] = kernel
    return m.flatten(0, len(kernel.shape[2:])).flatten(1)

    # Handle zero padding. Effectively, the zeros from padding do not
    # contribute to convolution output as the product at those elements is zero.
    # Hence the columns of the conv mat that are at the indices where the
    # padded flattened image would have zeros can be ignored. The number of
    # rows on the other hand must not be altered (with padding the output must
    # be larger than without). So..

    # We'll handle this the easy way and create a mask that accomplishes the
    # indexing
    if padding:
        mask = torch.nn.functional.pad(torch.ones(old_shape), pads).flatten()
        mask = mask.bool()
        m = m[:, mask]

    return m

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)


@torch.no_grad()
def build_sequential_linear(conv):
    current_input_shape = conv.__input_shape
    current_output_shape = conv.__output_shape
    rand_x = torch.rand(current_input_shape)
    rand_y = conv(rand_x)
    conv_fced = convmatrix2d(conv.weight, current_input_shape[1:], conv.padding)
    linear = nn.Linear(conv_fced.shape[1], conv_fced.shape[0], bias=False)
    linear.weight.data = conv_fced
    seq = nn.Sequential(
        LambdaLayer(lambda x: nn.functional.pad(x, (conv.padding[1], conv.padding[1], conv.padding[0], conv.padding[0]))),  # is padding conv dependent ? i.e. should we grab params ?
        nn.Flatten(),
        linear,
        LambdaLayer(lambda x: torch.reshape(x, (-1,) + current_output_shape[1:]))
    )
    rand_y_repl = seq(rand_x)
    assert torch.allclose(rand_y, rand_y_repl, atol=1e-6), 'Linear did not cast to a satisfying solution'
    return seq

def recursive_setattr(obj, name, new):
    splitted = name.split('.', 1)
    if len(splitted) == 1:
        return setattr(obj, name, new)
    else:
        return recursive_setattr(getattr(obj, splitted[0]), splitted[1], new)


def replace_op(model, fx):
    new_modules = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            new_fc = fx(module)
            new_modules[name] = new_fc
    [recursive_setattr(model, n, fc) for n, fc in new_modules.items()]


@torch.no_grad()
def conv_to_fc(model, input_shape, verbose=False):
    model = model.cpu()
    if verbose:
        print(model)

    x = torch.rand(input_shape)
    x = x[None, :, :, :]

    def hook_fn(self, input, output):
        self.__input_shape = input[0].shape
        self.__output_shape = output.size()
    hooks = {}
    for name, module in model.named_modules():
        hooks[name] = module.register_forward_hook(hook_fn)
    y = model(x)
    [h.remove() for h in hooks.values()]
    
    replace_op(model, build_sequential_linear)
    if verbose:
        print("==> converted Conv2d to Linear")
        print(model)
    assert torch.allclose(y, model(x)), 'Linear transformation did not go well'
    return model


def conv_decomposition(model, verbose=False):
    model = model.cpu()
    if verbose:
        print(count_parameters(model))
    replace_op(model, tucker_decomposition_conv_layer)
    if verbose:
        print("==> converted Conv2d to Linear")
        print(model)
        print(count_parameters(model))
    return model


def get_intermediates(model, input):
    hooks = {}
    def hook_fn(self, input, output):
        self.__original_output = output.clone().detach().cpu()
    for name, module in model.named_modules():
        hooks[name] = module.register_forward_hook(hook_fn)
    _ = model(input)
    [h.remove() for h in hooks.values()]


def fuse_conv_bn(model):
    #TODO https://github.com/pytorch/pytorch/blob/40cbf342d3c000712da92cfafeaca651b3e0bd3e/torch/fx/experimental/optimization.py#L50
    # when we get to model with bn
    pass