import copy
import torch
import torch.nn as nn
from MemSE.fx.Conv2d import build_sequential_unfolded, build_sequential_linear
from MemSE.fx.Linear import fuse_linear_bias
from MemSE.fx.trace import record_shapes
from MemSE.utils import count_parameters, listify
from MemSE.fx.conv_decompositions import tucker_decomposition_conv_layer

__all__ = ['conv_to_fc', 'conv_to_unfolded']

def NOOP(*args):
    return nn.Identity()

def ERROP(*args):
    raise ValueError('This layer should not appear during this operation')

CONVERT = {
    'linear': build_sequential_linear,
    'unfolded': build_sequential_unfolded
}

OPMAP = {
    nn.Linear: fuse_linear_bias,
    nn.Dropout: NOOP,
    nn.BatchNorm2d: ERROP
}


def recursive_setattr(obj, name, new):
    splitted = name.split('.', 1)
    if len(splitted) == 1:
        return setattr(obj, name, new)
    else:
        return recursive_setattr(getattr(obj, splitted[0]), splitted[1], new)


def replace_op(model: nn.Module, opmap: dict):
    # TODO maybe support args for transfo
    # TODO support pipeline of transfo
    new_modules = {}
    for name, module in model.named_modules():
        if type(module) in opmap:
            new_fc = opmap[type(module)](module)
            if not isinstance(new_fc, type(module)):  # suppose there's only one fx per final module
                new_fc = replace_op(new_fc, opmap)
            new_modules[name] = new_fc
    if len(list(model.modules())) == 1 and len(new_modules) == 1:
        return new_modules.popitem()[1]
    [recursive_setattr(model, n, fc) for n, fc in new_modules.items()]
    return model


@torch.no_grad()
def conv_to_memristor(model, input_shape, verbose=True, impl='linear', opmap: dict = OPMAP):
    assert not hasattr(model, '__memed'), 'This model has already been used in `conv_to_memristor`'
    op = CONVERT.get(impl)
    model = model.cpu()
    model = copy.deepcopy(model)
    model.eval()
    if verbose:
        print('Before conversion, model is:')
        print(model)

    if len(input_shape) == 4:
        input_shape = input_shape[1:]
    x = torch.rand(input_shape)
    x = x[None, :, :, :]

    y = record_shapes(model, x)

    model = replace_op(model, opmap | {nn.Conv2d: op})
    if verbose:
        print(f"==> converted Conv2d to {impl}")
        print(model)
    assert torch.allclose(y, model(x), atol=1e-5), f'{impl} transformation did not go well'
    model.train()
    model.__memed = True
    return model


def conv_to_fc(model, input_shape, verbose=False):
    return conv_to_memristor(model, input_shape, verbose, impl='linear')


def conv_to_unfolded(model, input_shape, verbose=False):
    return conv_to_memristor(model, input_shape, verbose, impl='unfolded')


def act_to_act(model, activation_type):
    assert isinstance(activation_type, nn.Module)
    ACTIVATIONS = [nn.ReLU, nn.Softplus, nn.GELU]
    ACTIVATIONS.pop(activation_type)
    replace_op(model, activation_type, ACTIVATIONS)


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
