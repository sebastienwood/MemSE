import torch
import torch.nn as nn
from MemSE.nn import Conv2DUF
from MemSE.utils import count_parameters
from MemSE.conv_decompositions import tucker_decomposition_conv_layer

def resnet_layer_fusion_generator(layer_idx, downsample: bool):
    lys = []
    b = f'layer{layer_idx}.'
    for i in range(2):
        for j in range(1, 3):
            lys.append([f'{b}{i}.conv{j}', f'{b}{i}.bn{j}'])
        if downsample and i == 0:
            lys.append([f'{b}{i}.downsample.0', f'{b}{i}.downsample.1'])
    return lys

MODELS_FUSION = {
    'resnet18': [
                  ['conv1', 'bn1'],
                  *resnet_layer_fusion_generator(1, False), # TODO c'est pas joli joli mais c'est plutôt flexible
                  *resnet_layer_fusion_generator(2, True),
                  *resnet_layer_fusion_generator(3, True),
                  *resnet_layer_fusion_generator(4, True)
                ],
    'smallest_vgg': None
}


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
    print(conv_fced.shape)
    linear = nn.Linear(conv_fced.shape[1], conv_fced.shape[0], bias=False)
    linear.weight.data = conv_fced
    seq = nn.Sequential(
        LambdaLayer(lambda x: nn.functional.pad(x, (conv.padding[1], conv.padding[1], conv.padding[0], conv.padding[0]))),  # is padding conv dependent ? i.e. should we grab params ?
        nn.Flatten(),
        linear,
        LambdaLayer(lambda x: torch.reshape(x, (-1,) + current_output_shape[1:]))
    )
    rand_y_repl = seq(rand_x)
    print(rand_y.shape)
    print('*'*15)
    print(rand_y_repl.shape)
    assert torch.allclose(rand_y, rand_y_repl, atol=1e-5), 'Linear did not cast to a satisfying solution'
    return seq

@torch.no_grad()
def build_sequential_unfolded_linear(conv):
    current_input_shape = conv.__input_shape
    current_output_shape = conv.__output_shape
    return Conv2DUF(conv, current_input_shape, current_output_shape)

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
def record_shapes(model, x):
    def hook_fn(self, input, output):
        self.__input_shape = input[0].shape
        self.__output_shape = output.size()
    hooks = {}
    for name, module in model.named_modules():
        hooks[name] = module.register_forward_hook(hook_fn)
    y = model(x)
    [h.remove() for h in hooks.values()]
    return y


@torch.no_grad()
def conv_to_memristor(model, input_shape, verbose=False, impl='linear'):
    assert impl in ['linear', 'unfolded']
    if impl == 'linear':
        op = build_sequential_linear
    else:
        op = build_sequential_unfolded_linear
    model = model.cpu()
    model.eval()
    if verbose:
        print(model)

    x = torch.rand(input_shape)
    x = x[None, :, :, :]

    y = record_shapes(model, x)
    
    replace_op(model, op)
    if verbose:
        print("==> converted Conv2d to Linear")
        print(model)
    assert torch.allclose(y, model(x), atol=1e-5), 'Linear transformation did not go well'
    model.train()
    return model


def conv_to_fc(model, input_shape, verbose=False):
    return conv_to_memristor(model, input_shape, verbose, impl='linear')


def conv_to_unfolded(model, input_shape, verbose=False):
    return conv_to_memristor(model, input_shape, verbose, impl='unfolded')


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


def store_add_intermediates_se(model):
    hooks = {}
    @torch.no_grad()
    def hook_fn(self, input, output):
        se = (output.clone().detach().cpu() - self.__original_output) ** 2
        if not hasattr(self, '__se_output'):
            self.__se_output = se
            self.__th_output = output.clone().detach().cpu()
        else:
            self.__se_output += se
            self.__th_output += output.clone().detach().cpu()
    for name, module in model.named_modules():
        hooks[name] = module.register_forward_hook(hook_fn)
    return hooks


def store_add_intermediates_var(model, reps):
    hooks = {}
    @torch.no_grad()
    def hook_fn(self, input, output):
        if not hasattr(self, '__var_output'):
            self.__var_output = (output.clone().detach().cpu() - self.__th_output / reps) ** 2
        else:
            self.__var_output += (output.clone().detach().cpu() - self.__th_output / reps) ** 2
    for name, module in model.named_modules():
        hooks[name] = module.register_forward_hook(hook_fn)
    return hooks


def fuse_conv_bn(model, model_name: str, model_fusion=MODELS_FUSION):
    #TODO https://github.com/pytorch/pytorch/blob/40cbf342d3c000712da92cfafeaca651b3e0bd3e/torch/fx/experimental/optimization.py#L50
    # when we get to model with bn
    #MODS = [torch.nn.Conv2d, torch.nn.BatchNorm2d]
    #modules_to_fuse = []
    #modules = list(model.named_modules())
    #for (f_n, f_m), (s_n, s_m) in zip(modules[1:], modules[:-1]):
    #    if type(f_m) in MODS and type(s_m) in MODS:
    #        modules_to_fuse.append([f_n, s_n])
        # TODO should skip next iter if found, shouldn't happen with classical topologies

    #print(modules_to_fuse)
    modules_to_fuse = MODELS_FUSION[model_name.lower()]
    if modules_to_fuse is not None:
        return torch.quantization.fuse_modules(model, modules_to_fuse)
    return model
