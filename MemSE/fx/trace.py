import torch

__all__ = ['record_shapes', 'get_intermediates', 'store_add_intermediates_mse', 'store_add_intermediates_var']

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


def get_intermediates(model, input):
    hooks = {}
    def hook_fn(self, input, output):
        self.__original_output = output.clone().detach().cpu()
    for name, module in model.named_modules():
        hooks[name] = module.register_forward_hook(hook_fn)
    _ = model(input)
    [h.remove() for h in hooks.values()]


def store_add_intermediates_mse(model, reps):
    hooks = {}
    @torch.no_grad()
    def hook_fn(self, input, output):
        se = (output.clone().detach().cpu() - self.__original_output) ** 2 / reps
        if not hasattr(self, '__se_output'):
            self.__se_output = se
            self.__th_output = output.clone().detach().cpu() / reps
        else:
            self.__se_output += se
            self.__th_output += output.clone().detach().cpu() / reps
    for name, module in model.named_modules():
        hooks[name] = module.register_forward_hook(hook_fn)
    return hooks


def store_add_intermediates_var(model, reps, compute_cov: bool = False):
    # TODO unbiased var estimates suggest we should have (reps - 1), it was debated 
    hooks = {}
    @torch.no_grad()
    def hook_fn(self, input, output):
        base = (output.clone().detach().cpu() - self.__th_output)
        if compute_cov:
            flattened = base.reshape(base.shape[0], -1)
            covs = torch.einsum('bi, bj -> bij', flattened, flattened).view(base.shape + base.shape[1:]) / reps
        if not hasattr(self, '__var_output'):
            self.__var_output = base ** 2 / reps
            if compute_cov:
                self.__cov_output = covs
        else:
            self.__var_output += base ** 2 / reps
            if compute_cov:
                self.__cov_output += covs
    for name, module in model.named_modules():
        hooks[name] = module.register_forward_hook(hook_fn)
    return hooks