from MemSE.nn import Conv2DUF


import torch.nn as nn


from typing import Tuple


def n_vars_computation(model: nn.Module) -> Tuple[int, int]:
    n_vars_column, n_vars_layer = 0, 0
    for _, module in model.named_modules():
        if isinstance(module, nn.Linear) or isinstance(module, Conv2DUF):
            n_vars_column += module.out_features  # [0]
            n_vars_layer += 1
    return n_vars_column, n_vars_layer