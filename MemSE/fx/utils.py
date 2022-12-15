import torch.nn as nn
from MemSE.definitions import SUPPORTED_OPS, UNSUPPORTED_OPS
from typing import Iterator

__all__ = ['net_param_iterator']


def net_param_iterator(model: nn.Module) -> Iterator:
    # TODO use torch fx to replace the graph elements and perform regular forward
    ignored = []
    for _, module in model.named_modules():
        if type(module) in SUPPORTED_OPS.keys() or hasattr(module, 'memse'):
            yield module
        elif type(module) in UNSUPPORTED_OPS:
            raise ValueError(f'The network is using an unsupported operation {type(module)}')
        else:
            #warnings.warn(f'The network is using an operation that is not supported or unsupported, ignoring it ({type(module)})')
            ignored.append(type(module))
    #print(set(ignored))