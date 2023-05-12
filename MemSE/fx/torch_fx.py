from torch.fx.node import Argument, Target
from torch.nn.utils.fusion import fuse_conv_bn_eval
from typing import Any, Dict, List, Tuple, Iterable, Type

import copy
import torch
import torch.fx as fx
import torch.nn as nn


__all__ = ['cast_to_memse', 'fuse', 'remove_dropout']


def _parent_name(target: str) -> Tuple[str, str]:
    """
    Splits a qualname into parent path and last atom.
    For example, `foo.bar.baz` -> (`foo.bar`, `baz`)
    """
    *parent, name = target.rsplit('.', 1)
    return parent[0] if parent else '', name


def matches_module_pattern(pattern: Iterable[Type], node: fx.Node, modules: Dict[str, Any]):
    if len(node.args) == 0:
        return False
    nodes: Tuple[Any, fx.Node] = (node.args[0], node)
    for expected_type, current_node in zip(pattern, nodes):
        if not isinstance(current_node, fx.Node):
            return False
        if current_node.op != 'call_module':
            return False
        if not isinstance(current_node.target, str):
            return False
        if current_node.target not in modules:
            return False
        if type(modules[current_node.target]) is not expected_type:
            return False
    return True


def replace_node_module(node: fx.Node, modules: Dict[str, Any], new_module: torch.nn.Module):
    assert(isinstance(node.target, str))
    parent_name, name = _parent_name(node.target)
    modules[node.target] = new_module
    setattr(modules[parent_name], name, new_module)


def fuse(model: torch.nn.Module, inplace=False) -> torch.nn.Module:
    """
    Fuses convolution/BN layers for inference purposes. Will deepcopy your
    model by default, but can modify the model inplace as well.
    """
    patterns = [(nn.Conv1d, nn.BatchNorm1d),
                (nn.Conv2d, nn.BatchNorm2d),
                (nn.Conv3d, nn.BatchNorm3d)]
    if not inplace:
        model = copy.deepcopy(model).eval()
    fx_model = fx.symbolic_trace(model)
    modules = dict(fx_model.named_modules())
    new_graph = copy.deepcopy(fx_model.graph)

    for pattern in patterns:
        for node in new_graph.nodes:
            if matches_module_pattern(pattern, node, modules):
                if len(node.args[0].users) > 1:  # Output of conv is used by other nodes
                    continue
                conv = modules[node.args[0].target]
                bn = modules[node.target]
                fused_conv = fuse_conv_bn_eval(conv, bn)
                replace_node_module(node.args[0], modules, fused_conv)
                node.replace_all_uses_with(node.args[0])
                new_graph.erase_node(node)
    return fx.GraphModule(fx_model, new_graph)


def remove_dropout(model: nn.Module) -> nn.Module:
    """
    Removes all dropout layers from the module.
    """
    fx_model = fx.symbolic_trace(model)

    class DropoutRemover(torch.fx.Transformer):
        def call_module(self, target : Target, args : Tuple[Argument, ...], kwargs : Dict[str, Any]) -> Any:
            if isinstance(self.submodules[target], nn.Dropout):
                assert len(args) == 1
                return args[0]
            else:
                return super().call_module(target, args, kwargs)
    return DropoutRemover(fx_model).transform()

def cast_to_memse(model:nn.Module, opmap:dict):
    # Fuse conv-bn
    fused = fuse(model)
    fused = remove_dropout(fused)
    #fused.graph.print_tabular()
    modules = dict(fused.named_modules())

    old_modules: Dict[nn.Module, nn.Module] = {}
    # Replace layers with memse layers
    for node in list(fused.graph.nodes):
        if node.op == 'call_module':
            assert(isinstance(node.target, str))
            cur_module = modules[node.target]
            if type(cur_module) in opmap:
                new_module = opmap[type(cur_module)](cur_module)
                assert(isinstance(new_module, nn.Module))
                old_modules[new_module] = copy.deepcopy(cur_module)
                replace_node_module(node, modules, new_module)
        elif node.op == 'call_function' or node.op == 'call_method':
            #  reference https://github.com/pytorch/examples/blob/main/fx/replace_op.py
            if node.target in opmap:
                # Set the insert point, add the new node, and replace all uses
                # of `n` with the new node
                with fused.graph.inserting_after(node):
                    new_node = fused.graph.call_function(opmap[node.target], node.args, node.kwargs)
                    node.replace_all_uses_with(new_node)
                # Remove the old node from the graph
                fused.graph.erase_node(node)
            else:
                raise ValueError(f'An unknown operation or method was cast ({node.target})')

    fused.recompile()
    # print('After cast')
    #fused.graph.print_tabular()

    # Assert all layers are memse layers/memse acts
    for m in fused.modules():
        if not 'torch.nn.modules.module.Module' in str(type(m)) and not 'MemSE' in str(type(m)) and not 'GraphModuleImpl' in str(type(m)):
            raise ValueError(f'A module has not been cast to MemSE equiv. ({type(m)})')

    # Return handle to MemSE module.
    return fused

def reset_modules(nodes: List[fx.Node], modules: Dict[str, nn.Module], old_modules: Dict[nn.Module, nn.Module]):
    """
    Maps each module that's been changed with `modules_to_mkldnn` back to its
    original.
    """
    for node in nodes:
        if node.op == 'call_module':
            assert(isinstance(node.target, str))
            cur_module = modules[node.target]
            if cur_module in old_modules:
                replace_node_module(node, modules, old_modules[cur_module])