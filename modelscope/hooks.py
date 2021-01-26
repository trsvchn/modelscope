"""Model hooks, handles etc.
"""
from typing import List, Set, Callable

import torch.nn as nn
from torch.utils.hooks import RemovableHandle

from .core import Result, Module
from .summary import get_size, get_num_params


def register_forward_hook(
        hook: Callable,
        module: nn.Module,
        module_ids: Set[int],
) -> RemovableHandle:
    """Registers a forward hook using module's internal method.
    """
    id_: int = id(module)
    if id_ not in module_ids:
        module_ids.add(id_)
        # Register forward hook here
        handle = module.register_forward_hook(hook)
        return handle


def prepare_forward_hook(module_name: str, results: List[Result]) -> Callable:
    """Prepares forward hook.
    """

    def hook(module, inp, out):
        """Get output sizes and module parameters.
        """
        # Get output size
        out_size = get_size(out)
        # Get params
        params = get_num_params(module)
        # Export info
        result = Result(module_name, module._get_name(), out_size, params)
        results.append(result)

    return hook


def register_wrapped_forward_hook(
        module: Module,
        module_ids: Set[int],
        results: List[Result],
) -> RemovableHandle:
    """Prepare and register forward hook.
    """
    hook = prepare_forward_hook(module.name, results)
    handle = register_forward_hook(hook, module.obj, module_ids)
    return handle
