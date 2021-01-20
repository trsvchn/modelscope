"""Model Summary.
"""
from typing import Tuple, Iterator, Generator

import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle


def module_walker(module: Tuple[str, nn.Module], parents: bool = True) -> Iterator[Tuple[str, nn.Module]]:
    """Recursive model walker. By default, it returns parents and children modules.
    """
    if parents:
        # Yield module anyway, when we want parents modules as well
        yield module

    # Get module submodules (children)
    submodules: Generator = module[-1].named_children()
    # Trying to get the first submodule
    submodule: Tuple[str, nn.Module] = next(submodules, None)

    # If possible go deeper
    if submodule is not None:
        # First submodule is already here
        yield from module_walker(submodule, parents=parents)
        # Next submodules
        for m in submodules:
            # If there is more than one submodule
            if m:
                yield from module_walker(m, parents=parents)

    elif submodule is None and not parents:
        yield module


def register_forward_hook(hook, module: nn.Module, module_ids: set = None) -> RemovableHandle:
    """Registers a forward hook using module's internal method.
    """
    id_ = id(module)
    if id_ not in module_ids:
        module_ids.add(id_)
        # Register forward hook here
        handle = module.register_forward_hook(hook)
        return handle
