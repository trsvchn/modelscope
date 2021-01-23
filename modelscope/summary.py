"""Model Summary.
"""
from collections import namedtuple
from typing import Tuple, List, Iterator, Set, Generator, Callable

import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle


Module = namedtuple("Module", "name obj")


def module_walker(module: Tuple[str, nn.Module], parents: bool = True) -> Iterator[Module]:
    """Recursive model walker. By default, it returns parents and children modules.
    """
    # Get module submodules (children)
    submodules: Generator = module[-1].named_children()
    # Trying to get the first submodule
    submodule: Tuple[str, nn.Module] = next(submodules, None)

    # If possible go deeper
    if submodule is not None:
        if parents:
            # Yield module anyway, when we need parents as well
            yield Module(*module)

        # First submodule is already here
        yield from module_walker(submodule, parents=parents)
        # Next submodules
        for m in submodules:
            # If there is more than one submodule
            if m:
                yield from module_walker(m, parents=parents)

    elif submodule is None:
        yield Module(*module)


def register_forward_hook(hook: Callable, module: nn.Module, module_ids: Set[int]) -> RemovableHandle:
    """Registers a forward hook using module's internal method.
    """
    id_: int = id(module)
    if id_ not in module_ids:
        module_ids.add(id_)
        # Register forward hook here
        handle = module.register_forward_hook(hook)
        return handle


def prepare_forward_hook(module_name: str, results: List[Tuple[str, str, torch.Size, None]]) -> Callable:
    """Prepares forward hook.
    """

    def hook(module, inp, out):
        """Get output sizes and module parameters.
        """
        # Get output size
        out_size = out.size()
        # Get params
        params = None
        # Export info
        results.append((module_name, module._get_name(), out_size, params))

    return hook


def register_wrapped_forward_hook(
        module: Module,
        module_ids: Set[int],
        results: List[Tuple[str, str, torch.Size, None]],
) -> RemovableHandle:
    """Prepare and register forward hook.
    """
    hook = prepare_forward_hook(module.name, results)
    handle = register_forward_hook(hook, module.obj, module_ids)
    return handle
