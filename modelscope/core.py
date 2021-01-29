"""Core module.
"""
from collections import namedtuple
from typing import Tuple, Generator

import torch.nn as nn


Module = namedtuple("Module", "name obj parent is_parent")
Handle = namedtuple("Handle", "obj module names parents is_parent")
HookOutput = namedtuple("HookOutput", "type names parents is_parent module inp out")
Log = namedtuple("Log", "type names parents is_parent module_type out_size num_params")


def module_walker(
        module: Tuple[str, nn.Module],
        yield_parents: bool = True,
        parent=None,
) -> Generator[Module, None, None]:
    """Recursive model walker. By default, it returns parents and children modules.
    """
    parent = parent or ""
    # Get module submodules (children)
    submodules: Generator = module[-1].named_children()
    # Trying to get the first submodule
    submodule: Tuple[str, nn.Module] = next(submodules, None)

    # If possible go deeper
    if submodule is not None:
        if yield_parents:
            # Yield module anyway, when we need parents as well
            yield Module(*module, parent=parent, is_parent=True)

        # First submodule is already here
        parent = ".".join([parent, module[0]])
        yield from module_walker(submodule, yield_parents=yield_parents, parent=parent)
        # Next submodules
        for m in submodules:
            # If there is more than one submodule
            if m:
                yield from module_walker(m, yield_parents=yield_parents, parent=parent)

    elif submodule is None:
        yield Module(*module, parent=parent, is_parent=False)
