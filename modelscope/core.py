"""Core module.
"""
import operator as op
from collections import namedtuple
from functools import reduce
from itertools import chain, groupby
from typing import Tuple, List, Generator, Union, Optional

import torch
import torch.nn as nn


Module = namedtuple("Module", "name obj parent is_parent")
Handle = namedtuple("Handle", "obj module names parents is_parent")
HookOutput = namedtuple("HookOutput", "type names parents is_parent module inp out time")
Log = namedtuple("Log", "event category type names parents is_parent out_size num_params depth time")


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


def get_size(obj) -> Optional[Union[torch.Size, List[Optional[torch.Size]]]]:
    """Get size of input/output (if possible).
    """
    try:
        # Solo tensor
        out = obj.size()
    except AttributeError:
        # Multiple
        if isinstance(obj, (tuple, list)):
            out = []
            for o in obj:
                out.append(get_size(o))
        # Something else is just None (for now)
        else:
            out = None
    return out


def get_num_params(module: nn.Module) -> Tuple[int, int]:
    """Counts parameters of the module (trainable, non-trainable).
    """
    try:
        # Get params
        params = module.parameters()
        # Get the first one
        param = next(params, None)
        # Module contains params
        if param is not None:
            params = chain(param, params)
            # Count number of elements of Parameters
            num_params = map(lambda p: (p.requires_grad, p.numel()), params)
            # Group by trainable, non-trainable
            num_params_grouped = groupby(num_params, lambda p: p[0])
            # Count total number for each group
            num_params = map(
                lambda kg: {kg[0]: reduce(op.add, map(lambda group: group[-1], kg[1]), 0)},
                num_params_grouped
            )
            # Convert to dict
            num_params_dict = {[*p.keys()][0]: int([*p.values()][0]) for p in num_params}
            # Final unpack
            num_train_params = num_params_dict.get(True, 0)
            num_non_train_params = num_params_dict.get(False, 0)
            return num_train_params, num_non_train_params
        else:
            # If params empty
            return 0, 0
    except AttributeError:
        # No params at all
        return 0, 0
