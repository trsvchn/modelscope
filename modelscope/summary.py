"""Model Summary.
"""
from collections import namedtuple
from functools import reduce
from itertools import chain, groupby
from typing import Tuple, List, Iterator, Set, Generator, Callable

import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle


Module = namedtuple("Module", "name obj")
Result = namedtuple("Result", "name module out_size num_params")


def module_walker(
        module: Tuple[str, nn.Module],
        parents: bool = True,
) -> Iterator[Module]:
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
        out_size = out.size()
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
                lambda kg: {kg[0]: reduce(torch.add, map(lambda group: group[-1], kg[1]), torch.zeros(1))},
                num_params_grouped
            )
            # Convert to dict
            num_params_dict = {[*p.keys()][0]: int([*p.values()][0].item()) for p in num_params}
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
