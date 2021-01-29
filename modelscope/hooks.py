"""Model hooks, handles etc.
"""
from typing import List, Set, Tuple, Callable, Generator, Optional

import torch.nn as nn
from torch.utils.hooks import RemovableHandle

from .core import Module, Handle, HookOutput


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


def prepare_pre_forward_hook(
        module_names: List[str],
        module_parents: List[str],
        is_parent: bool,
        logger: Generator[List[str], Optional[HookOutput], None]) -> Callable:
    """Prepares pre forward hook.
    """

    def hook(module, inp):
        # Export info
        output = HookOutput("pre_forward", module_names, module_parents, is_parent, module, inp, None)
        logger.send(output)

    return hook


def prepare_forward_hook(
        module_names: List[str],
        module_parents: List[str],
        is_parent: bool,
        logger: Generator[List[str], Optional[HookOutput], None]) -> Callable:
    """Prepares forward hook.
    """

    def hook(module, inp, out):
        # Export info
        output = HookOutput("forward", module_names, module_parents, is_parent, module, inp, out)
        logger.send(output)

    return hook


def register_wrapped_forward_hook(
        module: Module,
        module_ids: Set[int],
        logger: Generator[List[str], Optional[HookOutput], None],
) -> RemovableHandle:
    """Prepare and register forward hook.
    """
    hook = prepare_forward_hook(module.name, module.parent, module.is_parent, logger)
    handle = register_forward_hook(hook, module.obj, module_ids)
    return handle


def prepare_handles(modules: Generator[Module, None, None]) -> Tuple[dict, dict]:
    """Experimental function to handle cases when the same module
    is reused in multiple places under different names. (See Model4 from tests/conf.py.)
    """
    ids_pre_forward = set()
    handles_pre_forward = {}

    ids_forward = set()
    handles_forward = {}

    # Iterate over modules
    for m in modules:
        module_id = id(m.obj)

        for type_, ids_, handles in (
                ("_forward_pre_hooks", ids_pre_forward, handles_pre_forward),
                ("_forward_hooks", ids_forward, handles_forward),
        ):
            if module_id not in ids_:
                # Create a handle for a module
                handle = RemovableHandle(getattr(m.obj, type_))
                # Store handle in a dict
                handles[module_id] = Handle(handle, m.obj, [m.name], [m.parent], m.is_parent)
                # Update ids
                ids_.add(module_id)
            else:
                # If Module already registered append other name and parent
                handles[module_id].names.append(m.name)
                handles[module_id].parents.append(m.parent)

    return handles_pre_forward, handles_forward


def register_forward_hooks(handles, logger):
    """Registers forward hook on modules.
    """
    for handle in handles.values():
        id_ = handle.obj.id
        names = handle.names
        parents = handle.parents
        is_parent = handle.is_parent
        handle.module._forward_hooks[id_] = prepare_forward_hook(names, parents, is_parent, logger)


def register_pre_forward_hooks(handles, logger):
    """Registers pre-forward hook on modules.
    """
    for handle in handles.values():
        id_ = handle.obj.id
        names = handle.names
        parents = handle.parents
        is_parent = handle.is_parent
        handle.module._forward_pre_hooks[id_] = prepare_pre_forward_hook(names, parents, is_parent, logger)
