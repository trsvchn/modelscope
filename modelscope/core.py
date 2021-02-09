"""Core module.
"""
import operator as op
from copy import copy
from collections import namedtuple
from functools import reduce
from itertools import chain, groupby
from typing import Tuple, List, Generator, Union, Optional

import torch
import torch.nn as nn

from .utils import adjust_module_name

Module = namedtuple("Module", "name obj parent is_parent")
Handle = namedtuple("Handle", "obj module names parents is_parent")
HookOutput = namedtuple("HookOutput", "type names parents is_parent module inp out time")
Log = namedtuple("Log", "event category type names parents is_parent out_size num_params time")


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


class SummaryHandler:

    fn_types = {
        "method_descriptor": "method",
        "method": "method",
        "builtin_function_or_method": "function",
        "function": "function",
    }

    def __init__(self, logs, model_name):
        self.logs = logs
        self.model_name = model_name

        self.full_names = True
        self.full_type_names = False
        self.hide_names = []
        self.hide_types = []
        self.exclude_hidden = True
        self.fold_nodes = []
        self.top_level = False
        self.low_level = False
        self.max_depth = 1000

        self.depth = -1
        self.count = 1
        self.state = 1
        self.curr_module = [""]
        self.curr_parent = [""]

        self.total_num_train_params = 0
        self.total_num_non_train_params = 0

        self.fold = False
        self.force_hide = False
        self.prev_is_suppressed = False

    def reset(self):
        self.depth = -1
        self.count = 1
        self.state = 1
        self.curr_module = [""]
        self.curr_parent = [""]

        self.total_num_train_params = 0
        self.total_num_non_train_params = 0

        self.fold = False
        self.force_hide = False
        self.prev_is_suppressed = False

    def update_attrs(
            self,
            full_names: bool = True,
            full_type_names: bool = False,
            hide_names: Optional[List[str]] = None,
            hide_types: Optional[List[str]] = None,
            exclude_hidden: bool = True,
            fold_nodes: Optional[List[str]] = None,
            top_level: bool = False,
            low_level: bool = False,
            max_depth: int = 1000,
    ):
        self.full_names = full_names
        self.full_type_names = full_type_names
        self.hide_names = hide_names or []
        self.hide_types = hide_types or []
        self.hide_types = [t.lower() for t in self.hide_types]
        self.exclude_hidden = exclude_hidden
        self.fold_nodes = fold_nodes or []
        self.top_level = top_level
        self.low_level = low_level
        self.max_depth = max_depth

    def module_start(self, module_type, module_names, module_parents):
        if not self.low_level:
            if module_type in dir(torch.nn):
                self.fold = True

        self.depth += 1
        if self.state == 1:
            self.curr_parent = copy(self.curr_module)
        self.state = 1

        curr_parent = ".".join(self.curr_parent)
        module_name, module_parent = adjust_module_name(module_names, module_parents, curr_parent)

        self.curr_module.append(module_name)

        full_module_name = ".".join(self.curr_module[2:])
        if full_module_name in self.fold_nodes:
            self.force_hide = True

    def module_end(self, module_type, module_names, module_parents, out_size, num_params):
        if not self.low_level:
            if module_type in dir(torch.nn):
                self.fold = False

        is_comp = True if self.state == 0 else False
        self.state = 0

        self.curr_parent = self.curr_module[:-1]
        curr_parent = ".".join(self.curr_parent)
        module_name, module_parent = adjust_module_name(module_names, module_parents, curr_parent)

        if self.curr_module[-1] != module_name:
            raise RuntimeError(f"Module name mismatch error: {self.curr_module[-1]} != {module_name}")

        *_, num_train_params, num_non_train_params = num_params
        self.total_num_train_params += num_train_params
        self.total_num_non_train_params += num_non_train_params

        full_module_name = ".".join(self.curr_module[2:])
        display = (self.depth == 1) if self.top_level else (
                not is_comp or full_module_name in self.fold_nodes)

        out = None
        if display:
            out = out if self.exclude_hidden else False
            out = None if self.prev_is_suppressed else out
            if out is False:
                self.prev_is_suppressed = True

            if self.depth <= self.max_depth:
                if module_type.lower() not in self.hide_types:
                    if full_module_name not in self.hide_names:
                        if (not self.force_hide) or (self.force_hide and is_comp):
                            if self.full_names:
                                module_name = full_module_name
                            else:
                                module_name = self.curr_module[-1]

                            out = (
                                copy(self.count),
                                module_name,
                                module_type,
                                out_size,
                                num_params,
                            )
                            self.prev_is_suppressed = False
                            self.count += 1

        self.depth -= 1
        self.curr_module.pop()
        if full_module_name in self.fold_nodes:
            self.force_hide = False

        return out

    def fn_start(self, fn_name):
        if not self.fold:
            self.depth += 1
            if self.state == 1:
                self.curr_parent = copy(self.curr_module)
            self.state = 1

            self.curr_module.append(fn_name)

            full_fn_name = ".".join(self.curr_module[2:])
            if full_fn_name in self.fold_nodes:
                self.force_hide = True

    def fn_end(self, func_type, fn_name, out_size, num_params):
        num_params = num_params or (0, 0, 0, 0)
        if not self.fold:
            is_comp = True if self.state == 0 else False
            self.state = 0

            self.curr_parent = self.curr_module[:-1]

            name = fn_name

            if self.curr_module[-1] != name:
                raise RuntimeError(f"Module name mismatch error: {self.curr_module[-1]} != {name}")

            full_fn_name = ".".join(self.curr_module[2:])
            display = (self.depth == 1) if self.top_level else (not is_comp or full_fn_name in self.fold_nodes)

            out = None
            if display:
                out = out if self.exclude_hidden else False
                out = None if self.prev_is_suppressed else out
                if out is False:
                    self.prev_is_suppressed = True

                if self.depth <= self.max_depth:
                    fn_type = self.fn_types.get(func_type.__name__, "unknown")
                    if fn_type.lower() not in self.hide_types:
                        if self.full_type_names:
                            fn_type = f"{name} ({func_type.__name__})"
                        if full_fn_name not in self.hide_names:
                            if (not self.force_hide) or (self.force_hide and is_comp):
                                if self.full_names:
                                    fn_name = full_fn_name
                                else:
                                    fn_name = self.curr_module[-1]

                                out = (
                                    copy(self.count),
                                    fn_name,
                                    fn_type,
                                    out_size,
                                    num_params,
                                )
                                self.prev_is_suppressed = False
                                self.count += 1

            self.depth -= 1
            self.curr_module.pop()
            if full_fn_name in self.fold_nodes:
                self.force_hide = False
            return out
