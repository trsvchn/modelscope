"""Model Summary.
"""
import operator as op
from functools import reduce
from itertools import chain, groupby
from typing import Tuple, List, Optional, Union, Generator

import torch
import torch.nn as nn

from .core import HookOutput, Log


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


def logger() -> Generator[List[str], Optional[HookOutput], None]:
    """A kind of logger+filter. Stores and processes hook outputs. Yields logs.
    """
    ids_ = set()
    logs = []
    total_num_train_params = 0
    total_num_non_train_params = 0

    try:
        while True:
            try:
                hook_output: Optional[HookOutput] = (yield)
                if hook_output is not None:
                    if hook_output.type == "pre_forward":
                        # This handles outputs from pre-forward hooks

                        pre_forward = Log(
                            hook_output.type,
                            hook_output.names,
                            hook_output.parents,
                            hook_output.is_parent,
                            hook_output.module._get_name(),
                            None, None,
                        )

                        logs.append(pre_forward)

                    elif hook_output.type == "forward":
                        # Forward hooks

                        module_type = hook_output.module._get_name()
                        out_size = get_size(hook_output.out)

                        num_params = get_num_params(hook_output.module)
                        num_train_params, num_non_train_params = num_params

                        forward = Log(
                            hook_output.type,
                            hook_output.names,
                            hook_output.parents,
                            hook_output.is_parent,
                            module_type,
                            out_size,
                            num_params,
                        )

                        logs.append(forward)

                        # Count only basic modules
                        if not hook_output.is_parent:
                            # Count only once
                            module_id = id(hook_output.module)
                            if module_id not in ids_:
                                total_num_train_params += num_train_params
                                total_num_non_train_params += num_non_train_params
                                ids_.add(module_id)

            except StopIteration:
                final = Log(
                    "final",
                    None, None, None, None, None,
                    num_params=(
                        total_num_train_params + total_num_non_train_params,
                        total_num_train_params,
                        total_num_non_train_params,
                    ),
                )
                logs.append(final)
                yield logs
    finally:
        ...
