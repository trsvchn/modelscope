"""Model Summary.
"""
from functools import reduce
from itertools import chain, groupby
from typing import Tuple, List, Optional, Union

import torch
import torch.nn as nn


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
