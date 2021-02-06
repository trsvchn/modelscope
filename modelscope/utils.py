"""Utilities.
"""
from typing import Tuple, List, Any

import torch


def size_to_str(size: Any) -> str:
    """Size to string converter.
    """
    if isinstance(size, torch.Size):
        out = f"[{', '.join(map(str, size))}]"
    elif isinstance(size, (tuple, list)):
        out = f"[{', '.join(size_to_str(s) for s in size)}]"
    elif size is None:
        out = "None"
    else:
        out = ""
    return out


def adjust_module_name(
        module_names: List[str],
        module_parents: List[str],
        current_parent: str,
) -> Tuple[str, str]:
    """Adjust module name accordingly to current parent name.
    """
    module_name = module_names
    module_parent = module_parents

    if (len(module_names) > 1) and (len(module_parents) > 1):
        for n, p in zip(module_names, module_parents):
            if current_parent == p:
                module_name = n
                module_parent = p

    elif (len(module_names) == 1) and (len(module_parents) == 1):
        module_name = module_names[0]
        module_parent = module_parents[0]

    return module_name, module_parent
