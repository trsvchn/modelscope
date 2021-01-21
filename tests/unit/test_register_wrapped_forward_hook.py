import torch.nn as nn
from torch.utils.hooks import RemovableHandle

from modelscope import register_wrapped_forward_hook


def test_register_wrapped_forward_hook():
    module_name = "test_module"
    module = nn.Module()
    module_ids = set()
    results = []
    handle = register_wrapped_forward_hook(module_name, module, module_ids, results)
    assert isinstance(handle, RemovableHandle)
    assert {id(module)} == module_ids
    assert results == []
