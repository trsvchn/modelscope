import torch.nn as nn
from torch.utils.hooks import RemovableHandle

from modelscope import Module, register_wrapped_forward_hook


def test_register_wrapped_forward_hook():
    module = Module("module", nn.Module())
    module_ids = set()
    results = []
    handle = register_wrapped_forward_hook(module, module_ids, results)
    assert isinstance(handle, RemovableHandle)
    assert {id(module.obj)} == module_ids
    assert results == []
