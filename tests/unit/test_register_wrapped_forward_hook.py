import torch.nn as nn
from torch.utils.hooks import RemovableHandle

from modelscope.core import Module
from modelscope.hooks import register_wrapped_forward_hook
from modelscope.summary import result_holder


def test_register_wrapped_forward_hook():
    module = Module("module", nn.Module(), parent="", is_parent=False)
    module_ids = set()
    results = result_holder()
    handle = register_wrapped_forward_hook(module, module_ids, results)
    assert isinstance(handle, RemovableHandle)
    assert {id(module.obj)} == module_ids
    results.close()
