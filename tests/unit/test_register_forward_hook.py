import torch.nn as nn

from modelscope import register_forward_hook


def test_register_forward_hook_unique():
    module_ids = set()
    hook = lambda module, inp, out: None
    module = nn.Module()
    assert set() == module_ids

    handle = register_forward_hook(hook, module, module_ids)
    assert 0 == handle.id
    assert 1 == handle.next_id
    assert hook == handle.hooks_dict_ref()[0]
    assert {id(module)} == module_ids


def test_register_forward_hook_duplicated():
    module_ids = set()
    hook = lambda module, inp, out: None
    module = nn.Module()
    assert set() == module_ids

    module_ids.add(id(module))  # Module already has a registered hook
    handle = register_forward_hook(hook, module, module_ids)
    assert handle is None
    assert {id(module)} == module_ids
