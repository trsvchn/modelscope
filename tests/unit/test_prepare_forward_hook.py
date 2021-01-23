from types import FunctionType

import torch
import torch.nn as nn

from modelscope import Result, prepare_forward_hook


def test_prepare_forward_hook_return():
    module_name = ""
    results = []
    hook = prepare_forward_hook(module_name, results)
    assert isinstance(hook, FunctionType)
    assert results == []
    assert module_name == ""


def test_forward_hook():
    module_name = "fc"
    results = []
    hook = prepare_forward_hook(module_name, results)

    module = nn.Linear(2, 1)
    inp = torch.randn(1, 2)
    out = torch.randn(1, 1)
    expected = (
        Result("fc", "Linear", torch.Size([1, 1]), None),
    )
    hook(module, inp, out)
    assert expected == tuple(results)
