from types import FunctionType

import torch
import torch.nn as nn

from modelscope.core import Result
from modelscope.hooks import prepare_forward_hook
from modelscope.summary import result_holder


def test_prepare_forward_hook_return():
    module_name = [""]
    module_parent = [""]
    results = result_holder()
    next(results)
    expected = [
        Result("final", None, None, None, None, None, (0, 0, 0)),
    ]
    is_parent = True
    hook = prepare_forward_hook(module_name, module_parent, is_parent, results)
    summary = results.throw(StopIteration)
    results.close()
    assert isinstance(hook, FunctionType)
    assert summary == expected


def test_forward_hook():
    module_name = ["fc"]
    module_parent = [""]
    results = result_holder()
    next(results)
    is_parent = False
    hook = prepare_forward_hook(module_name, module_parent, is_parent, results)
    module = nn.Linear(2, 1)
    inp = torch.randn(1, 2)
    out = torch.randn(1, 1)
    expected = [
        Result("forward", ["fc"], [""], False, "Linear", "[1, 1]", (3, 0)),
        Result("final", None, None, None, None, None, (3, 3, 0)),
    ]
    hook(module, inp, out)
    summary = results.throw(StopIteration)
    results.close()
    assert summary == expected
