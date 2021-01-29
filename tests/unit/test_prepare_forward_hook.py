from types import FunctionType

import torch
import torch.nn as nn

from modelscope.core import Log
from modelscope.hooks import prepare_forward_hook
from modelscope.summary import logger


def test_prepare_forward_hook_return():
    module_name = [""]
    module_parent = [""]
    hook_logger = logger()
    next(hook_logger)
    expected = [
        Log("final", None, None, None, None, None, (0, 0, 0)),
    ]
    is_parent = True
    hook = prepare_forward_hook(module_name, module_parent, is_parent, hook_logger)
    logs = hook_logger.throw(StopIteration)
    hook_logger.close()
    assert isinstance(hook, FunctionType)
    assert logs == expected


def test_forward_hook():
    module_name = ["fc"]
    module_parent = [""]
    hook_logger = logger()
    next(hook_logger)
    is_parent = False
    hook = prepare_forward_hook(module_name, module_parent, is_parent, hook_logger)
    module = nn.Linear(2, 1)
    inp = torch.randn(1, 2)
    out = torch.randn(1, 1)
    expected = [
        Log("forward", ["fc"], [""], False, "Linear", torch.Size([1, 1]), (3, 0)),
        Log("final", None, None, None, None, None, (3, 3, 0)),
    ]
    hook(module, inp, out)
    logs = hook_logger.throw(StopIteration)
    hook_logger.close()
    assert logs == expected
