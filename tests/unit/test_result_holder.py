from typing import Generator

import torch.nn as nn

from modelscope import result_holder
from modelscope.core import HookOutput, Result


def test_result_holder_init():
    results = result_holder()
    assert isinstance(results, Generator)
    out = next(results)
    assert out is None
    results.close()


def test_result_holder_send_none():
    results = result_holder()
    next(results)
    out = results.send(None)
    assert out is None
    results.close()


def test_result_holder_close():
    results = result_holder()
    next(results)
    summary = results.throw(StopIteration)
    results.close()
    expected = [
        Result("final", None, None, None, None, None, (0, 0, 0)),
    ]
    assert summary == expected


def test_result_holder_send_forward():
    results = result_holder()
    next(results)
    results.send(HookOutput("forward", ["model"], [""], False, nn.Linear(1, 1), None, None))
    summary = results.throw(StopIteration)
    results.close()
    expected = [
        Result("forward", ["model"], [""], False, "Linear", None, (2, 0)),
        Result("final", None, None, None, None, None, (2, 2, 0)),
    ]
    assert summary == expected


def test_result_holder_send_pre_forward():
    results = result_holder()
    next(results)
    results.send(HookOutput("pre_forward", ["model"], [""], False, nn.Linear(1, 1), None, None))
    summary = results.throw(StopIteration)
    results.close()
    expected = [
        Result("pre_forward", ["model"], [""], False, "Linear", None, None),
        Result("final", None, None, None, None, None, (0, 0, 0)),
    ]
    assert summary == expected


def test_result_holder_send_both():
    results = result_holder()
    next(results)
    results.send(HookOutput("pre_forward", ["model"], [""], False, nn.Linear(1, 1), None, None))
    results.send(HookOutput("forward", ["model"], [""], False, nn.Linear(1, 1), None, None))
    summary = results.throw(StopIteration)
    results.close()
    expected = [
        Result("pre_forward", ["model"], [""], False, "Linear", None, None),
        Result("forward", ["model"], [""], False, "Linear", None, (2, 0)),
        Result("final", None, None, None, None, None, (2, 2, 0)),
    ]
    assert summary == expected
