from typing import Generator

import torch.nn as nn

from modelscope import logger
from modelscope.core import HookOutput, Log


def test_result_holder_init():
    main_logger = logger()
    assert isinstance(main_logger, Generator)
    out = next(main_logger)
    assert out is None
    main_logger.close()


def test_result_holder_send_none():
    main_logger = logger()
    next(main_logger)
    out = main_logger.send(None)
    assert out is None
    main_logger.close()


def test_result_holder_close():
    main_logger = logger()
    next(main_logger)
    logs = main_logger.throw(StopIteration)
    main_logger.close()
    expected = [
        Log("final", None, None, None, None, None, (0, 0, 0)),
    ]
    assert logs == expected


def test_result_holder_send_forward():
    main_logger = logger()
    next(main_logger)
    main_logger.send(HookOutput("forward", ["model"], [""], False, nn.Linear(1, 1), None, None, 0.0))
    logs = main_logger.throw(StopIteration)
    main_logger.close()
    expected = [
        Log("forward", ["model"], [""], False, "Linear", None, (2, 0)),
        Log("final", None, None, None, None, None, (2, 2, 0)),
    ]
    assert logs == expected


def test_result_holder_send_pre_forward():
    main_logger = logger()
    next(main_logger)
    main_logger.send(HookOutput("pre_forward", ["model"], [""], False, nn.Linear(1, 1), None, None, 0.0))
    logs = main_logger.throw(StopIteration)
    main_logger.close()
    expected = [
        Log("pre_forward", ["model"], [""], False, "Linear", None, None),
        Log("final", None, None, None, None, None, (0, 0, 0)),
    ]
    assert logs == expected


def test_result_holder_send_both():
    main_logger = logger()
    next(main_logger)
    main_logger.send(HookOutput("pre_forward", ["model"], [""], False, nn.Linear(1, 1), None, None, 0.0))
    main_logger.send(HookOutput("forward", ["model"], [""], False, nn.Linear(1, 1), None, None, 1.0))
    logs = main_logger.throw(StopIteration)
    main_logger.close()
    expected = [
        Log("pre_forward", ["model"], [""], False, "Linear", None, None),
        Log("forward", ["model"], [""], False, "Linear", None, (2, 0)),
        Log("final", None, None, None, None, None, (2, 2, 0)),
    ]
    assert logs == expected
