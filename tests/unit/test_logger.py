from typing import Generator

import torch.nn as nn

from modelscope import logger
from modelscope.core import HookOutput, Log


def test_logger_init():
    main_logger = logger()
    assert isinstance(main_logger, Generator)
    out = next(main_logger)
    assert out is None
    main_logger.close()


def test_logger_send_none():
    main_logger = logger()
    next(main_logger)
    out = main_logger.send(None)
    assert out is None
    main_logger.close()


def test_logger_close():
    main_logger = logger()
    next(main_logger)
    logs = main_logger.throw(StopIteration)
    main_logger.close()
    expected = [
        Log("final", None, None, None, None, None, (0, 0, 0), None),
    ]
    assert logs == expected


def test_logger_send_forward():
    main_logger = logger(live=False)
    next(main_logger)
    main_logger.send(HookOutput("forward", ["model"], [""], False, nn.Linear(1, 1), None, None, 0.0))
    logs = main_logger.throw(StopIteration)
    main_logger.close()
    expected = [
        Log("forward", ["model"], [""], False, "Linear", None, (2, 0), 0.0),
        Log("final", None, None, None, None, None, (2, 2, 0), None),
    ]
    assert logs == expected


def test_logger_send_pre_forward():
    main_logger = logger()
    next(main_logger)
    main_logger.send(HookOutput("pre_forward", ["model"], [""], False, nn.Linear(1, 1), None, None, 0.0))
    logs = main_logger.throw(StopIteration)
    main_logger.close()
    expected = [
        Log("pre_forward", ["model"], [""], False, "Linear", None, None, 0.0),
        Log("final", None, None, None, None, None, (0, 0, 0), None),
    ]
    assert logs == expected


def test_logger_send_both():
    main_logger = logger()
    next(main_logger)
    main_logger.send(HookOutput("pre_forward", ["model"], [""], False, nn.Linear(1, 1), None, None, 0.0))
    main_logger.send(HookOutput("forward", ["model"], [""], False, nn.Linear(1, 1), None, None, 1.0))
    logs = main_logger.throw(StopIteration)
    main_logger.close()
    expected = [
        Log("pre_forward", ["model"], [""], False, "Linear", None, None, 0.0),
        Log("forward", ["model"], [""], False, "Linear", None, (2, 0), 1.0),
        Log("final", None, None, None, None, None, (2, 2, 0), None),
    ]
    assert logs == expected
