from functools import partial

from ..conftest import Model1, Model2, Model3, Model4
from modelscope import module_walker, register_wrapped_forward_hook


def test_register_hook_on_model1():
    model = ("model1", Model1())

    module_ids = set()
    results = []

    modules = module_walker(model, yield_parents=False)
    register_wrapped_forward_hook_ = partial(
        register_wrapped_forward_hook,
        module_ids=module_ids,
        results=results,
    )
    handles = [*filter(None, map(register_wrapped_forward_hook_, modules))]

    assert module_ids == {id(model[1])}
    assert results == []
    assert handles != []
    assert len(handles) == 1
    assert handles[0] is not None


def test_register_hook_on_model2():
    model = ("model2", Model2())

    module_ids = set()
    results = []

    modules = module_walker(model, yield_parents=False)
    register_wrapped_forward_hook_ = partial(
        register_wrapped_forward_hook,
        module_ids=module_ids,
        results=results,
    )
    handles = [*filter(None, map(register_wrapped_forward_hook_, modules))]

    assert module_ids == {id(model[1].fc)}
    assert results == []
    assert handles != []
    assert len(handles) == 1
    assert handles[0] is not None


def test_register_hook_on_model3():
    model = ("model3", Model3())

    module_ids = set()
    results = []

    modules = module_walker(model, yield_parents=False)
    register_wrapped_forward_hook_ = partial(
        register_wrapped_forward_hook,
        module_ids=module_ids,
        results=results,
    )
    handles = [*filter(None, map(register_wrapped_forward_hook_, modules))]

    assert module_ids == {id(model[1].fc1), id(model[1].fc2)}
    assert results == []
    assert handles != []
    assert len(handles) == 2
    assert None not in handles


def test_register_hook_on_model4():
    model = ("model4", Model4())

    module_ids = set()
    results = []

    modules = module_walker(model, yield_parents=False)
    register_wrapped_forward_hook_ = partial(
        register_wrapped_forward_hook,
        module_ids=module_ids,
        results=results,
    )
    handles = [*filter(None, map(register_wrapped_forward_hook_, modules))]

    assert len(module_ids) == 1
    assert module_ids == {id(model[1].fc)}
    assert results == []
    assert handles != []
    assert len(handles) == 1
    assert handles[0] is not None
