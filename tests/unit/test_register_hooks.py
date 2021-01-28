from ..conftest import Model1, Model2, Model3, Model4
from modelscope.core import module_walker
from modelscope.hooks import prepare_handles, register_pre_forward_hooks, register_forward_hooks
from modelscope import result_holder


def test_register_hooks_model1():
    model = ("model1", Model1())
    modules = module_walker(model, yield_parents=True)
    handles_pre_forward, handles_forward = prepare_handles(modules)
    results = result_holder()
    register_pre_forward_hooks(handles_pre_forward, results)
    register_forward_hooks(handles_forward, results)
    assert len(model[-1]._forward_pre_hooks.items()) == 1
    assert len(model[-1]._forward_hooks.items()) == 1


def test_register_hooks_model2():
    model = ("model2", Model2())
    modules = module_walker(model, yield_parents=True)
    handles_pre_forward, handles_forward = prepare_handles(modules)
    results = result_holder()
    register_pre_forward_hooks(handles_pre_forward, results)
    register_forward_hooks(handles_forward, results)
    assert len(model[-1]._forward_pre_hooks.items()) == 1
    assert len(model[-1]._forward_hooks.items()) == 1
    assert len(model[-1].fc._forward_pre_hooks.items()) == 1
    assert len(model[-1].fc._forward_hooks.items()) == 1


def test_register_hooks_model3():
    model = ("model3", Model3())
    modules = module_walker(model, yield_parents=True)
    handles_pre_forward, handles_forward = prepare_handles(modules)
    results = result_holder()
    register_pre_forward_hooks(handles_pre_forward, results)
    register_forward_hooks(handles_forward, results)
    assert len(model[-1]._forward_pre_hooks.items()) == 1
    assert len(model[-1]._forward_hooks.items()) == 1
    assert len(model[-1].fc1._forward_pre_hooks.items()) == 1
    assert len(model[-1].fc1._forward_hooks.items()) == 1
    assert len(model[-1].fc2._forward_pre_hooks.items()) == 1
    assert len(model[-1].fc2._forward_hooks.items()) == 1


def test_register_hooks_model4():
    model = ("model4", Model4())
    modules = module_walker(model, yield_parents=True)
    handles_pre_forward, handles_forward = prepare_handles(modules)
    results = result_holder()
    register_pre_forward_hooks(handles_pre_forward, results)
    register_forward_hooks(handles_forward, results)
    assert len(model[-1]._forward_pre_hooks.items()) == 1
    assert len(model[-1]._forward_hooks.items()) == 1
    assert len(model[-1].fc._forward_pre_hooks.items()) == 1
    assert len(model[-1].fc._forward_hooks.items()) == 1
    assert len(model[-1].seq._forward_pre_hooks.items()) == 1
    assert len(model[-1].seq._forward_hooks.items()) == 1
