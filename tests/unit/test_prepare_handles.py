from ..conftest import Model1, Model2, Model3, Model4
from modelscope.core import module_walker
from modelscope.hooks import prepare_handles


def test_prepare_handles_model1():
    model = ("model1", Model1())
    modules = module_walker(model, yield_parents=True)
    handles_pre_forward, handles_forward = prepare_handles(modules)
    assert isinstance(handles_pre_forward, dict)
    assert isinstance(handles_forward, dict)
    assert len(handles_pre_forward.items()) == 1
    assert len(handles_forward.items()) == 1

    handle = [*handles_pre_forward.values()][0]
    assert handle.module == model[-1]
    assert handle.names == ["model1"]
    assert handle.parents == [""]
    assert handle.is_parent is False
    handle = [*handles_forward.values()][0]
    assert handle.module == model[-1]
    assert handle.names == ["model1"]
    assert handle.parents == [""]
    assert handle.is_parent is False


def test_prepare_handles_model2():
    model = ("model2", Model2())
    modules = module_walker(model, yield_parents=True)
    handles_pre_forward, handles_forward = prepare_handles(modules)
    assert isinstance(handles_pre_forward, dict)
    assert isinstance(handles_forward, dict)
    assert len(handles_pre_forward.items()) == 2
    assert len(handles_forward.items()) == 2

    handle = [*handles_pre_forward.values()][0]
    assert handle.module == model[-1]
    assert handle.names == ["model2"]
    assert handle.parents == [""]
    assert handle.is_parent is True
    handle = [*handles_pre_forward.values()][1]
    assert handle.module == model[-1].fc
    assert handle.names == ["fc"]
    assert handle.parents == [".model2"]
    assert handle.is_parent is False

    handle = [*handles_forward.values()][0]
    assert handle.module == model[-1]
    assert handle.names == ["model2"]
    assert handle.parents == [""]
    assert handle.is_parent is True
    handle = [*handles_forward.values()][1]
    assert handle.module == model[-1].fc
    assert handle.names == ["fc"]
    assert handle.parents == [".model2"]
    assert handle.is_parent is False


def test_prepare_handles_model3():
    model = ("model3", Model3())
    modules = module_walker(model, yield_parents=True)
    handles_pre_forward, handles_forward = prepare_handles(modules)
    assert isinstance(handles_pre_forward, dict)
    assert isinstance(handles_forward, dict)
    assert len(handles_pre_forward.items()) == 3
    assert len(handles_forward.items()) == 3

    handle = [*handles_pre_forward.values()][0]
    assert handle.module == model[-1]
    assert handle.names == ["model3"]
    assert handle.parents == [""]
    assert handle.is_parent is True
    handle = [*handles_pre_forward.values()][1]
    assert handle.module == model[-1].fc1
    assert handle.names == ["fc1"]
    assert handle.parents == [".model3"]
    assert handle.is_parent is False
    handle = [*handles_pre_forward.values()][2]
    assert handle.module == model[-1].fc2
    assert handle.names == ["fc2"]
    assert handle.parents == [".model3"]
    assert handle.is_parent is False

    handle = [*handles_forward.values()][0]
    assert handle.module == model[-1]
    assert handle.names == ["model3"]
    assert handle.parents == [""]
    assert handle.is_parent is True
    handle = [*handles_forward.values()][1]
    assert handle.module == model[-1].fc1
    assert handle.names == ["fc1"]
    assert handle.parents == [".model3"]
    assert handle.is_parent is False
    handle = [*handles_forward.values()][2]
    assert handle.module == model[-1].fc2
    assert handle.names == ["fc2"]
    assert handle.parents == [".model3"]
    assert handle.is_parent is False


def test_prepare_handles_model4():
    model = ("model4", Model4())
    modules = module_walker(model, yield_parents=True)
    handles_pre_forward, handles_forward = prepare_handles(modules)
    assert isinstance(handles_pre_forward, dict)
    assert isinstance(handles_forward, dict)
    assert len(handles_pre_forward.items()) == 3
    assert len(handles_forward.items()) == 3

    handle = [*handles_pre_forward.values()][0]
    assert handle.module == model[-1]
    assert handle.names == ["model4"]
    assert handle.parents == [""]
    assert handle.is_parent is True
    handle = [*handles_pre_forward.values()][1]
    assert handle.module == model[-1].fc
    assert handle.names == ["fc", "0"]
    assert handle.parents == [".model4", ".model4.seq"]
    assert handle.is_parent is False
    handle = [*handles_pre_forward.values()][2]
    assert handle.module == model[-1].seq
    assert handle.names == ["seq"]
    assert handle.parents == [".model4"]
    assert handle.is_parent is True

    handle = [*handles_forward.values()][0]
    assert handle.module == model[-1]
    assert handle.names == ["model4"]
    assert handle.parents == [""]
    assert handle.is_parent is True
    handle = [*handles_forward.values()][1]
    assert handle.module == model[-1].fc
    assert handle.names == ["fc", "0"]
    assert handle.parents == [".model4", ".model4.seq"]
    assert handle.is_parent is False
    handle = [*handles_forward.values()][2]
    assert handle.module == model[-1].seq
    assert handle.names == ["seq"]
    assert handle.parents == [".model4"]
    assert handle.is_parent is True


def test_prepare_handles_model4_children():
    model = ("model4", Model4())
    modules = module_walker(model, yield_parents=False)
    handles_pre_forward, handles_forward = prepare_handles(modules)
    assert isinstance(handles_pre_forward, dict)
    assert isinstance(handles_forward, dict)
    assert len(handles_pre_forward.items()) == 1
    assert len(handles_forward.items()) == 1

    handle = [*handles_pre_forward.values()][0]
    assert handle.module == model[-1].fc
    assert handle.names == ["fc", "0"]
    assert handle.parents == [".model4", ".model4.seq"]
    assert handle.is_parent is False

    handle = [*handles_forward.values()][0]
    assert handle.module == model[-1].fc
    assert handle.names == ["fc", "0"]
    assert handle.parents == [".model4", ".model4.seq"]
    assert handle.is_parent is False
