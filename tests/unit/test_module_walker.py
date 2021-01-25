from ..conftest import Model1, Model2, Model3, Model4
from modelscope import Module, module_walker


def test_model1_w_parents():
    model = ("model1", Model1())
    expected = [
        Module(*model, parent="", container=False),
    ]
    outputs = [*module_walker(model, yield_parents=True)]
    assert outputs == expected


def test_model2_w_parents():
    model = ("model2", Model2())
    expected = [
        Module(*model, parent="", container=True),
        Module("fc", model[-1].fc, parent=".model2", container=False),
    ]
    outputs = [*module_walker(model, yield_parents=True)]
    assert outputs == expected


def test_model3_w_parents():
    model = ("model3", Model3())
    expected = [
        Module(*model, parent="", container=True),
        Module("fc1", model[-1].fc1, parent=".model3", container=False),
        Module("fc2", model[-1].fc2, parent=".model3", container=False),
    ]
    outputs = [*module_walker(model, yield_parents=True)]
    assert outputs == expected


def test_model4_w_parents():
    model = ("model4", Model4())
    expected = [
        Module(*model, parent="", container=True),
        Module("fc", model[-1].fc, parent=".model4", container=False),
        Module("seq", model[-1].seq, parent=".model4", container=True),
        Module("0", model[-1].fc, parent=".model4.seq", container=False),
    ]
    outputs = [*module_walker(model, yield_parents=True)]
    assert outputs == expected


def test_model1_children():
    model = ("model1", Model1())
    expected = [
        Module(*model, parent="", container=False),
    ]
    outputs = [*module_walker(model, yield_parents=False)]
    assert outputs == expected


def test_model2_children():
    model = ("model2", Model2())
    expected = [
        Module("fc", model[-1].fc, parent=".model2", container=False),
    ]
    outputs = [*module_walker(model, yield_parents=False)]
    assert outputs == expected


def test_model3_children():
    model = ("model3", Model3())
    expected = [
        Module("fc1", model[-1].fc1, parent=".model3", container=False),
        Module("fc2", model[-1].fc2, parent=".model3", container=False),
    ]
    outputs = [*module_walker(model, yield_parents=False)]
    assert outputs == expected


def test_model4_children():
    model = ("model4", Model4())
    expected = [
        Module("fc", model[-1].fc, parent=".model4", container=False),
        Module("0", model[-1].fc, parent=".model4.seq", container=False),
    ]
    outputs = [*module_walker(model, yield_parents=False)]
    assert outputs == expected
