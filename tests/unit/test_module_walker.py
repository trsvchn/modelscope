from ..conftest import Model1, Model2, Model3, Model4
from modelscope import Module, module_walker


def test_model1_w_parents():
    model = ("model1", Model1())
    expected = (
        Module(*model),
    )
    outputs = tuple(module_walker(model, parents=True))
    assert expected == outputs


def test_model2_w_parents():
    model = ("model2", Model2())
    expected = (
        Module(*model),
        Module("fc", model[-1].fc),
    )
    outputs = tuple(module_walker(model, parents=True))
    assert expected == outputs


def test_model3_w_parents():
    model = ("model3", Model3())
    expected = (
        Module(*model),
        Module("fc1", model[-1].fc1),
        Module("fc2", model[-1].fc2),
    )
    outputs = tuple(module_walker(model, parents=True))
    assert expected == outputs


def test_model4_w_parents():
    model = ("model4", Model4())
    expected = (
        Module(*model),
        Module("fc", model[-1].fc),
        Module("seq", model[-1].seq),
        Module("0", model[-1].fc),
    )
    outputs = tuple(module_walker(model, parents=True))
    assert expected == outputs


def test_model1_children():
    model = ("model1", Model1())
    expected = (
        Module(*model),
    )
    outputs = tuple(module_walker(model, parents=False))
    assert expected == outputs


def test_model2_children():
    model = ("model2", Model2())
    expected = (
        Module("fc", model[-1].fc),
    )
    outputs = tuple(module_walker(model, parents=False))
    assert expected == outputs


def test_model3_children():
    model = ("model3", Model3())
    expected = (
        Module("fc1", model[-1].fc1),
        Module("fc2", model[-1].fc2),
    )
    outputs = tuple(module_walker(model, parents=False))
    assert expected == outputs


def test_model4_children():
    model = ("model4", Model4())
    expected = (
        Module("fc", model[-1].fc),
        Module("0", model[-1].fc),
    )
    outputs = tuple(module_walker(model, parents=False))
    assert expected == outputs
