import pytest

from ..conftest import Model1, Model2, Model3, Model4
from modelscope import get_num_params


def test_get_num_params_no_attr():
    model = None
    with pytest.raises(AttributeError):
        model.parameters()

    num_params = get_num_params(model)
    assert num_params == (0, 0)


def test_get_num_params_model1():
    model = Model1()
    assert next(model.parameters(), None) is None

    num_params = get_num_params(model)
    assert num_params == (0, 0)


def test_get_num_params_model1_no_grad():
    model = Model1()
    model.requires_grad_(False)
    assert next(model.parameters(), None) is None

    num_params = get_num_params(model)
    assert num_params == (0, 0)


def test_get_num_params_model2():
    model = Model2()
    assert next(model.parameters(), None) is not None

    num_params = get_num_params(model)
    assert num_params == (2, 0)


def test_get_num_params_model2_no_grad():
    model = Model2()
    model.requires_grad_(False)
    assert next(model.parameters(), None) is not None

    num_params = get_num_params(model)
    assert num_params == (0, 2)


def test_get_num_params_model3():
    model = Model3()
    assert next(model.parameters(), None) is not None

    num_params = get_num_params(model)
    assert num_params == (6, 0)


def test_get_num_params_model3_no_grad():
    model = Model3()
    model.requires_grad_(False)
    assert next(model.parameters(), None) is not None

    num_params = get_num_params(model)
    assert num_params == (0, 6)


def test_get_num_params_model3_mix():
    model = Model3()
    model.fc1.requires_grad_(False)
    assert next(model.parameters(), None) is not None

    num_params = get_num_params(model)
    assert num_params == (4, 2)


def test_get_num_params_model4():
    model = Model4()
    assert next(model.parameters(), None) is not None

    num_params = get_num_params(model)
    assert num_params == (2, 0)


def test_get_num_params_model4_no_grad():
    model = Model4()
    model.requires_grad_(False)
    assert next(model.parameters(), None) is not None

    num_params = get_num_params(model)
    assert num_params == (0, 2)
