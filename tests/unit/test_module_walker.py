import torch.nn as nn

from modelscope import module_walker


class Model1(nn.Module):
    def __init__(self):
        super().__init__()


class Model2(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1, 1)


class Model3(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 1)
        self.fc2 = nn.Linear(1, 2)


class Model4(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1, 1)
        self.seq = nn.Sequential(self.fc, self.fc)


def test_model1_w_parents():
    model = ("model1", Model1())
    expected = (model,)
    outputs = tuple(module_walker(model, parents=True))
    assert expected == outputs


def test_model2_w_parents():
    model = ("model2", Model2())
    expected = (model, ("fc", model[-1].fc))
    outputs = tuple(module_walker(model, parents=True))
    assert expected == outputs


def test_model3_w_parents():
    model = ("model3", Model3())
    expected = (model, ("fc1", model[-1].fc1), ("fc2", model[-1].fc2))
    outputs = tuple(module_walker(model, parents=True))
    assert expected == outputs


def test_model4_w_parents():
    model = ("model4", Model4())
    expected = (
        model,
        ("fc", model[-1].fc),
        ("seq", model[-1].seq),
        ("0", model[-1].fc),
    )
    outputs = tuple(module_walker(model, parents=True))
    assert expected == outputs


def test_model1_children():
    model = ("model1", Model1())
    expected = (model, )
    outputs = tuple(module_walker(model, parents=False))
    assert expected == outputs


def test_model2_children():
    model = ("model2", Model2())
    expected = (("fc", model[-1].fc),)
    outputs = tuple(module_walker(model, parents=False))
    assert expected == outputs


def test_model3_children():
    model = ("model3", Model3())
    expected = (("fc1", model[-1].fc1), ("fc2", model[-1].fc2))
    outputs = tuple(module_walker(model, parents=False))
    assert expected == outputs


def test_model4_children():
    model = ("model4", Model4())
    expected = (("fc", model[-1].fc), ("0", model[-1].fc))
    outputs = tuple(module_walker(model, parents=False))
    assert expected == outputs
