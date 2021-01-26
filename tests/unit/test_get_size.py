import torch
from modelscope import get_size


def test_get_size_tensor():
    tensor = torch.randn(1, 2, 3, 4)
    size = get_size(tensor)
    expected = torch.Size([1, 2, 3, 4])
    assert size == expected

    tensor = torch.randn(1)
    size = get_size(tensor)
    expected = torch.Size([1])
    assert size == expected

    tensor = torch.randn([])
    size = get_size(tensor)
    expected = torch.Size([])
    assert size == expected

    tensor = torch.tensor(0)
    size = get_size(tensor)
    expected = torch.Size([])
    assert size == expected

    tensor = torch.tensor([])
    size = get_size(tensor)
    expected = torch.Size([0])
    assert size == expected


def test_get_size_tensors():
    tensors = (torch.randn(1, 2, 3, 4), torch.randn(4, 3, 2, 1))
    size = get_size(tensors)
    expected = [torch.Size([1, 2, 3, 4]), torch.Size([4, 3, 2, 1])]
    assert size == expected

    tensors = (
        (torch.randn(1, 2, 3, 4), torch.randn(5, 6, 7, 8)),
        (torch.randn(9, 10, 11, 12), torch.randn(13, 14, 15, 16)),
    )
    size = get_size(tensors)
    expected = [
        [torch.Size([1, 2, 3, 4]), torch.Size([5, 6, 7, 8])],
        [torch.Size([9, 10, 11, 12]), torch.Size([13, 14, 15, 16])],
    ]
    assert size == expected


def test_get_size_tensor_no_attr():
    tensor = None
    size = get_size(tensor)
    expected = None
    assert size == expected


def test_get_size_tensors_no_attr():
    tensors = (None, None)
    size = get_size(tensors)
    expected = [None, None]
    assert size == expected

    tensors = (1, (2, 3, (4, 5, (6,))))
    size = get_size(tensors)
    expected = [None, [None, None, [None, None, [None]]]]
    assert size == expected


def test_get_size_tensors_mix():
    tensors = (1, torch.tensor([1]), (((((1,),),),),), torch.tensor(1))
    size = get_size(tensors)
    expected = [None, torch.Size([1]), [[[[[None]]]]], torch.Size([])]
    assert size == expected
