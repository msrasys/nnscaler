from cube.tensor.indices import BaseIndices, TileIndices

import torch

def test_base_indices():

    tensor = torch.randn((10, 10, 10))
    
    # test init
    sparse_indices = (
        [2,3,1,4],
        [0,4,8,4],
        [7,5,9,4]
    )
    indices = BaseIndices(sparse_indices)
    assert indices.indices == sparse_indices

    # test ndim
    assert indices.ndim() == 3

    # test size
    assert indices.size() == 4

    # test get
    sub_tensor = tensor[indices.get()]
    assert torch.allclose(sub_tensor, tensor[sparse_indices]) is True

    # test reorder
    arg_order = [2, 1, 0, 3]
    indices.reorder(arg_order)
    sub_tensor = tensor[indices.get()]

    sparse_indices = (
        [1,3,2,4],
        [8,4,0,4],
        [9,5,7,4]
    )
    ref_tensor = tensor[sparse_indices]
    assert torch.allclose(sub_tensor, ref_tensor) is True


if __name__ == '__main__':
    test_base_indices()
