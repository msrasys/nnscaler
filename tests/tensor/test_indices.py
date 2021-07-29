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


def test_tile_indices():

    tensor = torch.randn((10, 10, 10))

    anchor = [3,4,5]
    ofst = [2,4,3]
    indices = TileIndices(anchor, ofst)
    assert indices.anchor == anchor
    assert indices.shape == ofst
    assert indices.elenum == 2 * 4 * 3

    # test ndim
    assert indices.ndim() == 3
    
    # test size
    assert indices.size() == 2 * 4 * 3

    # test get
    sub_tensor = tensor[indices.get()]
    assert sub_tensor.size() == torch.Size(ofst)
    ref_tensor = tensor[(slice(3,3+2), slice(4,4+4), slice(5,5+3))]
    assert torch.allclose(sub_tensor, ref_tensor) is True

    # test reorder
    ##TODO


if __name__ == '__main__':
    test_base_indices()
    test_tile_indices()