import gc
import weakref

import pytest
import torch

import nnscaler


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_defer_release_does_not_keep_tensor_object_alive():
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        tensor = (torch.randn(8, device='cuda', requires_grad=True) * 2).relu()

    tensor_ref = weakref.ref(tensor)
    nnscaler.runtime.device.defer_release(tensor, stream=stream)
    del tensor
    gc.collect()

    try:
        assert tensor_ref() is None
    finally:
        nnscaler.runtime.device.flush_deferred_releases()
