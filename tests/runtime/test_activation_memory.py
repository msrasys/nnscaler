"""Best-fit pool behavior, including buffers still owned by transfers."""
import pytest
import torch

from nnscaler.runtime import activation_memory as memory
from nnscaler.runtime.activation_config import CPUOffloadConfig


def test_capacity_buckets_preserve_best_fit_and_pending_ownership(monkeypatch):
    empty = torch.empty
    monkeypatch.setattr(memory.torch, 'empty', lambda *args, **kwargs:
                        empty(*args, **dict(kwargs, pin_memory=False)))
    monkeypatch.setattr(memory, 'available_host_bytes', lambda: 1024**3)
    pool = memory._OffloadPool(CPUOffloadConfig(max_cpu_bytes=11264, host_reserve_bytes=0))
    small, medium, large = (pool.acquire(size) for size in (700, 1500, 5000))
    assert [buffer.numel() for buffer in (small, medium, large)] == [1024, 2048, 8192]

    class Pending:
        ready = False
        def query(self):
            return self.ready

    event = Pending()
    pool.release(small, event)
    pool.release(large, None)
    pool.release(medium, None)
    assert pool.acquire(1500) is medium  # Smallest sufficient completed block.
    assert pool.acquire(3000) is large
    assert pool.acquire(700) is None  # The remaining small block is still in use.
    event.ready = True
    assert pool.acquire(700) is small
    assert pool.snapshot()['allocated_pinned_bytes'] == 11264
    with pytest.raises(RuntimeError, match='live'):
        pool.close()
    for buffer in (large, small, medium):
        pool.release(buffer, None)
    pool.close()
    assert pool.snapshot()['allocated_pinned_bytes'] == 0
