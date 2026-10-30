#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import gc
import sys

import pytest
import torch

import nnscaler.runtime.io.mapped_tensor_deallocator as mapped_tensor_deallocator
from nnscaler.runtime.io import MappedTensorDeallocator

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="madvise eviction is Linux only")

NUMEL = 4 << 20  # 16 MiB per float32 tensor


def vma_rss_bytes(address):
    """Resident bytes of the mapping containing address, from /proc/self/smaps."""
    with open("/proc/self/smaps") as f:
        lines = f.read().splitlines()
    for i, line in enumerate(lines):
        head = line.split()[0]
        if "-" not in head or ":" in head:
            continue
        start, end = (int(x, 16) for x in head.split("-"))
        if start <= address < end:
            return next(int(l.split()[1]) for l in lines[i + 1:i + 30] if l.startswith("Rss:")) * 1024
    raise AssertionError("address not mapped")


@pytest.fixture
def shard_path(tmp_path):
    if torch.__version__ < (2, 1):
        pytest.skip("torch.load(mmap=True) requires PyTorch 2.1")
    path = tmp_path / "shard.pt"
    torch.save({"a": torch.arange(NUMEL, dtype=torch.float32), "b": torch.ones(NUMEL)}, path)
    return path


@pytest.fixture
def shard(shard_path):
    return torch.load(shard_path, mmap=True, weights_only=True)


def test_evict_drops_pages_and_keeps_data(shard):
    deallocator = MappedTensorDeallocator()
    deallocator.register(shard)
    a, b = shard["a"], shard["b"]
    nbytes = NUMEL * 4
    # The mapping also holds the zip/pickle metadata pages torch.load touched,
    # and faults map a few neighbouring pages (fault-around), so compare against
    # the RSS before the tensors were read and allow 128 KiB of slack.
    slack = 128 * 1024
    base = vma_rss_bytes(a.data_ptr())
    a.sum(); b.sum()
    loaded = vma_rss_bytes(a.data_ptr())
    assert loaded >= base + 2 * nbytes

    deallocator.evict(a)
    assert vma_rss_bytes(a.data_ptr()) <= loaded - nbytes + slack
    deallocator.evict(a)                                          # idempotent
    deallocator.evict(b)
    assert vma_rss_bytes(a.data_ptr()) <= base + slack
    assert a[-1] == NUMEL - 1 and b[0] == 1                       # refaulted from the file

    a.sum()
    assert vma_rss_bytes(a.data_ptr()) >= base + nbytes
    deallocator.evict(a[::3])                                     # strided view of a registered storage
    assert vma_rss_bytes(a.data_ptr()) <= base + slack


def test_register_skips_non_evictable(shard):
    anon = torch.ones(NUMEL)
    shared = torch.ones(1024).share_memory_()
    mixed = {
        "mmap": shard["a"], "anon": anon, "shared": shared, "empty": torch.empty(0),
        "sparse": torch.zeros(4, layout=torch.sparse_coo),
        "cuda": torch.ones(4, device="cuda") if torch.cuda.is_available() else torch.ones(4),
        "nested": [shard["b"][::2], (anon[:10],)],
    }
    deallocator = MappedTensorDeallocator()
    deallocator.register(mixed)
    registered = set(deallocator._storages)
    assert registered == {shard["a"].untyped_storage().data_ptr(), shard["b"].untyped_storage().data_ptr()}

    # Unregistered tensors are left untouched; evicting them would zero their memory.
    deallocator.evict(anon)
    deallocator.evict(shared)
    deallocator.evict(mixed["sparse"])        # no untyped_storage(): must not raise
    deallocator.evict(mixed["cuda"])
    assert anon.sum() == NUMEL and shared.sum() == 1024


def test_register_fails_closed_without_maps(shard, monkeypatch):
    monkeypatch.setattr(mapped_tensor_deallocator, "_file_backed_ranges", lambda: None)
    deallocator = MappedTensorDeallocator()
    deallocator.register(shard)
    assert not deallocator._storages


def test_evict_ignores_dead_storage(shard_path):
    # Load here rather than via the fixture, which would keep the dict alive.
    shard = torch.load(shard_path, mmap=True, weights_only=True)
    deallocator = MappedTensorDeallocator()
    deallocator.register(shard)
    address = shard["a"].untyped_storage().data_ptr()
    del shard
    gc.collect()
    assert deallocator._storages[address].expired()
    deallocator.evict(torch.ones(NUMEL))     # different storage, no error, no effect


def test_noop_off_linux(monkeypatch, shard):
    monkeypatch.setattr(sys, "platform", "darwin")
    deallocator = MappedTensorDeallocator()
    assert deallocator._libc is None
    deallocator.register(shard)
    deallocator.evict(shard["a"])             # must not touch libc


def test_is_file_backed():
    ranges = [(0x1000, 0x3000), (0x5000, 0x6000)]
    is_file_backed = mapped_tensor_deallocator._is_file_backed
    assert is_file_backed(ranges, 0x1000) and is_file_backed(ranges, 0x2fff) and is_file_backed(ranges, 0x5000)
    assert not is_file_backed(ranges, 0x0fff) and not is_file_backed(ranges, 0x3000) and not is_file_backed(ranges, 0x4000)
    assert not is_file_backed(ranges, 0x6000) and not is_file_backed([], 0x1000)
