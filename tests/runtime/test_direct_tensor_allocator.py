#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import gc
import mmap
import sys

import pytest
import torch

import nnscaler.runtime.io.direct_tensor_allocator as direct_tensor_allocator
from nnscaler.runtime.io import DirectTensorAllocator


@pytest.fixture
def allocator():
    allocator = DirectTensorAllocator()
    if allocator.hugepage_bytes is None:
        pytest.skip("transparent huge pages are unavailable on this system")
    return allocator


def aligned(tensor, allocator):
    return tensor.data_ptr() % allocator.hugepage_bytes == 0


def test_intercepts_large_cpu_factories(allocator):
    n = allocator.hugepage_bytes // 4  # float32 elements per huge page
    with allocator:
        zeros = torch.zeros(2, n)
        empty = torch.empty(n, dtype=torch.float16)      # exactly half a huge page: native
        big_empty = torch.empty(n, dtype=torch.float64)  # exactly two huge pages
        strided = torch.empty_strided((n // 2, 2), (4, 1))
        small = torch.zeros(10)
    assert aligned(zeros, allocator) and torch.count_nonzero(zeros) == 0
    assert aligned(big_empty, allocator) and big_empty.dtype == torch.float64
    assert aligned(strided, allocator) and strided.stride() == (4, 1)
    # Only the three large tensors were served by the allocator (malloc may
    # happen to return an aligned address, so native tensors are not checked
    # by alignment). The strided storage covers (n // 2 - 1) * 4 + 2 elements.
    assert allocator.ram_stats["allocated_bytes"] == sum(
        -(-t.untyped_storage().nbytes() // mmap.PAGESIZE) * mmap.PAGESIZE
        for t in (zeros, big_empty, strided))
    assert allocator.ram_stats["live_bytes"] == allocator.ram_stats["peak_bytes"] == allocator.ram_stats["allocated_bytes"]


def test_passthrough(allocator):
    n = allocator.hugepage_bytes // 4
    with allocator:
        channels_last = torch.empty(1, 4, n // 64, 16, memory_format=torch.channels_last)
        ones = torch.ones(n)                 # not a factory op we intercept
        sparse = torch.zeros(n, layout=torch.sparse_coo)
        # Pinned memory needs an accelerator runtime, so both are CUDA-gated.
        cuda = torch.zeros(n, device="cuda") if torch.cuda.is_available() else None
        pinned = torch.zeros(n, pin_memory=True) if torch.cuda.is_available() else None
    assert pinned is None or pinned.is_pinned()
    assert channels_last.is_contiguous(memory_format=torch.channels_last)
    assert ones.sum() == n
    assert sparse.layout == torch.sparse_coo
    assert cuda is None or cuda.device.type == "cuda"
    assert allocator.ram_stats["allocated_bytes"] == 0


def test_inactive_without_thp(monkeypatch):
    monkeypatch.setattr(direct_tensor_allocator, "_detect_hugepage_bytes", lambda: None)
    allocator = DirectTensorAllocator()
    assert allocator.hugepage_bytes is None
    with allocator:
        tensor = torch.zeros(1 << 20)
    assert torch.count_nonzero(tensor) == 0
    assert allocator.ram_stats["allocated_bytes"] == 0


def test_release_tracks_views(allocator):
    with allocator:
        tensor = torch.zeros(allocator.hugepage_bytes, dtype=torch.uint8)
    view = tensor[::2]
    del tensor
    gc.collect()
    assert allocator.ram_stats["live_bytes"] == allocator.hugepage_bytes  # view keeps the mapping
    assert view[-1] == 0
    del view
    gc.collect()
    assert allocator.ram_stats["live_bytes"] == 0
    assert allocator.ram_stats["released_bytes"] == allocator.ram_stats["allocated_bytes"]


def test_save_and_load_roundtrip(allocator, tmp_path):
    n = allocator.hugepage_bytes // 4
    with allocator:
        tensor = torch.empty(n)
    tensor.copy_(torch.arange(n, dtype=torch.float32))
    torch.save({"w": tensor}, tmp_path / "t.pt")
    assert torch.equal(torch.load(tmp_path / "t.pt", weights_only=True)["w"], tensor)


@pytest.mark.parametrize(
    "enabled, pmd_size, expected",
    [
        ("always [madvise] never\n", "2097152\n", 2 << 20),
        ("[always] madvise never\n", "536870912\n", 512 << 20),
        ("always madvise [never]\n", "2097152\n", None),
        ("always [madvise] never\n", None, 2 << 20),  # pre-4.11 kernel: no hpage_pmd_size
        (None, None, None),                           # THP compiled out: no sysfs directory
    ],
)
def test_detect_hugepage_bytes(tmp_path, monkeypatch, enabled, pmd_size, expected):
    if enabled is not None:
        (tmp_path / "enabled").write_text(enabled)
    if pmd_size is not None:
        (tmp_path / "hpage_pmd_size").write_text(pmd_size)
    monkeypatch.setattr(direct_tensor_allocator, "_THP_SYSFS", str(tmp_path))
    monkeypatch.setattr(sys, "platform", "linux")
    assert direct_tensor_allocator._detect_hugepage_bytes() == expected


def test_detect_hugepage_bytes_non_linux(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    assert direct_tensor_allocator._detect_hugepage_bytes() is None
