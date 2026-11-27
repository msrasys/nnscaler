import ctypes
import mmap
import os
import sys

import pytest
import torch

from nnscaler.runtime.module import AttrMeta, CubeModule
from nnscaler.runtime.page_release import FilePageReleaser

pytestmark = pytest.mark.skipif(not sys.platform.startswith('linux'), reason='Linux only')


def _cached_pages(path):
    """Number of the file's pages in the page cache (via mincore)."""
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
                          ctypes.c_int, ctypes.c_long)
    size = os.path.getsize(path)
    pages = (size + mmap.PAGESIZE - 1) // mmap.PAGESIZE
    vec = (ctypes.c_ubyte * pages)()
    fd = os.open(path, os.O_RDONLY)
    try:
        addr = libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0)
        assert libc.mincore(ctypes.c_void_p(addr), ctypes.c_size_t(size), vec) == 0
        libc.munmap(ctypes.c_void_p(addr), ctypes.c_size_t(size))
    finally:
        os.close(fd)
    return sum(v & 1 for v in vec)


def _save_cold(tmp_path, name, obj):
    path = str(tmp_path / name)
    torch.save(obj, path)
    fd = os.open(path, os.O_RDONLY)
    os.fsync(fd)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)
    return path


def _require_page_cache(path):
    torch.load(path, mmap=True, weights_only=False)['x'].sum()
    if _cached_pages(path) == 0:
        pytest.skip('filesystem does not keep file pages in the page cache')


def test_release_drops_consumed_file_pages(tmp_path):
    expected = torch.randn(4 << 20)  # 16 MiB
    path = _save_cold(tmp_path, 'shard.pt', {'x': expected})
    _require_page_cache(path)
    fd = os.open(path, os.O_RDONLY)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)

    x = torch.load(path, mmap=True, weights_only=False)['x']
    copy = x.clone()  # consume: faults every page in
    releaser = FilePageReleaser()
    assert releaser.enabled
    before = _cached_pages(path)
    releaser.release(x)
    assert releaser.close() > 0
    assert _cached_pages(path) < before // 4
    # Dropped clean pages fault back in from the file unchanged.
    assert torch.equal(x, expected) and torch.equal(copy, expected)


def test_release_only_whole_pages_inside_tensor(tmp_path):
    x = torch.load(_save_cold(tmp_path, 'small.pt', {'x': torch.randn(16)}),
                   mmap=True, weights_only=False)['x']
    releaser = FilePageReleaser()
    releaser.release(x)  # 64 bytes: no whole page inside
    assert releaser.close() == 0


def test_release_ignores_anonymous_memory():
    x = torch.randn(4 << 20)
    releaser = FilePageReleaser()
    releaser.release(x)
    assert releaser.close() == 0
    assert torch.isfinite(x).all()


def test_release_can_be_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv('NNSCALER_MERGE_RELEASE_PAGES', '0')
    x = torch.load(_save_cold(tmp_path, 'shard.pt', {'x': torch.randn(4 << 20)}),
                   mmap=True, weights_only=False)['x']
    releaser = FilePageReleaser()
    assert not releaser.enabled
    releaser.release(x)
    assert releaser.close() == 0


def test_merge_mmap_shards_releases_pages_and_stays_exact(tmp_path):
    # ZeRO-1 expert layout: EP2 x Z2, each shard's state split over its ZeRO group.
    ep_size, z_size, shape = 2, 2, (4, 256, 512)
    sub = (shape[0] // ep_size,) + shape[1:]
    numel = sub[0] * shape[1] * shape[2]
    full = {k: torch.randn(shape) for k in ('exp_avg', 'exp_avg_sq')}
    weight = torch.randn(shape)
    fullmaps, zero_maps, paths = [], [], []
    for rank in range(ep_size * z_size):
        ep, z = divmod(rank, z_size)
        slicers = (slice(ep * sub[0], (ep + 1) * sub[0]), slice(0, shape[1]), slice(0, shape[2]))
        fullmaps.append({'w_1': AttrMeta(1, True, 'w', shape, slicers, 1, torch.float32, sub)})
        zero_maps.append(({0: (0, 0, numel, sub)},
                          {0: (list(range(ep * z_size, ep * z_size + z_size)), numel)}, 1))
        chunk = numel // z_size
        state = {k: v[slicers].reshape(-1)[z * chunk:(z + 1) * chunk].clone() for k, v in full.items()}
        state['step'] = torch.tensor(3.0)
        paths.append(_save_cold(tmp_path, f'{rank}.pt', {
            'x': weight[slicers].clone(),
            'optim': {'state': {0: state}, 'param_groups': [{'params': [0]}]}}))
    _require_page_cache(paths[0])
    for path in paths:
        fd = os.open(path, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)

    shards = [torch.load(p, mmap=True, weights_only=False) for p in paths]
    model = CubeModule.merge_model_state_dicts([{'w_1': s['x']} for s in shards], fullmaps)
    optim = CubeModule.merge_opt_state_dicts(fullmaps, [s['optim'] for s in shards], zero_maps)

    assert torch.equal(model['w'], weight)
    for key, value in full.items():
        assert torch.equal(optim['state'][0][key], value)
    assert optim['state'][0]['step'].item() == 3.0
    total = sum(os.path.getsize(p) for p in paths) // mmap.PAGESIZE
    assert sum(_cached_pages(p) for p in paths) < total // 4
