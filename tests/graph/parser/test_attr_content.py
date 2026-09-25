#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event, Lock

import pytest
import torch

from nnscaler.flags import CompileFlag
from nnscaler.graph.parser import FxModuleParser
from nnscaler.graph.parser import frame as frame_module
from nnscaler.graph.parser.frame import Frame
from nnscaler.ir.tensor import IRFullTensor
from nnscaler.runtime.module import AttrMeta, CubeModule


@pytest.mark.parametrize('max_workers', [1, 2, 4, None])
def test_save_attr_content_index(tmp_path: Path, max_workers):
    frame = Frame()
    tensors = [IRFullTensor((4,), name=f'w{idx}') for idx in range(3)]
    values = [torch.arange(4) + idx * 4 for idx in range(3)]
    for idx, (tensor, value) in enumerate(zip(tensors, values)):
        frame.add_attr(tensor, value, f'w{idx}')

    file_stem = tmp_path / FxModuleParser.ATTR_CONTENT_FILE_STEM
    frame.save_attr_content(file_stem, params_per_file=5, max_workers=max_workers)

    tid_to_chunk = torch.load(
        tmp_path / FxModuleParser.ATTR_CONTENT_INDEX_FILE,
        weights_only=True,
    )
    assert tid_to_chunk == {tensor.tid: idx for idx, tensor in enumerate(tensors)}
    for idx, (tensor, value) in enumerate(zip(tensors, values)):
        assert torch.equal(torch.load(f'{file_stem}.{idx}', weights_only=True, mmap=True)[tensor.tid], value)
    assert {path.name for path in tmp_path.iterdir()} == {
        'fullmodel.pt.0', 'fullmodel.pt.1', 'fullmodel.pt.2', 'fullmodel.pt.index',
    }


@pytest.mark.parametrize('num_tensors', [0, 1])
def test_save_attr_content_single_chunk(tmp_path: Path, num_tensors):
    frame = Frame()
    expected = {}
    if num_tensors:
        tensor = IRFullTensor((4,), name='weight')
        value = torch.arange(4)
        frame.add_attr(tensor, value, 'weight')
        expected[tensor.tid] = value

    file_stem = tmp_path / FxModuleParser.ATTR_CONTENT_FILE_STEM
    frame.save_attr_content(file_stem, max_workers=4)

    assert torch.load(f'{file_stem}.index', weights_only=True) == {tid: 0 for tid in expected}
    saved = torch.load(f'{file_stem}.0', weights_only=True, mmap=True)
    assert saved.keys() == expected.keys()
    for tid, value in expected.items():
        assert torch.equal(saved[tid], value)
    assert {path.name for path in tmp_path.iterdir()} == {'fullmodel.pt.0', 'fullmodel.pt.index'}


@pytest.mark.parametrize('max_workers,use_flag', [(1, False), (2, False), (4, True)])
def test_save_attr_content_bounded_concurrency(tmp_path: Path, monkeypatch, max_workers, use_flag):
    frame = Frame()
    expected = {}
    for idx in range(12):
        tensor = IRFullTensor((4,), name=f'w{idx}')
        value = torch.arange(4) + idx * 4
        frame.add_attr(tensor, value, f'w{idx}')
        expected[tensor.tid] = value

    file_stem = tmp_path / FxModuleParser.ATTR_CONTENT_FILE_STEM
    index_path = Path(f'{file_stem}.index')
    barrier = Barrier(max_workers, timeout=10)
    lock = Lock()
    active = peak_active = 0
    completed = set()
    submitted = []
    torch_save = torch.save

    class RecordingExecutor(ThreadPoolExecutor):
        def submit(self, *args, **kwargs):
            # Include queued tasks, not just the threads actively writing.
            assert sum(not future.done() for future in submitted) < max_workers
            future = super().submit(*args, **kwargs)
            submitted.append(future)
            return future

    def record_save(content, filename):
        nonlocal active, peak_active
        assert not index_path.exists()
        if all(isinstance(value, int) for value in content.values()):
            assert completed == expected.keys()
            assert active == 0
            return torch_save(content, filename)

        with lock:
            active += 1
            peak_active = max(peak_active, active)
        try:
            # A barrier proves overlap without relying on timing or large files.
            barrier.wait()
            for tid, value in content.items():
                assert value is expected[tid]  # CPU storage is shared, not copied.
            torch_save(content, filename)
            with lock:
                completed.update(content)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(frame_module, 'ThreadPoolExecutor', RecordingExecutor)
    monkeypatch.setattr(torch, 'save', record_save)
    monkeypatch.setattr(CompileFlag, 'attr_save_workers', max_workers if use_flag else 1)
    frame.save_attr_content(file_stem, params_per_file=4, max_workers=None if use_flag else max_workers)

    assert peak_active == max_workers
    assert active == 0
    assert len(submitted) == (12 if max_workers > 1 else 0)
    assert torch.load(index_path, weights_only=True) == {tid: idx for idx, tid in enumerate(expected)}


@pytest.mark.parametrize('existing_index', [False, True])
def test_save_attr_content_failure_joins_writers(tmp_path: Path, monkeypatch, existing_index):
    frame = Frame()
    tensors = [IRFullTensor((4,), name=f'w{idx}') for idx in range(6)]
    for idx, tensor in enumerate(tensors):
        frame.add_attr(tensor, torch.arange(4), f'w{idx}')

    file_stem = tmp_path / FxModuleParser.ATTR_CONTENT_FILE_STEM
    index_path = Path(f'{file_stem}.index')
    old_chunk = Path(f'{file_stem}.0')
    if existing_index:
        torch.save({tensors[0].tid: 0}, index_path)
        old_chunk.write_bytes(b'previous shard')

    barrier = Barrier(2, timeout=10)
    failed = Event()
    release_writer = Event()
    writer_finished = Event()
    torch_save = torch.save

    class ObservedExecutor(ThreadPoolExecutor):
        def submit(self, *args, **kwargs):
            future = super().submit(*args, **kwargs)
            # Release the other writer only after the failed future is observable.
            future.add_done_callback(lambda done: failed.set() if done.exception() is not None else None)
            return future

    def fail_save(content, filename):
        assert not index_path.exists()
        if tensors[0].tid in content:
            barrier.wait()
            Path(filename).write_bytes(b'partial shard')
            raise OSError('shard write failed')
        assert set(content) == {tensors[1].tid}, 'Submitted more shards after failure'
        barrier.wait()
        try:
            assert release_writer.wait(timeout=10)
            torch_save(content, filename)
        finally:
            writer_finished.set()

    monkeypatch.setattr(frame_module, 'ThreadPoolExecutor', ObservedExecutor)
    monkeypatch.setattr(torch, 'save', fail_save)
    with ThreadPoolExecutor(max_workers=1) as executor:
        save = executor.submit(frame.save_attr_content, file_stem, 4, max_workers=2)
        try:
            assert failed.wait(timeout=10)
            assert not save.done(), 'Returned while a writer was still active'
            assert not index_path.exists()
        finally:
            release_writer.set()
        with pytest.raises(OSError, match='shard write failed'):
            save.result(timeout=10)

    assert writer_finished.is_set()
    assert not index_path.exists()
    assert {path.name for path in tmp_path.iterdir()} == (
        {'fullmodel.pt.0', 'fullmodel.pt.1'} if existing_index else {'fullmodel.pt.1'}
    )
    if existing_index:
        assert old_chunk.read_bytes() == b'previous shard'
    assert torch.equal(torch.load(f'{file_stem}.1', weights_only=True)[tensors[1].tid], torch.arange(4))


def test_save_attr_content_index_failure_is_atomic(tmp_path: Path, monkeypatch):
    frame = Frame()
    tensor = IRFullTensor((4,), name='weight')
    frame.add_attr(tensor, torch.arange(4), 'weight')
    file_stem = tmp_path / FxModuleParser.ATTR_CONTENT_FILE_STEM
    torch_save = torch.save

    def fail_index(content, filename):
        if isinstance(content[tensor.tid], int):
            Path(filename).write_bytes(b'partial index')
            raise OSError('index write failed')
        torch_save(content, filename)

    monkeypatch.setattr(torch, 'save', fail_index)
    with pytest.raises(OSError, match='index write failed'):
        frame.save_attr_content(file_stem, max_workers=1)
    assert {path.name for path in tmp_path.iterdir()} == {'fullmodel.pt.0'}
    assert torch.equal(torch.load(f'{file_stem}.0', weights_only=True)[tensor.tid], torch.arange(4))


@pytest.mark.parametrize('max_workers', [0, -1, True, 1.5, '4'])
@pytest.mark.parametrize('use_flag', [False, True])
def test_save_attr_content_invalid_workers(tmp_path: Path, monkeypatch, max_workers, use_flag):
    if use_flag:
        monkeypatch.setattr(CompileFlag, 'attr_save_workers', max_workers)
    with pytest.raises(ValueError, match='must be a positive integer'):
        Frame().save_attr_content(tmp_path / 'fullmodel.pt', max_workers=None if use_flag else max_workers)
    assert list(tmp_path.iterdir()) == []


def test_load_attr_content_only_reads_required_chunks(tmp_path: Path, monkeypatch):
    file_stem = tmp_path / FxModuleParser.ATTR_CONTENT_FILE_STEM
    torch.save({10: torch.full((4,), 10.0)}, f'{file_stem}.0')
    # This chunk must never be opened by the loader.
    Path(f'{file_stem}.1').write_bytes(b'not a torch checkpoint')
    torch.save({30: torch.arange(8, dtype=torch.float32)}, f'{file_stem}.2')
    torch.save({10: 0, 20: 1, 30: 2}, tmp_path / FxModuleParser.ATTR_CONTENT_INDEX_FILE)

    module = CubeModule()
    module.register_parameter('local_weight', torch.nn.Parameter(torch.empty(3)))
    module._fullmap['local_weight'] = AttrMeta(
        tid=30,
        is_param=True,
        orig_name='weight',
        shape=(8,),
        slicers=(slice(2, 5),),
        val_chunks=1,
        dtype=torch.float32,
        sub_shape=(3,),
    )

    load_calls = []
    torch_load = torch.load

    def record_load(filename, *args, **kwargs):
        load_calls.append((Path(filename).name, kwargs.copy()))
        return torch_load(filename, *args, **kwargs)

    monkeypatch.setattr(torch, 'load', record_load)
    module.load_attr_content(str(file_stem))

    assert torch.equal(module.local_weight, torch.arange(2, 5, dtype=torch.float32))
    assert load_calls == [
        (FxModuleParser.ATTR_CONTENT_INDEX_FILE, {'weights_only': True}),
        ('fullmodel.pt.2', {'mmap': True, 'weights_only': True}),
    ]
