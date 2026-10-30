#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import errno
from concurrent.futures import ThreadPoolExecutor
import gc
import io
import mmap
import os
from unittest.mock import Mock

import pytest
import torch

import nnscaler.runtime.io.direct_io_writer as direct_io_writer
from nnscaler.runtime.io.direct_io_writer import DirectIOWriter


@pytest.fixture
def direct_io_path(tmp_path):
    if not hasattr(os, "O_DIRECT") or not hasattr(os, "pwrite"):
        pytest.skip("O_DIRECT and pwrite are required")
    path = tmp_path / "output.bin"
    fd = None
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_DIRECT, 0o600)
        with mmap.mmap(-1, mmap.PAGESIZE) as buffer:
            assert os.pwrite(fd, buffer, 0) == mmap.PAGESIZE
    except OSError as exc:
        if exc.errno in (errno.EINVAL, errno.EOPNOTSUPP):
            pytest.skip(f"Test filesystem does not support page-aligned O_DIRECT: {exc}")
        raise
    finally:
        if fd is not None:
            os.close(fd)
            path.unlink()
    return path


@pytest.mark.parametrize("size", [0, mmap.PAGESIZE - 1, mmap.PAGESIZE, 9 * mmap.PAGESIZE + 123])
def test_write(direct_io_path, size):
    data = os.urandom(size)
    with DirectIOWriter(direct_io_path, chunk_bytes=mmap.PAGESIZE, workers=2) as writer:
        assert writer.tell() == 0
        assert writer.write(b"") == 0
        assert direct_io_path.stat().st_size == 0
        with pytest.raises(io.UnsupportedOperation):
            writer.fileno()
        for start in range(0, size, 997):
            piece = memoryview(data)[start:start + 997]
            assert writer.write(piece) == len(piece)
            assert writer.tell() == start + len(piece)

    assert direct_io_path.read_bytes() == data
    assert writer.closed
    writer.close()
    with pytest.raises(ValueError, match="closed"):
        writer.write(b"x")
    with pytest.raises(ValueError, match="closed"):
        writer.flush()


def test_flush_then_write(direct_io_path):
    prefix = os.urandom(mmap.PAGESIZE + 123)
    suffix = os.urandom(8 * mmap.PAGESIZE + 17)
    with DirectIOWriter(direct_io_path, chunk_bytes=2 * mmap.PAGESIZE, workers=2) as writer:
        writer.flush()
        assert direct_io_path.read_bytes() == b""
        writer.write(prefix)
        writer.flush()
        assert direct_io_path.read_bytes() == prefix
        writer.flush()
        assert direct_io_path.read_bytes() == prefix
        writer.write(suffix)
        assert writer.tell() == len(prefix) + len(suffix)

    assert direct_io_path.read_bytes() == prefix + suffix


# The legacy format writes storages straight to f.fileno() when available,
# bypassing the bounce buffer; it must fall back to write() here.
@pytest.mark.parametrize("zipfile", [True, False], ids=["zip", "legacy"])
def test_torch_save(direct_io_path, zipfile):
    tensor = torch.arange(128 * 128, dtype=torch.float32).reshape(128, 128)
    state = {"weight": tensor, "shared": tensor, "step": 7}
    with DirectIOWriter(direct_io_path, chunk_bytes=mmap.PAGESIZE, workers=2) as writer:
        torch.save(state, writer, _use_new_zipfile_serialization=zipfile)
        assert writer.tell() == direct_io_path.stat().st_size

    loaded = torch.load(direct_io_path, weights_only=True)
    assert loaded["step"] == state["step"]
    assert torch.equal(loaded["weight"], tensor)
    assert loaded["weight"].data_ptr() == loaded["shared"].data_ptr()


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"chunk_bytes": 0}, "chunk_bytes"),
        ({"chunk_bytes": mmap.PAGESIZE + 1}, "chunk_bytes"),
        ({"workers": 0}, "workers"),
    ],
)
def test_invalid_arguments(tmp_path, kwargs, message):
    path = tmp_path / "invalid.bin"
    with pytest.raises(ValueError, match=message):
        DirectIOWriter(path, **kwargs)
    assert not path.exists()


def test_existing_file(tmp_path):
    path = tmp_path / "existing.bin"
    path.write_bytes(b"existing checkpoint")
    with pytest.raises(FileExistsError):
        DirectIOWriter(path, chunk_bytes=mmap.PAGESIZE)
    assert path.read_bytes() == b"existing checkpoint"


@pytest.mark.parametrize("failure", ["pool", "first_buffer", "second_buffer"])
def test_initialization_failure_releases_resources(direct_io_path, monkeypatch, failure):
    real_mmap = mmap.mmap
    buffers = []
    shutdown = Mock()
    close = Mock(wraps=os.close)
    error = OSError(errno.ENOMEM, "injected initialization failure")

    def create_pool(**kwargs):
        if failure == "pool":
            raise error
        pool = ThreadPoolExecutor(**kwargs)
        shutdown.side_effect = pool.shutdown
        monkeypatch.setattr(pool, "shutdown", shutdown)
        return pool

    def allocate_buffer(*args, **kwargs):
        if failure == "first_buffer" or buffers:
            raise error
        buffer = real_mmap(*args, **kwargs)
        buffers.append(buffer)
        return buffer

    monkeypatch.setattr(direct_io_writer, "ThreadPoolExecutor", create_pool)
    monkeypatch.setattr(mmap, "mmap", allocate_buffer)
    monkeypatch.setattr(os, "close", close)
    with pytest.raises(OSError, match="injected initialization failure") as caught:
        DirectIOWriter(direct_io_path, chunk_bytes=mmap.PAGESIZE, workers=1)

    assert caught.value is error
    assert not direct_io_path.exists()      # created by os.open, removed by the rollback
    close.assert_called_once()
    fd = close.call_args[0][0]
    with pytest.raises(OSError) as caught:
        os.fstat(fd)
    assert caught.value.errno == errno.EBADF
    assert all(buffer.closed for buffer in buffers)
    if failure == "pool":
        shutdown.assert_not_called()
    else:
        shutdown.assert_called_once_with(wait=True)


@pytest.mark.parametrize("delete_on_error", [True, False])
def test_exception_in_with_block(direct_io_path, delete_on_error):
    with pytest.raises(RuntimeError, match="boom"):
        with DirectIOWriter(direct_io_path, chunk_bytes=mmap.PAGESIZE, workers=1,
                            delete_on_error=delete_on_error) as writer:
            writer.write(b"x" * (mmap.PAGESIZE + 1))
            raise RuntimeError("boom")
    assert writer.closed
    assert direct_io_path.exists() == (not delete_on_error)
    if not delete_on_error:
        # Unflushed tail is lost, but completed full blocks remain for inspection.
        assert direct_io_path.read_bytes() == b"x" * mmap.PAGESIZE


def test_del_releases_resources(direct_io_path):
    writer = DirectIOWriter(direct_io_path, chunk_bytes=mmap.PAGESIZE, workers=1)
    writer.write(b"x" * 10)
    buffers = tuple(writer._buffers)
    fd = writer._fd
    del writer
    gc.collect()
    assert all(buffer.closed for buffer in buffers)
    with pytest.raises(OSError) as error:
        os.fstat(fd)
    assert error.value.errno == errno.EBADF
    # Finalization does not flush; the partial file is left as-is.
    assert direct_io_path.read_bytes() == b""


@pytest.mark.parametrize("size", [123, mmap.PAGESIZE], ids=["tail", "async"])
def test_write_error_releases_resources(direct_io_path, monkeypatch, size):
    def fail_write(fd, data, offset):
        raise OSError(errno.EIO, "injected write failure")

    writer = DirectIOWriter(direct_io_path, chunk_bytes=mmap.PAGESIZE, workers=1)
    buffers = tuple(writer._buffers)
    fd = writer._fd
    monkeypatch.setattr(os, "pwrite", fail_write)
    with pytest.raises(OSError, match="injected write failure") as error:
        with writer:
            writer.write(b"x" * size)
            for _ in range(2):
                with pytest.raises(OSError, match="injected write failure") as flush_error:
                    writer.flush()
                assert flush_error.value.errno == errno.EIO

    assert error.value.errno == errno.EIO
    assert writer.closed
    assert not direct_io_path.exists()
    assert all(buffer.closed for buffer in buffers)
    with pytest.raises(OSError) as error:
        os.fstat(fd)
    assert error.value.errno == errno.EBADF
    writer.close()
    with pytest.raises(ValueError, match="closed"):
        writer.flush()
