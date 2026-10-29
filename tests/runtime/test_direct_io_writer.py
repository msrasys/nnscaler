#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import errno
import mmap
import os

import pytest
import torch

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
        assert os.fstat(writer.fileno()).st_size == 0
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


def test_torch_save(direct_io_path):
    tensor = torch.arange(128 * 128, dtype=torch.float32).reshape(128, 128)
    state = {"weight": tensor, "shared": tensor, "step": 7}
    with DirectIOWriter(direct_io_path, chunk_bytes=mmap.PAGESIZE, workers=2) as writer:
        torch.save(state, writer)
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


@pytest.mark.parametrize("size", [123, mmap.PAGESIZE], ids=["tail", "async"])
def test_write_error_releases_resources(direct_io_path, monkeypatch, size):
    def fail_write(fd, data, offset):
        raise OSError(errno.EIO, "injected write failure")

    writer = DirectIOWriter(direct_io_path, chunk_bytes=mmap.PAGESIZE, workers=1)
    buffers = tuple(writer._buffers)
    fd = writer.fileno()
    monkeypatch.setattr(os, "pwrite", fail_write)
    with pytest.raises(OSError, match="injected write failure") as error:
        with writer:
            writer.write(b"x" * size)

    assert error.value.errno == errno.EIO
    assert writer.closed
    assert all(buffer.closed for buffer in buffers)
    with pytest.raises(OSError) as error:
        os.fstat(fd)
    assert error.value.errno == errno.EBADF
    writer.close()
