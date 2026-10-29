#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""A simple, standalone sequential file writer using O_DIRECT.

Background
----------
Regular write() calls copy data into the kernel page cache, which the kernel
flushes to disk asynchronously. Terabyte-scale checkpoints can fill the page
cache with dirty pages, making kernel throttling, writeback, and the final fsync
slow. O_DIRECT bypasses the page cache but imposes three alignment requirements:

    1. The user-space buffer address must be aligned (typically to a 4 KiB page).
    2. The file offset must be aligned.
    3. The length of each write must be aligned.

Callers such as torch.save produce byte streams with arbitrary lengths and
addresses that do not meet these requirements. This class uses a bounce buffer
to bridge the gap:

    Arbitrary byte stream -> copy into a page-aligned, fixed-size buffer
                          -> pwrite(O_DIRECT) when the buffer is full

The final block is usually shorter than a page. It is zero-padded to the next
page boundary before writing, then ftruncate restores the actual file length.
The resulting contents are identical to those produced by regular writes.

Usage
-----
    with DirectIOWriter("/data/ckpt.pt", chunk_bytes=64 << 20, workers=2) as f:
        torch.save(state_dict, f)      # Only requires write() and flush().

The interface implements a subset of io.RawIOBase: write / tell / flush / close.
Instances can be passed directly to torch.save.

Limitations
-----------
    - Append-only: no read, seek, or truncate. torch.save with skip_data=True
      needs seek() and is not supported.
    - fileno() raises io.UnsupportedOperation. The descriptor is opened with
      O_DIRECT and written only through pwrite, so its file position never
      advances; callers that write to the raw descriptor (e.g. torch.save with
      _use_new_zipfile_serialization=False) would corrupt the file or get EINVAL.
      Raising makes such callers fall back to write().
    - write() accepts any object exposing a C-contiguous buffer (bytes,
      bytearray, memoryview, array). Non-contiguous memoryviews are rejected.
    - Not thread-safe: all methods must be called from one thread.
    - Always close() the writer or use it as a context manager. A writer that
      is garbage collected or still open at interpreter exit only releases its
      resources without flushing; unflushed data is lost.

Performance
-----------
Linux/ext4, 64 MiB chunks, 2 workers, 64 MiB warmup; medians of 3 runs including
fsync and close.
In the raw-write benchmark, each write() call passes 8 MiB to either writer.
The torch.save benchmark includes CPU tensor serialization.
Throughput in MiB/s (higher is better):

    Device  Workload    Size    DirectIOWriter  Buffered
    ------  ----------  ------  --------------  --------
    SSD     raw          4 GiB            1196       813
    SSD     torch.save   4 GiB             928       652
    HDD     raw          4 GiB             191       185

NOTE: The optimal workers and chunk_bytes values depend on the storage device.
"""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import io
import mmap
import os
import weakref

__all__ = ["DirectIOWriter"]

# O_DIRECT alignment granularity. The actual minimum is the device's logical
# block size (often 512 B), but page alignment is safe across filesystems and
# naturally matches buffers allocated by mmap.
_ALIGN = mmap.PAGESIZE


class DirectIOWriter:
    """Pack arbitrary byte streams into page-aligned, threaded O_DIRECT writes."""

    def __init__(self, path, *, chunk_bytes=64 << 20, workers=2, delete_on_error=True):
        """
        path:            Output path. The file must not exist (O_EXCL), preventing
                         existing checkpoints from being overwritten.
        chunk_bytes:     Bounce buffer size and full-block pwrite length. Must be a
                         multiple of the page size. Larger buffers reduce system
                         calls and improve throughput but use more memory.
        workers:         Number of parallel writer threads. The buffer pool contains
                         2 * workers buffers, leaving workers buffers for the main
                         thread to fill while workers threads are writing.
        delete_on_error: Remove the output file when the with block exits with an
                         exception or its final flush fails. An incomplete file is
                         not a usable checkpoint, and because of O_EXCL it would
                         also block a retry to the same path. Set False to keep
                         the partial file for inspection.
        """
        if chunk_bytes < _ALIGN or chunk_bytes % _ALIGN:
            raise ValueError(f"chunk_bytes must be a positive multiple of {_ALIGN}")
        if workers < 1:
            raise ValueError("workers must be >= 1")

        with ExitStack() as cleanup:
            # O_EXCL prevents overwrites; O_DIRECT requires aligned writes.
            self._fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_DIRECT, 0o666)
            self._path = path
            self._delete_on_error = delete_on_error
            # The file was created above; if construction fails the caller has no
            # writer to clean up, and O_EXCL would reject a retry. Registered
            # before os.close so LIFO cleanup closes first, then unlinks.
            if delete_on_error:
                cleanup.callback(self._unlink)
            cleanup.callback(os.close, self._fd)
            self._chunk_bytes = chunk_bytes
            self._pool = cleanup.enter_context(ThreadPoolExecutor(max_workers=workers))

            # Anonymous mmap is page-aligned; bytearray does not guarantee alignment.
            # MAP_PRIVATE: Python defaults to MAP_SHARED, which would share the
            # buffers with forked children for no reason.
            self._buffers = [
                cleanup.enter_context(mmap.mmap(-1, chunk_bytes,
                                                flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS))
                for _ in range(2 * workers)
            ]
            self._free = list(self._buffers)  # Free buffers available for filling.
            self._inflight = deque()          # (future, buffer) pairs being written.

            self._cur = None                 # None until a buffer is acquired.
            self._cur_used = 0               # Bytes filled in the current buffer.
            self._cur_flushed = True         # All current contents written (see flush).

            # _logical_offset: Bytes written by the caller; both tell() and the actual
            #                  file length use this value.
            # _disk_offset:    File offset of the current buffer, always a multiple of
            #                  chunk_bytes. Advances by chunk_bytes per full block submitted.
            self._logical_offset = 0
            self._disk_offset = 0

            self.closed = False
            # Initialization succeeded; close() now owns resource cleanup.
            cleanup.pop_all()

        # Release resources if the caller forgets to close(), like io objects do.
        # weakref.finalize instead of __del__: it also runs at interpreter exit,
        # but during atexit while modules are intact, whereas __del__ may run
        # after module teardown and fail inside ThreadPoolExecutor.shutdown.
        # Deliberately no flush: finalizing a checkpoint the caller never asked
        # to complete would be surprising.
        self._finalizer = weakref.finalize(self, self._release, self._fd, self._pool, self._buffers)

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------
    def write(self, data):
        """Append data and return the number of bytes written (always len(data))."""
        self._check_open()
        view = memoryview(data).cast("B")
        pos = 0
        while pos < len(view):
            if self._cur is None:
                self._acquire_buffer()
            # Copy no more than the remaining data or available buffer space.
            n = min(len(view) - pos, self._chunk_bytes - self._cur_used)
            self._cur[self._cur_used:self._cur_used + n] = view[pos:pos + n]
            self._cur_used += n
            self._logical_offset += n
            self._cur_flushed = False
            pos += n
            # Submit full blocks to the thread pool for asynchronous writes.
            if self._cur_used == self._chunk_bytes:
                self._submit_current()
        return len(view)

    def tell(self):
        """Return the number of bytes written so far, i.e. the final file length."""
        return self._logical_offset

    def fileno(self):
        # See "Limitations" in the module docstring: exposing the O_DIRECT
        # descriptor lets callers bypass the bounce buffer and corrupt the file.
        raise io.UnsupportedOperation("fileno")

    # ------------------------------------------------------------------
    # Flushing
    # ------------------------------------------------------------------
    def flush(self):
        """Write all data, including any partial trailing block, and fsync.

        Writes may continue after flush: the partial trailing block stays in
        memory for further filling and is rewritten on the next flush, leaving
        no zero-padding gaps in the file.
        """
        self._check_open()
        # 1. Wait for all in-flight full-block writes.
        while self._inflight:
            self._wait_oldest()

        # 2. Zero-pad the partial trailing block to a page boundary and write it
        #    synchronously. Keep _disk_offset unchanged so it can be rewritten.
        if self._cur is not None and self._cur_used and not self._cur_flushed:
            padded = -(-self._cur_used // _ALIGN) * _ALIGN      # Round up to a page boundary.
            self._cur[self._cur_used:padded] = b"\0" * (padded - self._cur_used)
            self._pwrite_all(self._cur, padded, self._disk_offset)
            self._cur_flushed = True

        # 3. Remove padding with ftruncate to restore the actual length, then fsync.
        os.ftruncate(self._fd, self._logical_offset)
        os.fsync(self._fd)

    # ------------------------------------------------------------------
    # Closing / context management
    # ------------------------------------------------------------------
    def close(self):
        """Flush and release all resources. Repeated calls are safe."""
        if self.closed:
            return
        try:
            self.flush()
        finally:
            self._release_resources()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        failed = exc_type is not None
        try:
            if failed:
                # Only clean up: flushing could raise again while handling an error.
                self._release_resources()
            else:
                self.close()
        except BaseException:
            failed = True
            raise
        finally:
            if failed and self._delete_on_error:
                self._unlink()

    # ------------------------------------------------------------------
    # Internal implementation
    # ------------------------------------------------------------------
    def _check_open(self):
        if self.closed:
            raise ValueError("DirectIOWriter is closed")

    def _acquire_buffer(self):
        """Acquire a free buffer, waiting for the oldest write if none are free."""
        if not self._free:
            self._wait_oldest()
        self._cur = self._free.pop()
        self._cur_used = 0
        self._cur_flushed = True        # An empty buffer has no pending data.

    def _submit_current(self):
        """Submit the full current buffer to the thread pool and clear _cur."""
        future = self._pool.submit(self._pwrite_all, self._cur, self._cur_used, self._disk_offset)
        self._inflight.append((future, self._cur))
        self._disk_offset += self._cur_used
        self._cur = None
        self._cur_used = 0
        self._cur_flushed = True

    def _wait_oldest(self):
        """Wait for the oldest write and reclaim its buffer, propagating errors."""
        future, buffer = self._inflight[0]
        future.result()                 # Re-raise any write error here.
        self._inflight.popleft()
        self._free.append(buffer)

    def _pwrite_all(self, buffer, length, offset):
        """Write all of buffer[:length] at the file offset, handling short writes.

        The buffer address, offset, and length already meet O_DIRECT alignment
        requirements. The kernel may return a short write; loop until complete.
        """
        view = memoryview(buffer)[:length]
        try:
            while view:
                n = os.pwrite(self._fd, view, offset)
                if n <= 0:
                    raise OSError(f"O_DIRECT write made no progress (offset={offset})")
                # Direct I/O completes in whole filesystem blocks, so a short
                # write keeps the remaining address and offset aligned. Assert
                # it instead of letting the next pwrite fail with EINVAL.
                assert n % _ALIGN == 0, f"unaligned O_DIRECT short write: {n} bytes"
                offset += n
                view = view[n:]
        finally:
            view.release()              # Release exported views before mmap.close().

    @staticmethod
    def _release(fd, pool, buffers):
        """Shut down the pool, close the file, and free buffers.

        A staticmethod so weakref.finalize holds no reference to the writer; a
        bound method would keep it alive forever.
        """
        try:
            pool.shutdown(wait=True, cancel_futures=True)
        finally:
            os.close(fd)
            for buf in buffers:
                buf.close()

    def _release_resources(self):
        """Shut down the pool, close the file, and free buffers exactly once."""
        if self.closed:
            return
        self.closed = True
        try:
            self._finalizer()           # Runs _release once; later calls are no-ops.
        finally:
            self._cur = None
            self._inflight.clear()
            self._free.clear()
            self._buffers.clear()

    def _unlink(self):
        """Remove the incomplete output file; it may already be gone."""
        try:
            os.unlink(self._path)
        except FileNotFoundError:
            pass
