"""Drop the page cache behind mmap-loaded checkpoint tensors once they are consumed.

Merging a sharded checkpoint reads every shard (often `torch.load(mmap=True)`)
exactly once while the merged state grows in anonymous memory. The consumed
shard pages stay cached, so the merged state has to push them out of the
memory cgroup through reclaim, slowing the merge and risking OOM kills. After
each copy, the merge releases its source range: `madvise(MADV_DONTNEED)` unmaps
the pages from this process, then `posix_fadvise(POSIX_FADV_DONTNEED)` drops
them from the page cache (it skips pages that are still mapped).

Only ranges inside read-only-loaded file mappings are released: their pages
are clean, so a later read faults them back from the file and no data can be
lost. Anything else, including anonymous memory, is ignored.

Dropping pages costs the kernel ~0.15 s/GiB, so releases run on a background
thread (both calls drop the GIL) instead of stalling the merge. Each queued
release holds a reference to its tensor, which keeps the mapping alive until
the release ran; `close()` waits for all of them.
Disable with NNSCALER_MERGE_RELEASE_PAGES=0.
"""

import bisect
import ctypes
import logging
import mmap
import os
import queue
import sys
import threading

import torch

_logger = logging.getLogger(__name__)

_MADV_DONTNEED = 4
_PAGE_SIZE = mmap.PAGESIZE


def _file_mappings():
    """Sorted (start, end, file_offset, path) of file-backed mappings."""
    mappings = []
    with open('/proc/self/maps') as maps:
        for line in maps:
            fields = line.split(maxsplit=5)
            # address perms offset dev inode [path]; inode 0 means anonymous
            if len(fields) < 6 or fields[4] == '0':
                continue
            path = fields[5].rstrip('\n')
            if not path.startswith('/') or path.endswith(' (deleted)'):
                continue
            start, end = (int(x, 16) for x in fields[0].split('-'))
            mappings.append((start, end, int(fields[2], 16), path))
    mappings.sort()
    return mappings


class FilePageReleaser:
    """Releases consumed pages of tensors backed by file mappings.

    Build it after the checkpoint files are mapped (mappings created later are
    not tracked) and call `release(tensor)` after a tensor has been copied.
    """

    def __init__(self):
        self.enabled = (sys.platform.startswith('linux')
                        and hasattr(os, 'posix_fadvise')
                        and os.environ.get('NNSCALER_MERGE_RELEASE_PAGES', '1') != '0')
        self.released_bytes = 0
        self._mappings = []
        self._fds = {}  # used by the worker thread only
        self._queue = queue.SimpleQueue()
        self._thread = None
        self._error = None
        if not self.enabled:
            return
        try:
            self._mappings = _file_mappings()
            libc = ctypes.CDLL(None, use_errno=True)
            self._madvise = libc.madvise
            self._madvise.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
            self._madvise.restype = ctypes.c_int
        except (OSError, AttributeError) as e:
            _logger.info('Not releasing checkpoint pages while merging: %s', e)
            self.enabled = False
            return
        self._starts = [m[0] for m in self._mappings]
        self.enabled = bool(self._mappings)

    def release(self, tensor):
        """Queue the whole pages of a consumed tensor for release."""
        if not self.enabled or self._error is not None:
            return
        if not isinstance(tensor, torch.Tensor) or tensor.device.type != 'cpu':
            return
        if not tensor.is_contiguous() or tensor.numel() == 0:
            return
        start = tensor.data_ptr()
        end = start + tensor.numel() * tensor.element_size()
        index = bisect.bisect_right(self._starts, start) - 1
        if index < 0:
            return
        map_start, map_end, file_offset, path = self._mappings[index]
        if end > map_end:
            return  # not entirely inside one file mapping
        # Only whole pages: neighbouring bytes may belong to unmerged tensors.
        first = -(-start // _PAGE_SIZE) * _PAGE_SIZE
        last = end // _PAGE_SIZE * _PAGE_SIZE
        if last <= first:
            return
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._run, name='nnscaler-release-pages', daemon=True)
            self._thread.start()
        # `tensor` keeps the mapping alive until the worker released it.
        self._queue.put((tensor, first, last - first, path, file_offset + first - map_start))

    def _run(self):
        while True:
            item = self._queue.get()
            if item is None:
                return
            tensor, addr, length, path, offset = item
            if self._error is not None:
                continue
            try:
                fd = self._fds.get(path)
                if fd is None:
                    fd = self._fds[path] = os.open(path, os.O_RDONLY)
                if self._madvise(addr, length, _MADV_DONTNEED) != 0:
                    raise OSError(ctypes.get_errno(), 'madvise(MADV_DONTNEED) failed')
                os.posix_fadvise(fd, offset, length, os.POSIX_FADV_DONTNEED)
                self.released_bytes += length
            except OSError as e:
                self._error = e
            finally:
                del tensor, item

    def close(self):
        """Wait for queued releases; returns the number of released bytes."""
        if self._thread is not None:
            self._queue.put(None)
            self._thread.join()
            self._thread = None
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()
        if self._error is not None:
            _logger.info('Stopped releasing checkpoint pages while merging: %s', self._error)
        return self.released_bytes
