#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""Compatibility entry point; the implementation lives in :mod:`nnscaler.runtime.cpu_offload`.

Generated code refers to ``nnscaler.runtime.cpu_offloading.hint_forward`` and
``nnscaler.runtime.cpu_offloading.prefetch_next_backward``; keep this module
importable with those names.
"""

from nnscaler.runtime.cpu_offload import (  # noqa: F401
    OffloadConfig, OffloadManager, CPUOffloadContext, comm_exclusive, configure,
    hint_forward, prefetch_next_backward, offload_stats, offload_report, all_rank_max,
    OffloadStats, GroupProfile,
)
from nnscaler.runtime.cpu_offload.hooks import (  # noqa: F401
    OffloadedTensor as _OffloadedTensor,
    ModuleTensorRef as _ModuleTensorRef,
)
from nnscaler.runtime.cpu_offload.tracker import OffloadBatch as _OffloadBatch  # noqa: F401
from nnscaler.runtime.cpu_offload.host_memory import PinnedPool as _PinnedPool  # noqa: F401
from nnscaler.runtime.cpu_offload.transport import H2D_STREAM_NAME as _PREFETCH_STREAM_NAME  # noqa: F401
from nnscaler.runtime.device import DeviceGroup as _DeviceGroup


def _get_prefetch_stream():
    return _DeviceGroup().get_stream(_PREFETCH_STREAM_NAME)
