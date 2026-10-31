#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""Whether and how a transfer's bytes really move.

:class:`CopyMode` is the real transport.  The A/B modes selected by
``OffloadConfig.fake_mode`` keep the saved tensors on the device while
exercising the rest of the machinery, so a run can be split into "bookkeeping
cost" and "cost of the bytes".  :class:`~.transport.Transfer` never looks at
the mode's name: it calls the steps below and the mode decides what happens.
"""

from __future__ import annotations

from typing import Tuple, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .hooks import OffloadedTensor
    from .transport import Chunk, Transfer


class CopyMode:
    """Real transport: D2H after the group, H2D into device buffers, sources released after the copy."""

    name = 'off'
    moves_bytes = True        # some copy happens (communication windows are pointless otherwise)
    keeps_sources = False     # saved views stay alive and are what consumers read

    def commit(self, transfer: 'Transfer') -> None:
        """The regions are laid out: order the D2H copies after the producing kernels."""
        transfer.record_ready()
        transfer.engine.schedule_d2h(transfer)

    def after_d2h(self, transfer: 'Transfer') -> None:
        """The D2H copies are issued: the device memory of the saved views may go."""
        if not transfer.hold_src:
            transfer.drop_sources()

    def release_hold(self, transfer: 'Transfer') -> None:
        """A later group no longer needs the sources for sharing."""
        transfer.hold_src = False
        if transfer.d2h_issued:
            transfer.drop_sources()

    def reload(self, transfer: 'Transfer', h2d: torch.cuda.Stream, consumer: torch.cuda.Stream) -> None:
        """Bring the slab back: one device buffer and one copy per chunk."""
        transfer.copy_chunks_to_device(h2d, consumer)

    def locate(self, handle: 'OffloadedTensor') -> Tuple[torch.Tensor, int]:
        """``(bytes, start)``: the device bytes holding the region and the region's start inside them."""
        region = handle.region
        return region.chunk.buffer, region.offset - region.chunk.start

    def missing_reload(self, chunk: 'Chunk') -> bool:
        """Whether a consumer of ``chunk`` finds no device bytes (its buffer was consumed and dropped)."""
        return chunk.buffer is None


class NoCopy(CopyMode):
    """Nothing moves; consumers read the saved views.  Measures the bookkeeping alone."""

    name = 'no_copy'
    moves_bytes = False
    keeps_sources = True

    def commit(self, transfer: 'Transfer') -> None:
        transfer.mark_resident()

    def after_d2h(self, transfer: 'Transfer') -> None:
        pass

    def release_hold(self, transfer: 'Transfer') -> None:
        transfer.hold_src = False

    def reload(self, transfer: 'Transfer', h2d: torch.cuda.Stream, consumer: torch.cuda.Stream) -> None:
        pass

    def locate(self, handle: 'OffloadedTensor') -> Tuple[torch.Tensor, int]:
        return handle.region.src, 0

    def missing_reload(self, chunk: 'Chunk') -> bool:
        return False


class D2HOnly(NoCopy):
    """The D2H copies run (and consumers wait for them), nothing comes back; sources stay."""

    name = 'd2h_only'
    moves_bytes = True

    def commit(self, transfer: 'Transfer') -> None:
        CopyMode.commit(self, transfer)

    def reload(self, transfer: 'Transfer', h2d: torch.cuda.Stream, consumer: torch.cuda.Stream) -> None:
        transfer.alias_reload_to_d2h()


class BothKeep(D2HOnly):
    """Both copies run; consumers still read the saved views.  Measures the full transfer cost."""

    name = 'both_keep'

    def reload(self, transfer: 'Transfer', h2d: torch.cuda.Stream, consumer: torch.cuda.Stream) -> None:
        transfer.copy_chunks_to_device(h2d, consumer)


_MODES = {mode.name: mode for mode in (CopyMode, NoCopy, D2HOnly, BothKeep)}


def copy_mode_for(fake_mode: str) -> CopyMode:
    """The :class:`CopyMode` for an ``OffloadConfig.fake_mode`` value."""
    try:
        return _MODES[fake_mode]()
    except KeyError:
        raise ValueError(f'unknown fake_mode {fake_mode!r}, expected one of {sorted(_MODES)}') from None
