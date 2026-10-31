#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""When saved tensors come back: the reload policies.

The tracker owns the facts (which segment executions are pending, in which
order their backwards will run, which group is being unpacked); a policy turns
them into reload requests.  Two policies exist, selected by
``OffloadConfig.prefetch_unit``:

* :class:`ExecutionPrefetch` (``backward``): a segment execution is the unit,
  as the pipeline schedule sees it.  The demanded execution is reloaded as a
  whole and ``prefetch_backwards`` announced executions are reloaded ahead.
* :class:`GroupPrefetch` (``group``): offload groups inside the running
  execution are reloaded ``prefetch_groups`` ahead of the demanded one.

Both honour the ``prefetch_bytes`` budget for ahead-of-demand reloads and
reload demanded tensors unconditionally.
"""

from __future__ import annotations

from typing import Sequence, TYPE_CHECKING

from .config import OffloadConfig

if TYPE_CHECKING:
    from .tracker import ExecContext, OffloadBatch
    from .transport import CopyEngine


class PrefetchPolicy:
    """Base: the demand-driven parts every policy shares."""

    def __init__(self, config: OffloadConfig, engine: 'CopyEngine') -> None:
        self.config = config
        self.engine = engine

    # -- the three decisions ---------------------------------------------------------
    def reload_demanded(self, ctx: 'ExecContext') -> None:
        """The backward of ``ctx`` starts now."""
        raise NotImplementedError

    def reload_ahead(self, upcoming: Sequence['ExecContext']) -> bool:
        """``upcoming`` are the executions whose backwards follow, in order.

        Returns ``True`` when everything the policy wanted is in flight; ``False``
        when the budget stopped it (the caller retries later).
        """
        raise NotImplementedError

    def on_unpack(self, batch: 'OffloadBatch') -> None:
        """A saved tensor of ``batch`` (inside a tracked execution) is being unpacked."""
        raise NotImplementedError

    # -- shared behaviour ---------------------------------------------------------------
    def on_eager_unpack(self, batch: 'OffloadBatch') -> None:
        """No executor: reload the batch and ``prefetch_groups`` batches created before it."""
        batch.prefetch()
        previous = batch.previous() if batch.previous is not None else None
        count = self.config.prefetch_groups
        while previous is not None and count > 0:
            previous.prefetch()
            count -= 1
            previous = previous.previous() if previous.previous is not None else None

    def _reload_last_groups(self, ctx: 'ExecContext', index: int, *, include_self: bool) -> None:
        """Reload ``ctx.batches[index]`` (if ``include_self``) and ``prefetch_groups`` groups before it.

        Groups that saved nothing cost nothing; the budget stops the lookahead,
        never the demanded group.
        """
        budget_groups = self.config.prefetch_groups
        taken = 0
        for i in range(index, -1, -1):
            batch = ctx.batches[i]
            if i == index:
                if include_self:
                    batch.prefetch()
                continue
            if not batch.has_regions or batch.prefetched:
                if batch.prefetched:
                    taken += 1
                continue
            if taken >= budget_groups or not self.engine.within_reload_budget():
                break
            batch.prefetch()
            taken += 1


class ExecutionPrefetch(PrefetchPolicy):
    """Whole segment executions, ``prefetch_backwards`` of them ahead."""

    def reload_demanded(self, ctx: 'ExecContext') -> None:
        ctx.prefetch()

    def reload_ahead(self, upcoming: Sequence['ExecContext']) -> bool:
        for ctx in upcoming[:self.config.prefetch_backwards]:
            if ctx.prefetched or not ctx.batches:
                continue
            if not self.engine.within_reload_budget():
                return False
            ctx.prefetch()
        return True

    def on_unpack(self, batch: 'OffloadBatch') -> None:
        batch.prefetch()          # demanded; normally already in flight with its execution


class GroupPrefetch(PrefetchPolicy):
    """Groups inside the running execution, ``prefetch_groups`` ahead of demand."""

    def reload_demanded(self, ctx: 'ExecContext') -> None:
        if ctx.batches:
            self._reload_last_groups(ctx, len(ctx.batches) - 1, include_self=True)

    def reload_ahead(self, upcoming: Sequence['ExecContext']) -> bool:
        for ctx in upcoming[:1]:
            if ctx.prefetched or not ctx.batches:
                continue
            if not self.engine.within_reload_budget():
                return False
            self._reload_last_groups(ctx, len(ctx.batches) - 1, include_self=True)
            ctx.prefetched = True
        return True

    def on_unpack(self, batch: 'OffloadBatch') -> None:
        self._reload_last_groups(batch.context, batch.index, include_self=True)


_POLICIES = {'backward': ExecutionPrefetch, 'group': GroupPrefetch}


def prefetch_policy_for(config: OffloadConfig, engine: 'CopyEngine') -> PrefetchPolicy:
    """The policy selected by ``config.prefetch_unit``."""
    try:
        return _POLICIES[config.prefetch_unit](config, engine)
    except KeyError:
        raise ValueError(f'unknown prefetch_unit {config.prefetch_unit!r}, expected one of {sorted(_POLICIES)}') from None
