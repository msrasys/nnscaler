# Copyright (c) Microsoft Corporation. Licensed under the MIT License.
"""Local F/B action scheduling, independent of model layers and CUDA transfers."""
from bisect import bisect_left, bisect_right
from dataclasses import dataclass


@dataclass(frozen=True)
class ActivationScopeUse:
    key: object
    forward: int
    backward: int


class ActivationSchedulePlan:
    """One key per virtual-stage/microbatch; positions count local F/B actions."""
    def __init__(self, entries):
        uses = [ActivationScopeUse(*entry) for entry in entries]
        keys = set()
        for use in uses:
            if use.key in keys:
                raise ValueError(f'Duplicate activation scope in schedule: {use.key}')
            keys.add(use.key)
            if (any(isinstance(v, bool) or not isinstance(v, int) for v in (use.forward, use.backward))
                    or not 0 <= use.forward < use.backward):
                raise ValueError('Activation scope must be produced before its backward action')
        self.uses = tuple(sorted(uses, key=lambda use: use.backward))
        self.positions = tuple(use.backward for use in self.uses)

    @classmethod
    def from_actions(cls, actions):
        """Build from (scope_key, is_forward, has_offload) for every local action."""
        forwards, backwards, entries = {}, set(), []
        for index, (key, is_forward, eligible) in enumerate(actions):
            if is_forward and eligible:
                if key in forwards:
                    raise ValueError(f'Duplicate forward action for activation scope: {key}')
                forwards[key] = (index, eligible)
            elif not is_forward and eligible:
                if key not in forwards or key in backwards:
                    raise ValueError(f'Backward action has no unique local producer: {key}')
                backwards.add(key)
                entries.append((key, forwards[key][0], index))
        return cls(entries)

    def entries(self):
        return tuple((use.key, use.forward, use.backward) for use in self.uses)

    def candidates(self, position, steps, *, after=False):
        """Nearest backward first, clipped to completed forward production."""
        if not steps:
            return ()
        low = position + int(after)
        first = bisect_left(self.positions, low)
        last = bisect_right(self.positions, position + steps)
        return tuple(use for use in self.uses[first:last]
                     if use.forward < position + int(after))

    def describe(self, steps):
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError('Prefetch steps must be a non-negative integer')
        rows = []
        for use in self.uses:
            wanted = max(0, use.backward - steps)
            trigger = None if not steps else (
                ('before', wanted) if use.forward < wanted else ('after', use.forward))
            rows.append(dict(key=use.key, forward=use.forward, backward=use.backward,
                             first_prefetch=trigger))
        return rows


class ActivationScheduleCursor:
    """Retry ready scopes within the window; nearer blocked readers take priority."""
    def __init__(self, plan, steps, request):
        plan.describe(steps)  # Validate even an empty schedule.
        self.plan, self.steps, self.request = plan, steps, request
        self.admitted = set()

    def _advance(self, position, after):
        for use in self.plan.candidates(position, self.steps, after=after):
            if use.key in self.admitted:
                continue
            lead = use.backward - position - int(after)
            if self.request(use.key, lead) is False:
                break
            self.admitted.add(use.key)

    def before(self, position):
        self._advance(position, False)

    def after(self, position):
        self._advance(position, True)
