#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

# CREDITS: most of the code is from torch: https://github.com/pytorch/pytorch/blob/main/torch/distributed/pipelining/schedules.py

from collections import defaultdict
from typing import Dict, List, Optional
from enum import Enum
from typing import NamedTuple
import re
import logging

logger = logging.getLogger(__name__)


class _ComputationType(Enum):
    # TODO(whc) rename to _ActType?
    FORWARD = 1
    BACKWARD_INPUT = 2
    BACKWARD_WEIGHT = 3
    UNSHARD = 4
    RESHARD = 5
    SEND_F = 6
    RECV_F = 7
    SEND_B = 8
    RECV_B = 9
    FULL_BACKWARD = 10

    def __str__(self):
        str_map = {
            _ComputationType.FORWARD: "F",
            _ComputationType.BACKWARD_INPUT: "I",
            _ComputationType.BACKWARD_WEIGHT: "W",
            _ComputationType.UNSHARD: "UNSHARD",
            _ComputationType.RESHARD: "RESHARD",
            _ComputationType.SEND_F: "SEND_F",
            _ComputationType.RECV_F: "RECV_F",
            _ComputationType.SEND_B: "SEND_B",
            _ComputationType.RECV_B: "RECV_B",
            _ComputationType.FULL_BACKWARD: "B",
        }
        return str_map[self]

    @staticmethod
    def from_str(action):
        if action == "F":
            return _ComputationType.FORWARD
        elif action == "I":
            return _ComputationType.BACKWARD_INPUT
        elif action == "W":
            return _ComputationType.BACKWARD_WEIGHT
        elif action == "UNSHARD":
            return _ComputationType.UNSHARD
        elif action == "RESHARD":
            return _ComputationType.RESHARD
        elif action == "SEND_F":
            return _ComputationType.SEND_F
        elif action == "RECV_F":
            return _ComputationType.RECV_F
        elif action == "SEND_B":
            return _ComputationType.SEND_B
        elif action == "RECV_B":
            return _ComputationType.RECV_B
        elif action == "B":
            return _ComputationType.FULL_BACKWARD
        else:
            raise RuntimeError(f"Invalid computation type {action}")


FORWARD = _ComputationType.FORWARD
BACKWARD_INPUT = _ComputationType.BACKWARD_INPUT
BACKWARD_WEIGHT = _ComputationType.BACKWARD_WEIGHT
UNSHARD = _ComputationType.UNSHARD
RESHARD = _ComputationType.RESHARD
SEND_F = _ComputationType.SEND_F
RECV_F = _ComputationType.RECV_F
SEND_B = _ComputationType.SEND_B
RECV_B = _ComputationType.RECV_B
FULL_BACKWARD = _ComputationType.FULL_BACKWARD

# Convenience shorthand for compute actions only since they are used in 'simple schedule format'
F = FORWARD
I = BACKWARD_INPUT
W = BACKWARD_WEIGHT
B = FULL_BACKWARD

# Helper to parse an action string like 1F0 into a tuple of (stage_index, computation_type, microbatch_index)
_action_regex = re.compile(
    r"(\d+)(F|I|B|W|UNSHARD|RESHARD|SEND_F|RECV_F|SEND_B|RECV_B)(\d*)"
)


class _Action(NamedTuple):
    stage_index: int
    computation_type: _ComputationType
    microbatch_index: Optional[int] = None

    def __repr__(self):
        repr = str(self.stage_index)
        repr += str(self.computation_type)
        if self.microbatch_index is not None:
            repr += str(self.microbatch_index)
        return repr

    @staticmethod
    def from_str(action_string: str):
        """
        Reverse of __repr__

        String should be formatted as [stage][action type][(microbatch)]
            e.g. `2F0`, `1UNSHARD`, `3SEND_F1`
        """
        action_string = action_string.strip()
        if match := _action_regex.match(action_string):
            stage_index, computation_type, microbatch_index = match.groups()
            return _Action(
                int(stage_index),
                _ComputationType.from_str(computation_type),
                int(microbatch_index) if len(microbatch_index) else None,
            )
        elif action_string == "":
            return None
        raise RuntimeError(
            f"Invalid action string: {action_string}, should be formatted as [stage][action type][(microbatch)] e.g. 2F0"
        )


def _get_1f1b_rank_ops(
    n_local_stages,
    pp_group_size,
    warmup_ops,
    fwd_bwd_ops,
    cooldown_ops,
    rank,
    forward_stage_index,
    backward_stage_index,
    num_1f1b_microbatches=0,
    enable_zero_bubble=False,
):
    # All stages start with handling microbatch 0
    fwd_stage_mb_index: Dict[int, int] = defaultdict(int)
    bwd_stage_mb_index: Dict[int, int] = defaultdict(int)
    weight_stage_mb_index: Dict[int, int] = defaultdict(int)

    # Store the list of operations used for that rank
    # Pre-padding, rank starts with no-ops based on the warmup.
    rank_ops: List[Optional[_Action]] = [None for _ in range(rank)]
    # These are used to calculate the number of slots to fill with no-ops, to account for the delay in warmup
    # when we want to wait for the backward to trickle back up and start 1f1b to align all ranks.
    # Formula:
    # pre-padding + warmup_ops + post_warmup_ops = earliest time step of first backward
    # post_warmup_ops = [earliest time step of first backward] - (warmup_ops + pre-padding)
    # earliest time step of first backward = [local_stages * group_size + 2 * (group_size - 1 - rank)]
    # warmup_ops = calculated above
    post_warmup_ops = (
        n_local_stages * pp_group_size + 2 * (pp_group_size - 1 - rank)
    ) - (warmup_ops + rank)

    if enable_zero_bubble:
        post_warmup_ops = pp_group_size - rank - 1

    total_ops = warmup_ops + fwd_bwd_ops + cooldown_ops

    backward_op_ids = []
    weight_op_count = 0

    FULL_BACKWARD_OR_BACKWARD_INPUT = (
        BACKWARD_INPUT if enable_zero_bubble else FULL_BACKWARD
    )

    for op in range(total_ops):
        # Warmup phase
        if op < warmup_ops:
            fwd_stage_index = forward_stage_index(op)
            # This will assign the current microbatch index and update it as well
            fwd_stage_mb_index[fwd_stage_index] = (
                mb_index := fwd_stage_mb_index[fwd_stage_index]
            ) + 1
            rank_ops.append(
                _Action(fwd_stage_index, _ComputationType.FORWARD, mb_index)
            )
            if op == warmup_ops - 1:
                # This is the last step in the warmup phase, so we need to wait for the backward to trickle back up
                rank_ops.extend([None] * post_warmup_ops)
        # 1F1B Phase (forward and backward)
        elif warmup_ops <= op < warmup_ops + fwd_bwd_ops:
            fwd_stage_index = forward_stage_index(op)
            fwd_stage_mb_index[fwd_stage_index] = (
                fwd_mb_index := fwd_stage_mb_index[fwd_stage_index]
            ) + 1
            rank_ops.append(
                _Action(fwd_stage_index, _ComputationType.FORWARD, fwd_mb_index)
            )
            bwd_stage_index = backward_stage_index(op)
            bwd_stage_mb_index[bwd_stage_index] = (
                bwd_mb_index := bwd_stage_mb_index[bwd_stage_index]
            ) + 1
            rank_ops.append(
                _Action(bwd_stage_index, FULL_BACKWARD_OR_BACKWARD_INPUT, bwd_mb_index)
            )
            backward_op_ids.append(op)

            if enable_zero_bubble and op - warmup_ops >= num_1f1b_microbatches:
                weight_stage_index = backward_stage_index(
                    backward_op_ids[weight_op_count]
                )
                weight_stage_mb_index[weight_stage_index] = (
                    weight_mb_index := weight_stage_mb_index[weight_stage_index]
                ) + 1
                rank_ops.append(
                    _Action(
                        weight_stage_index,
                        _ComputationType.BACKWARD_WEIGHT,
                        weight_mb_index,
                    )
                )
                weight_op_count += 1
        # Cooldown phase
        else:
            # During cooldown phase, we need steps to align with 1f1b happening in other ranks
            # TODO: we don't need to always append, after all 1f1b are finished we can stop appending None
            if not enable_zero_bubble:
                rank_ops.append(None)

            bwd_stage_index = backward_stage_index(op)
            bwd_stage_mb_index[bwd_stage_index] = (
                bwd_mb_index := bwd_stage_mb_index[bwd_stage_index]
            ) + 1
            rank_ops.append(
                _Action(bwd_stage_index, FULL_BACKWARD_OR_BACKWARD_INPUT, bwd_mb_index)
            )
            backward_op_ids.append(op)

            if enable_zero_bubble and op - warmup_ops >= num_1f1b_microbatches:
                weight_stage_index = backward_stage_index(
                    backward_op_ids[weight_op_count]
                )
                weight_stage_mb_index[weight_stage_index] = (
                    weight_mb_index := weight_stage_mb_index[weight_stage_index]
                ) + 1
                rank_ops.append(
                    _Action(
                        weight_stage_index,
                        _ComputationType.BACKWARD_WEIGHT,
                        weight_mb_index,
                    )
                )
                weight_op_count += 1

    while enable_zero_bubble and weight_op_count < len(backward_op_ids):
        weight_stage_index = backward_stage_index(backward_op_ids[weight_op_count])
        weight_stage_mb_index[weight_stage_index] = (
            weight_mb_index := weight_stage_mb_index[weight_stage_index]
        ) + 1
        rank_ops.append(
            _Action(
                weight_stage_index, _ComputationType.BACKWARD_WEIGHT, weight_mb_index
            )
        )
        weight_op_count += 1

    return rank_ops


def _add_bubbles_to_actions(
    actions: Dict[int, List[Optional[_Action]]],
    pp_group_size: int,
    num_stages_global: int,
) -> Dict[int, List[Optional[_Action]]]:
    """Insert bubbles that align per-rank ZB1P actions into global steps.

    ``_calculate_single_rank_operations`` generates one ordered action stream
    for each pipeline rank. Those streams preserve each rank's local execution
    order, but an action may still appear before its producer on another rank.
    This function walks all streams together and inserts ``None`` bubbles until
    every emitted action satisfies the cross-stage dependencies represented by
    :class:`SchedulePlan`.

    Each iteration of the outer loop creates one global step. ``next_pointer``
    identifies the next unconsumed action on every rank. A ready action is added
    to ``result`` and advances its pointer. A blocked action contributes a
    bubble for that step and keeps its pointer unchanged so it can be retried in
    the next step. An input ``None`` is an intentional bubble from the local
    stream; it is copied to the result and consumed immediately.

    The dependency rules are:

    * ``F(s, m)`` waits for ``F(s - 1, m)``, except at the first stage.
    * ``I(s, m)`` waits for ``I(s + 1, m)``, except at the last stage.
    * ``W(s, m)`` waits for ``I(s, m)`` on the same stage.

    Actions emitted in the current step are accumulated in ``temp_seen_ops``
    and become visible through ``seen_ops`` only after every rank has been
    visited. Consequently, a producer and its consumer cannot occupy the same
    global step: the producer must finish before the consumer starts. The local
    streams are assumed to already preserve same-rank ordering such as F before
    I; this pass supplies the missing cross-rank alignment and explicit I-to-W
    ordering.

    This implementation is adapted from PyTorch v2.10.0's
    ``ScheduleInterleavedZeroBubble._add_bubbles_to_actions``:
    https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/pipelining/schedules.py

    PyTorch's pass checks forward and full-backward dependencies, but does not
    explicitly align separate I/W actions. This is valid in PyTorch because its
    compute-only columns are not execution barriers. ``_prepare_schedule_with_comms``
    later discards the ``None`` placeholders and calls ``_add_send_recv``, which
    inserts SEND_B/RECV_B actions and reorders each rank's executable sequence
    according to communication readiness. The runtime also waits for RECV_B
    before executing a non-last-stage I action.

    For example, consider two ranks, one local stage per rank, and one
    microbatch. PyTorch's compute-only table puts ``0I0`` and ``1I0`` in the
    same column:

    .. code-block:: text

        step       0      1      2      3
        rank 0    0F0    ---    0I0    0W0
        rank 1    ---    1F0    1I0    1W0

    This does not mean they run simultaneously. Communication lowering turns
    the relevant part into the following happens-before chain:

    .. code-block:: text

        rank 1: 1I0 -> 1SEND_B0
                         |
                         v
        rank 0:       0RECV_B0 -> 0I0

    Thus ``1I0`` computes the gradient first, SEND_B/RECV_B transfers it, and
    only then can ``0I0`` run. The lowered and dependency-simulated PyTorch
    schedule places ``1I0`` at step 3, SEND_B/RECV_B at step 4, and ``0I0`` at
    step 5.

    nnScaler does not have this later PyTorch schedule-lowering pass. Its
    SchedulePlan columns directly define dependency ordering and validation, so
    the compute schedule itself must express the I dependency. nnScaler extends
    ``need_bubble`` with explicit I/W rules and produces:

    .. code-block:: text

        step       0      1      2      3      4
        rank 0    0F0    ---    ---    0I0    0W0
        rank 1    ---    1F0    1I0    1W0    ---

    At step 3, ``0I0`` and ``1W0`` may run together because both depend on the
    already completed ``1I0`` and neither consumes the other's output. The
    extra bubble therefore does not add a mathematical dependency absent from
    PyTorch; it represents earlier, in SchedulePlan, an ordering that PyTorch
    materializes later while inserting communication actions.

    Args:
        actions: Mapping from pipeline rank to its ordered local ZB1P stream.
            Entries are F/I/W actions or ``None`` bubbles.
        pp_group_size: Number of pipeline ranks (device groups).
        num_stages_global: Total number of stages across all pipeline ranks.

    Returns:
        A rank-to-action mapping aligned to global steps. Result rows may have
        different lengths because a finished rank is not padded with trailing
        ``None`` entries; omitted trailing entries are semantically idle.

    Raises:
        ValueError: If a stream contains an action other than F, I, or W.
        RuntimeError: If unfinished streams exist but no rank can consume an
            action, indicating malformed or cyclic local action ordering.
    """
    def need_bubble(stage, op, microbatch, seen_ops):
        if op == FORWARD:
            return stage != 0 and (stage - 1, op, microbatch) not in seen_ops
        if op == BACKWARD_INPUT:
            return (
                stage != num_stages_global - 1
                and (stage + 1, op, microbatch) not in seen_ops
            )
        if op == BACKWARD_WEIGHT:
            return (stage, BACKWARD_INPUT, microbatch) not in seen_ops
        raise ValueError(f'Unsupported zero-bubble action: {op}')

    seen_ops = set()
    result = {rank: [] for rank in range(pp_group_size)}
    next_pointer = {rank: 0 for rank in range(pp_group_size)}

    while True:
        should_stop = True
        made_progress = False
        temp_seen_ops = set()

        for rank in range(pp_group_size):
            timestamp = next_pointer[rank]
            if timestamp >= len(actions[rank]):
                continue

            should_stop = False
            if actions[rank][timestamp] is not None:
                temp_action = actions[rank][timestamp]
                stage_index, op, microbatch = temp_action
                if not need_bubble(stage_index, op, microbatch, seen_ops):
                    result[rank].append(temp_action)
                    temp_seen_ops.add((stage_index, op, microbatch))
                    next_pointer[rank] += 1
                    made_progress = True
                else:
                    result[rank].append(None)
            else:
                next_pointer[rank] += 1
                result[rank].append(None)
                made_progress = True

        seen_ops.update(temp_seen_ops)
        if should_stop:
            break
        if not made_progress:
            raise RuntimeError('zero_bubble schedule cannot make progress')

    return result


# Unified adaptation of PyTorch v2.10.0's
# ScheduleInterleaved1F1B._calculate_single_rank_operations and
# ScheduleInterleavedZeroBubble._calculate_single_rank_operations:
# https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/pipelining/schedules.py
def _calculate_single_rank_operations(self, rank, enable_zero_bubble=False):
    def get_rank_warmup_ops(rank):
        # Warms up operations for last stage
        warmups_ops_last_stage = (
            self.n_local_stages - 1
        ) * self.microbatches_per_round
        # Increment warmup for each hop away from the last stage. PyTorch uses
        # factor 2 for interleaved 1F1B and factor 1 for interleaved ZB1P.
        multiply_factor = 1 if enable_zero_bubble else 2
        warmup_ops = warmups_ops_last_stage + multiply_factor * (
            (self.pp_group_size - 1) - rank
        )

        # We cannot have more warmup operations than there are number of microbatches, so cap it there
        return min(warmup_ops, self._n_microbatches * self.n_local_stages)

    warmup_ops = get_rank_warmup_ops(rank)
    microbatch_ops = self.n_local_stages * self._n_microbatches
    # fwd_bwd_ops should encompass the remaining forwards
    fwd_bwd_ops = microbatch_ops - warmup_ops
    # cooldown_ops should encompass the remaining backwards
    cooldown_ops = microbatch_ops - fwd_bwd_ops
    # total ops encompass both forward and backward ops
    total_ops = warmup_ops + fwd_bwd_ops + cooldown_ops
    # warmup_ops + fwd_bwd_ops * 2 + cooldown_ops == microbatch_ops * 2
    logger.debug(
        "rank %s, warmup_ops %s, 1f1b %s, cooldown_ops %s total_ops %s",
        rank,
        warmup_ops,
        fwd_bwd_ops,
        cooldown_ops,
        total_ops,
    )

    # Calculates the stage index based on step and pp_group_size
    def forward_stage_index(step):
        # Get the local index from 0 to n_local_stages-1
        local_index = (step // self.microbatches_per_round) % self.n_local_stages
        return (local_index * self.pp_group_size) + rank

    def backward_stage_index(step):
        local_index = (
            self.n_local_stages
            - 1
            - ((step - warmup_ops) // self.microbatches_per_round)
            % self.n_local_stages
        )
        return (local_index * self.pp_group_size) + rank

    return _get_1f1b_rank_ops(
        self.n_local_stages,
        self.pp_group_size,
        warmup_ops,
        fwd_bwd_ops,
        cooldown_ops,
        rank,
        forward_stage_index,
        backward_stage_index,
        num_1f1b_microbatches=rank if enable_zero_bubble else 0,
        enable_zero_bubble=enable_zero_bubble,
    )
