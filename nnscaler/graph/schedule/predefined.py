#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""
Common scheduling descriptions
"""

from collections import deque
from typing import Dict, List, Sequence

from nnscaler.graph.schedule.schedplan import (
    Block,
    PipelineAction,
    ScheduleDependency,
    SchedulePlan,
)
from nnscaler.graph.graph import IRGraph
from nnscaler.graph.segment import IRSegment


def _schedule_local_actions(
    graph: IRGraph,
    num_microbatches: int,
    local_actions: Sequence[Sequence[Block]],
) -> SchedulePlan:
    """Build the earliest valid global plan for fixed per-rank action orders."""
    actions = [block for sequence in local_actions for block in sequence]
    predecessors = {block: set() for block in actions}
    order = {block: index for index, block in enumerate(actions)}

    for sequence in local_actions:
        for prev, next_block in zip(sequence, sequence[1:]):
            predecessors[next_block].add(prev)

    dependency = ScheduleDependency(graph)
    for prev in actions:
        for next_block in actions:
            if dependency.depends(prev, next_block):
                predecessors[next_block].add(prev)

    start_steps: Dict[Block, int] = {}
    remaining = set(actions)
    while remaining:
        ready = [
            block for block in remaining
            if predecessors[block].issubset(start_steps)
        ]
        if not ready:
            unresolved = sorted(remaining, key=order.__getitem__)
            raise RuntimeError(
                f'Pipeline action order contains a dependency cycle: {unresolved[:8]}'
            )
        for block in sorted(ready, key=order.__getitem__):
            start_steps[block] = max(
                (start_steps[pred] + pred.span for pred in predecessors[block]),
                default=0,
            )
            remaining.remove(block)

    sched = SchedulePlan(graph, num_microbatches)
    for block in sorted(actions, key=lambda item: (start_steps[item], order[item])):
        sched.add_segment(
            block.content,
            block.mid,
            start_steps[block],
            block.span,
            block.stream_context,
            block.action,
        )
    sched.finish()
    return sched


class PredefinedSched:

    @staticmethod
    def sched_1f1b(graph: IRGraph, num_microbatches: int, num_stages: int) -> SchedulePlan:
        """
        1F1B scheduling. The graph should be staged into segments.

        An illustration of scheduling schema (the number is micro-batch index):
        ```
        f0    f1    f2    | f3 b0 |    b1    b2    b3
           f0    f1    f2 | b0 f3 | b1    b2    b3
              f0    f1 b0 | f2 b1 | f3 b2    b3
                 f0 b0 f1 | b1 f2 | b2 f3 b3
        ```
        """
        if num_microbatches <= 0:
            raise ValueError(f"expected num_microbatches > 0, but got {num_microbatches} ")
        segments: List[IRSegment] = graph.select(ntype=IRSegment, flatten=False)
        fsegs = [seg for seg in segments if seg.isfw()]
        assert len(fsegs) == num_stages, f"Mismatch of forward segment number ({len(fsegs)}) with num_stages ({num_stages})"

        # describe schedule
        sched = SchedulePlan(graph, num_microbatches)

        wait_steps = [sid for sid in range(num_stages)]
        bw_ofst = [num_stages - 1 - sid for sid in range(num_stages)]
        total_steps = num_microbatches * 2 + (num_stages - 1) * 2

        for step in range(total_steps):
            for sid in range(num_stages):
                ofst = wait_steps[sid]
                if step < ofst: continue
                fw_idx = (step - ofst) // 2
                # forward or backward segment
                segment = fsegs[sid] if (step - ofst) % 2 == 0 else fsegs[sid].mirror
                mb_idx = fw_idx if (step - ofst) % 2 == 0 else fw_idx - bw_ofst[sid]
                # append for execution
                if mb_idx < 0 or mb_idx >= num_microbatches: continue
                sched.add_segment(segment, mb_idx, step)
        sched.finish()
        return sched

    @staticmethod
    def sched_1f1b_interleaved(
        graph: IRGraph,
        num_microbatches: int,
        num_stages: int,
        *,
        _bind_graph: bool = True,
    ) -> SchedulePlan:
        """
        1F1B interleaved scheduling. The graph should be staged into segments. You can refer to the paper
        [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/pdf/2104.04473)
        for more details. Different from the 1f1b scheduling where each pipeline device group corresponds to exactly
        one forward segment and its backward segment, in 1F1B interleaved scheduling, each pipeline device group
        maintains multiple forward segments and their corresponding backward segments.

        Notations:
        - `n`: number of pipeline device groups
        - `m`: number of pipeline stages (model is split into m parts)
        - `k`: number of local stages in each pipeline device group, thus `m = n * k`
        - `p`: number of micro-batches in a training step, currently constrained to be a multiple of n
        - `q`: in interleaved 1f1b, p is divided into q groups, each group contains n micro-batches, thus `p = n * q`
        - (x)F(y) denotes the x-th forward segment with micro-batch y, B denotes the backward segment.

        Therefore:
        - each pipeline rank runs k * p * 2 steps in total (will run `k` forward and `k` backward for each micro-batch).
        - i-th segment is placed at *i mod n*-th device group
        - if i mod n = j mod n, then i-th segment and j-th segment should be placed at the same device group.

        Furthermore, the paper also assumes that all forward segments have a similar execution time, and all backward
        segments have a similar execution time.

        For 1f1b like schedule, computation process on each rank is divided into three parts:
        1. warmup: composed of forward stages
        2. steady: a list of pairs of a forward stage and a backward stage
        3. cooldown: composed of backward stages

        Here is an example with 4 devices, 8 stages, and 4 micro-batches. Note that in this schedule representation,
        the steady part is different from the formulation that is currently used in nnScaler.
        0F0 0F1 0F2 0F3 4F0     4F1     4F2     | 4F3 4B0 |     4B1     4B2     4B3 0B0 0B1 0B2 0B3
            1F0 1F1 1F2 1F3 5F0     5F1     5F2 | 5B0 5F3 | 5B1     5B2     5B3 1B0 1B1 1B2 1B3
                2F0 2F1 2F2 2F3 6F0     6F1 6B0 | 6F2 6B1 | 6F3 6B2     6B3 2B0 2B1 2B2 2B3
                    3F0 3F1 3F2 3F3 7F0 7B0 7F1 | 7B1 7F2 | 7B2 7F3 7B3 3B0 3B1 3B2 3B3

        In modern LLMs, the backward segments takes more time than the forward segments. As a result,
        the schedule can be adjusted like that in Megatron-LM (this schedule is clearer and easier to
        calculate the end to end span). In addition, `4F3` above is executed much earlier in the runtime.
        As a result, we prefer to make the schedule closer to the real execution order. The schedule is like:
        0F0 0F1 0F2 0F3 4F0 4F1 4F2 4F3                         4B0 4B1 4B2 4B3 0B0 0B1 0B2 0B3
            1F0 1F1 1F2 1F3 5F0 5F1 5F2 5F3             5B0     5B1 5B2 5B3 1B0 1B1 1B2 1B3
                2F0 2F1 2F2 2F3 6F0 6F1     6F2 6B0 6F3 6B1     6B2 6B3 2B0 2B1 2B2 2B3
                    3F0 3F1 3F2 3F3 7F0 7B0 7F1 7B1 7F2 7B2 7F3 7B3 3B0 3B1 3B2 3B3
        In this representation, #step for the 3 parts in each rank is:
        | rank | warmup | steady | cooldown |
        | 0    | 8      | 0      | 8        |
        | 1    | 8      | 0      | 8        |
        | 2    | 6      | 4      | 6        |
        | 3    | 4      | 8      | 4        |
        There is a subtle difference between the two schedules on memory usage on rank 1 and rank 2. However, the order
        of the difference is a small constant (1 forward stage's memory footprint). Considering the memory is bounded
        by the first device group, we can omit the difference for now.

        In torch, another schedule representation is used, which is equivalent to the Megatron-LM schedule.
        Note the blank step between 3F3 and 7F0 will be 'squeezed' in runtime.
        0F0 0F1 0F2 0F3 4F0 4F1 4F2 4F3                             4B0 4B1 4B2 4B3 0B0 0B1 0B2 0B3 
            1F0 1F1 1F2 1F3 5F0 5F1 5F2 5F3                 5B0     5B1 5B2 5B3 1B0 1B1 1B2 1B3 
                2F0 2F1 2F2 2F3 6F0 6F1         6F2 6B0 6F3 6B1     6B2 6B3 2B0 2B1 2B2 2B3 
                    3F0 3F1 3F2 3F3     7F0 7B0 7F1 7B1 7F2 7B2 7F3 7B3 3B0 3B1 3B2 3B3

        Here is another example when num_microbatches is 8:
        0F0 0F1 0F2 0F3 4F0 4F1 4F2 4F3 0F4 0F5                 0F6 4B0 0F7 4B1 4F4 4B2 4F5 4B3 4F6 0B0 4F7 0B1     0B2     0B3     4B4 4B5 4B6 4B7 0B4 0B5 0B6 0B7 
            1F0 1F1 1F2 1F3 5F0 5F1 5F2 5F3             1F4 5B0 1F5 5B1 1F6 5B2 1F7 5B3 5F4 1B0 5F5 1B1 5F6 1B2 5F7 1B3     5B4     5B5 5B6 5B7 1B4 1B5 1B6 1B7 
                2F0 2F1 2F2 2F3 6F0 6F1         6F2 6B0 6F3 6B1 2F4 6B2 2F5 6B3 2F6 2B0 2F7 2B1 6F4 2B2 6F5 2B3 6F6 6B4 6F7 6B5     6B6 6B7 2B4 2B5 2B6 2B7 
                    3F0 3F1 3F2 3F3     7F0 7B0 7F1 7B1 7F2 7B2 7F3 7B3 3F4 3B0 3F5 3B1 3F6 3B2 3F7 3B3 7F4 7B4 7F5 7B5 7F6 7B6 7F7 7B7 3B4 3B5 3B6 3B7
        In this setting, #step for the 3 parts in each rank is:
        | rank | warmup | steady | cooldown |
        | 0    | 10     | 12     | 10       |
        | 1    | 8      | 16     | 8        |
        | 2    | 6      | 20     | 6        |
        | 3    | 4      | 24     | 4        |

        Based on the example above, we can deduce the whole schedule from the last rank.
        For the last pipeline rank, the steady part starts as long as it receives the last forward stage for the
        0-th micro-batch (we index from 0). It is easy to calculate that the last rank's warmup part takes n * (k - 1)
        steps.
        After the warmup part, the steady part begins:
        - in the 0th round, it executes the (k-1)th stage's forward and backward stage for 0th micro batch groups
        - in the 1st round, it executes the 0th stage's forward for 1st micro batch group and (k-2)th stage's backward for 0th micro batch group
        - in the 2nd round, it executes the 1st stage's forward for 1st micro batch group and (k-3)th stage's backward for 0th micro batch group
        - ...
        - in the kth round, it executes the (k-1)th stage's forward and backward for 1st micro batch group
        - ...
        - in the ((q-1) * k)th round, it executes the (k-1)th stage's forward and backward for (q-1)th micro batch group
        In all, the steady part takes ((q-1) * k + 1) * n * 2 steps.
        The cooldown part for the last rank is symmetric to the warmup part. It takes n * (k - 1) steps to execute the backward
        stage for the last micro-batch group on 0-th to (k-2)-th stages.

        Based on the analysis of the last rank, we can deduce the execution order for remaining ranks. For example, for the
        (n-2)th rank. The steady part takes 2 less 1f1b pairs than the last rank. Since
        - it depends on the backward stage in 0-th 1f1b pair finishes on the last rank
        - the forward stage finishes one step earlier than the last rank
        As a result, there will be
        - 2 additional forward steps in the warmup part to provide the data that 0th and 1st 1f1b pair need for the last rank
        - 2 additional backward steps in the cooldown part to consume the data that last and (last-1)th 1f1b pair produce for the last rank

        In general, for the i-th rank:
        - the warmup part takes min(n * (k - 1) + 2 * (n - 1 - i), p * k) steps for forward computation
        - the steady part is composed of (p * k - warmup_steps) 1f1b pairs
        - the cooldown_steps equals to warmup_steps for backward computation
        """
        if num_microbatches <= 0:
            raise ValueError(f"expected num_microbatches > 0, but got {num_microbatches} ")
        segments: List[IRSegment] = graph.select(ntype=IRSegment, flatten=False)
        fsegs = [seg for seg in segments if seg.isfw()]
        assert len(fsegs) == num_stages, f"Mismatch of forward segment number ({len(fsegs)}) with num_stages ({num_stages})"
        # collect segments by device assignment info
        devs2segs = {}
        for seg in fsegs:
            cur_devs = tuple(seg.device)
            for devs in devs2segs.keys():
                for dev in devs:
                    if dev in cur_devs:
                        assert devs == cur_devs, f"find illegal device assignment: {devs} vs {cur_devs} in 1f1b interleaved scheduling"
            devs2segs.setdefault(cur_devs, []).append(seg)
        assert num_microbatches % len(devs2segs) == 0, f"num_microbatches: {num_microbatches} should be a multiple of the number of pipeline groups: {len(devs2segs)}"

        sched = SchedulePlan(graph, num_microbatches, bind_graph=_bind_graph)
        # an adapter class to fit in torch's implementation
        class ScheduleInfo:
            def __init__(self, pp_group_size, num_stages, num_micro_batch):
                self.pp_group_size = pp_group_size
                self.n_local_stages = num_stages // pp_group_size
                self.num_of_rounds = max(1, num_micro_batch // pp_group_size)
                self.microbatches_per_round = num_micro_batch // self.num_of_rounds
                self._n_microbatches = num_micro_batch
                assert num_micro_batch % self.num_of_rounds == 0

        from nnscaler.graph.schedule.interleaved_1f1b import _calculate_single_rank_operations
        pp_group_size = len(devs2segs)
        schedule_info = ScheduleInfo(pp_group_size, num_stages, num_microbatches)
        for rank in range(pp_group_size):
            rank_ops = _calculate_single_rank_operations(schedule_info, rank)
            for step, op in enumerate(rank_ops):
                # use None to represent the blank step
                if op is None: continue
                seg = fsegs[op.stage_index]
                if str(op.computation_type) == 'B':
                    seg = seg.mirror
                sched.add_segment(seg, op.microbatch_index, step)

        sched.finish()
        return sched

    @staticmethod
    def sched_1f1b_interleaved_fbw(
        graph: IRGraph,
        num_microbatches: int,
        num_stages: int,
    ) -> SchedulePlan:
        """Interleaved 1F1B expressed as explicit F/I/W actions.

        This intentionally preserves the existing local execution order: every
        full backward is replaced by adjacent I and W actions.  It is the
        representation bridge used before Zero-Bubble starts moving W actions.
        """
        base = PredefinedSched.sched_1f1b_interleaved(
            graph, num_microbatches, num_stages, _bind_graph=False
        )
        sched = SchedulePlan(graph, num_microbatches)
        for block in base.all_blocks():
            step = base.start(block) * 2
            if block.content.isfw():
                sched.add_segment(
                    block.content,
                    block.mid,
                    step,
                    block.span,
                    block.stream_context,
                    PipelineAction.FORWARD,
                )
            else:
                sched.add_segment(
                    block.content,
                    block.mid,
                    step,
                    block.span,
                    block.stream_context,
                    PipelineAction.BACKWARD_INPUT,
                )
                sched.add_segment(
                    block.content,
                    block.mid,
                    step + block.span,
                    block.span,
                    block.stream_context,
                    PipelineAction.BACKWARD_WEIGHT,
                )
        sched.finish()
        return sched

    @staticmethod
    def sched_1f1b_interleaved_zero_bubble_steady(
        graph: IRGraph,
        num_microbatches: int,
        num_stages: int,
    ) -> SchedulePlan:
        """Interleaved FBW with delayed W actions in the steady phase only.

        The delay on physical rank ``r`` is ``r`` input-backward actions, as in
        PyTorch's interleaved Zero-Bubble schedule. Pending steady-phase W work
        is drained before cooldown, whose I/W pairs remain adjacent here.
        """
        base = PredefinedSched.sched_1f1b_interleaved(
            graph, num_microbatches, num_stages, _bind_graph=False
        )
        device_groups = sorted({tuple(block.device) for block in base.all_blocks()})
        local_actions = []

        for rank, devices in enumerate(device_groups):
            base_blocks = sorted(
                (block for block in base.all_blocks() if tuple(block.device) == devices),
                key=base.start,
            )
            block_at_step = {base.start(block): block for block in base_blocks}
            sequence = []
            pending_weights = deque()

            def append_action(block: Block, action: PipelineAction) -> None:
                sequence.append(Block(
                    block.content,
                    block.mid,
                    block.span,
                    block.stream_context,
                    action,
                ))

            def drain_pending_weights() -> None:
                while pending_weights:
                    append_action(
                        pending_weights.popleft(),
                        PipelineAction.BACKWARD_WEIGHT,
                    )

            for block in base_blocks:
                if block.content.isfw():
                    append_action(block, PipelineAction.FORWARD)
                    continue

                step = base.start(block)
                previous = block_at_step.get(step - 1)
                in_steady_phase = previous is not None and previous.content.isfw()
                if not in_steady_phase:
                    drain_pending_weights()
                    append_action(block, PipelineAction.BACKWARD_INPUT)
                    append_action(block, PipelineAction.BACKWARD_WEIGHT)
                    continue

                append_action(block, PipelineAction.BACKWARD_INPUT)
                pending_weights.append(block)
                if len(pending_weights) > rank:
                    append_action(
                        pending_weights.popleft(),
                        PipelineAction.BACKWARD_WEIGHT,
                    )

            drain_pending_weights()
            local_actions.append(sequence)

        return _schedule_local_actions(graph, num_microbatches, local_actions)

    @staticmethod
    def sched_1f1b_interleaved_zero_bubble(
        graph: IRGraph,
        num_microbatches: int,
        num_stages: int,
        *,
        max_pending_weight_backwards: int | None = None,
    ) -> SchedulePlan:
        """Interleaved ZB1P schedule with F/I/W actions in every phase.

        Compared with interleaved 1F1B, each rank starts input backward after
        one rather than two warmup forwards per remaining physical stage.
        Weight backward is delayed by ``rank`` input-backward actions and fills
        otherwise idle slots through steady state and cooldown.
        """
        if num_microbatches <= 0:
            raise ValueError(
                f'expected num_microbatches > 0, but got {num_microbatches}'
            )
        if (max_pending_weight_backwards is not None
                and max_pending_weight_backwards <= 0):
            raise ValueError(
                'max_pending_weight_backwards must be > 0, got '
                f'{max_pending_weight_backwards}'
            )

        segments: List[IRSegment] = graph.select(
            ntype=IRSegment, flatten=False
        )
        fsegs = [segment for segment in segments if segment.isfw()]
        if len(fsegs) != num_stages:
            raise ValueError(
                f'Mismatch of forward segment number ({len(fsegs)}) '
                f'with num_stages ({num_stages})'
            )

        device_groups = sorted({tuple(segment.device) for segment in fsegs})
        pp_group_size = len(device_groups)
        if num_stages % pp_group_size != 0:
            raise ValueError(
                f'num_stages ({num_stages}) must be divisible by the number '
                f'of pipeline groups ({pp_group_size})'
            )
        if num_microbatches % pp_group_size != 0:
            raise ValueError(
                f'num_microbatches ({num_microbatches}) must be a multiple '
                f'of the number of pipeline groups ({pp_group_size})'
            )

        from nnscaler.graph.schedule.interleaved_1f1b import (
            BACKWARD_INPUT,
            BACKWARD_WEIGHT,
            FORWARD,
            _get_1f1b_rank_ops,
        )

        n_local_stages = num_stages // pp_group_size
        num_rounds = max(1, num_microbatches // pp_group_size)
        microbatches_per_round = num_microbatches // num_rounds
        microbatch_ops = n_local_stages * num_microbatches
        action_map = {
            FORWARD: PipelineAction.FORWARD,
            BACKWARD_INPUT: PipelineAction.BACKWARD_INPUT,
            BACKWARD_WEIGHT: PipelineAction.BACKWARD_WEIGHT,
        }
        local_actions = []

        for rank, devices in enumerate(device_groups):
            warmup_ops = min(
                (n_local_stages - 1) * microbatches_per_round
                + pp_group_size - 1 - rank,
                microbatch_ops,
            )
            fwd_bwd_ops = microbatch_ops - warmup_ops
            cooldown_ops = warmup_ops

            def forward_stage_index(step: int) -> int:
                local_index = (
                    step // microbatches_per_round
                ) % n_local_stages
                return local_index * pp_group_size + rank

            def backward_stage_index(step: int) -> int:
                local_index = (
                    n_local_stages
                    - 1
                    - ((step - warmup_ops) // microbatches_per_round)
                    % n_local_stages
                )
                return local_index * pp_group_size + rank

            weight_backward_delay = rank
            if max_pending_weight_backwards is not None:
                # A delay of d produces at most d + 1 pending I states before
                # the first W. Cap that queue for models whose retained
                # dWeight operands dominate pipeline memory.
                weight_backward_delay = min(
                    weight_backward_delay,
                    max_pending_weight_backwards - 1,
                )

            rank_ops = _get_1f1b_rank_ops(
                n_local_stages,
                pp_group_size,
                warmup_ops,
                fwd_bwd_ops,
                cooldown_ops,
                rank,
                forward_stage_index,
                backward_stage_index,
                num_1f1b_microbatches=weight_backward_delay,
                enable_zero_bubble=True,
            )
            sequence = []
            for op in rank_ops:
                if op is None:
                    continue
                segment = fsegs[op.stage_index]
                action = action_map[op.computation_type]
                if action != PipelineAction.FORWARD:
                    segment = segment.mirror
                if tuple(segment.device) != devices:
                    raise ValueError(
                        f'stage {op.stage_index} is assigned to '
                        f'{tuple(segment.device)}, expected {devices}'
                    )
                sequence.append(Block(
                    segment,
                    op.microbatch_index,
                    1,
                    action=action,
                ))
            local_actions.append(sequence)

        return _schedule_local_actions(graph, num_microbatches, local_actions)

    @staticmethod
    def sched_1f1b_plus(graph: IRGraph, num_microbatches: int, num_stages: int) -> SchedulePlan:
        """1F1B Plus Scheduling.

        f0 f0    f1 f1    f2 f2    | f3 f3 b0 |    b1    b2    b3
        f0    f0 f1    f1 f2    f2 | f3 b0 f3 | b1    b2    b3
        f0       f1 f0    f2 f1 b0 | f3 f2 b1 | f3 b2    b3
        f0       f1    f0 f2 b0 f1 | f3 b1 f2 | b2 f3 b3
        """
        if num_microbatches <= 0:
            raise ValueError(f"expected num_microbatches > 0, but got {num_microbatches} ")
        segments: List[IRSegment] = graph.select(ntype=IRSegment, flatten=False)
        fsegs = [seg for seg in segments if seg.isfw()]
        tp_fsegs = [seg for seg in fsegs if len(seg.device) == len(graph.device)]
        fb_fsegs = [seg for seg in fsegs if seg not in tp_fsegs]
        assert len(fb_fsegs) == num_stages, f"got only {len(fb_fsegs)} stages but need {num_stages} stages"
        assert all(tuple(seg.device) == tuple(graph.device) for seg in tp_fsegs)

        # describe schedule
        sched = SchedulePlan(graph, num_microbatches)

        wait_steps = [sid for sid in range(num_stages)]
        bw_ofst = [num_stages - 1 - sid for sid in range(num_stages)]
        total_steps = num_microbatches * 2 + (num_stages - 1) * 2

        # 1f1b schedule
        for step in range(total_steps):
            for sid in range(num_stages):
                ofst = wait_steps[sid]
                if step < ofst: continue
                fw_idx = (step - ofst) // 2
                # forward or backward segment
                segment = fb_fsegs[sid] if (step - ofst) % 2 == 0 else fb_fsegs[sid].mirror
                mb_idx = fw_idx if (step - ofst) % 2 == 0 else fw_idx - bw_ofst[sid]
                # append for execution
                if mb_idx < 0 or mb_idx >= num_microbatches: continue
                sched.add_segment(segment, mb_idx, step)

        # insert
        for mid in range(num_microbatches):
            for tp_seg in tp_fsegs:
                # TODO: not work case: tp_seg at tail fsegs
                next_seg = fsegs[fsegs.index(tp_seg)+1]
                assert next_seg in fsegs
                insert_fw, insert_bw = False, tp_seg.mirror is None
                if tp_seg.mirror is not None:
                    assert next_seg.mirror is not None

                for step in range(sched.nsteps-1, -1, -1):
                    segments = [blk.content for blk in sched.segments(step) if blk.mid == mid]
                    # insert forward
                    if next_seg in segments:
                        sched.insert_step(step, tp_seg, mid, 1)
                        assert not insert_fw
                        insert_fw = True
                    # insert backward
                    if next_seg.mirror in segments:
                        sched.insert_step(step+1, tp_seg.mirror, mid, 1)
                        assert not insert_bw
                        insert_bw = True
                    if insert_fw and insert_bw: break

                assert insert_fw and insert_bw, (
                    f'find one segment cannot be inserted in schedplan: ',
                    f'mid: {mid}, fw: {insert_fw}, bs: {insert_bw}')

        sched.finish()
        # print(sched)
        return sched

    @staticmethod
    def sched_gpipe(graph: IRGraph, num_microbatches: int, num_stages: int) -> SchedulePlan:
        """
        GPipe scheduling. The graph should be staged into segments.

        An illustration of scheduling schema (the number is micro-batch index):
        ```
        f0 f1 f2 f3                   b0 b1 b2 b3
           f0 f1 f2 f3             b0 b1 b2 b3
              f0 f1 f2 f3       b0 b1 b2 b3
                 f0 f1 f2 f3 b0 b1 b2 b3
        ```
        """
        if num_microbatches <= 0:
            raise ValueError(f"expected num_microbatches > 0, but got {num_microbatches} ")
        segments: List[IRSegment] = graph.select(ntype=IRSegment, flatten=False)
        fsegs = [seg for seg in segments if seg.isfw()]
        assert len(fsegs) == num_stages, "Mismatch of forward segment number with num_stages"
        # describe schedule
        sched = SchedulePlan(graph, num_microbatches)

        fwait_steps = [sid for sid in range(num_stages)]
        bwait_steps = [num_stages - 1 - sid for sid in range(num_stages)]

        total_steps = num_microbatches * 2 + (num_stages - 1) * 2
        middle_step = total_steps // 2
        for step in range(total_steps):
            for sid in range(num_stages):
                segment = fsegs[sid] if step < middle_step else fsegs[sid].mirror
                mb_idx = step - fwait_steps[sid] if step < middle_step else step - middle_step - bwait_steps[sid]
                if mb_idx < 0 or mb_idx >= num_microbatches: continue
                sched.add_segment(segment, mb_idx, step)
        sched.finish()
        return sched

    @staticmethod
    def sched_chimera_direct(graph: IRGraph, num_microbatches: int, num_stages: int):
        """Chimera-direct scheduling.

        The graph should be staged into segments.

        An illustration of scheduling schema (the number is micro-batch index):
        ```
        f0    f1 f2 b2-b2 f3    b3-b3 b0-b0       b1-b1
           f0 f2 f1 f3    b2-b2 b0-b0 b3-b3 b1-b1
           f2 f0 f3 f1    b0-b0 b2-b2 b1-b1 b3-b3
        f2    f3 f0 b0-b0 f1    b1-b1 b2-b2       b3-b3

        0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 (-> steps)
        ```

        Note the f0 and f2 (step 0) should be considered to be one segment in graph.
        """
        if num_microbatches <= 0:
            raise ValueError(f"expected num_microbatches > 0, but got {num_microbatches} ")
        segments: List[IRSegment] = graph.select(ntype=IRSegment, flatten=False)
        fsegs = [seg for seg in segments if seg.isfw()]
        assert len(fsegs) == 4, f"Chimera-direct scheduling only applies for 4 segments, but {len(fsegs)} detected"
        sched = SchedulePlan(graph, num_microbatches)
        assert num_microbatches % 2 == 0
        mid = 0
        while mid < num_microbatches:
            ofst = 16 * (mid // 2)
            # first micro-batch
            sched.add_segment(fsegs[0], micro_batch_id=mid, step=max(0, 0+ofst-3))  # tight compact
            sched.add_segment(fsegs[1], micro_batch_id=mid, step=max(1, 1+ofst-3))  # tight compact
            sched.add_segment(fsegs[2], micro_batch_id=mid, step=2+ofst)
            sched.add_segment(fsegs[3], micro_batch_id=mid, step=3+ofst)
            sched.add_segment(fsegs[3].mirror, micro_batch_id=mid, step=4+ofst, span=2)
            sched.add_segment(fsegs[2].mirror, micro_batch_id=mid, step=6+ofst, span=2)
            sched.add_segment(fsegs[1].mirror, micro_batch_id=mid, step=8+ofst, span=2)
            sched.add_segment(fsegs[0].mirror, micro_batch_id=mid, step=10+ofst, span=2)
            # second micro-batch
            sched.add_segment(fsegs[0], micro_batch_id=mid+1, step=2+ofst)
            sched.add_segment(fsegs[1], micro_batch_id=mid+1, step=3+ofst)
            sched.add_segment(fsegs[2], micro_batch_id=mid+1, step=4+ofst)
            sched.add_segment(fsegs[3], micro_batch_id=mid+1, step=6+ofst)
            sched.add_segment(fsegs[3].mirror, micro_batch_id=mid+1, step=8+ofst, span=2)
            sched.add_segment(fsegs[2].mirror, micro_batch_id=mid+1, step=10+ofst, span=2)
            sched.add_segment(fsegs[1].mirror, micro_batch_id=mid+1, step=12+ofst, span=2)
            sched.add_segment(fsegs[0].mirror, micro_batch_id=mid+1, step=14+ofst, span=2)
            # update
            mid += 2
        sched.finish()
        return sched


    @staticmethod
    def sched_infer_pipe(graph: IRGraph, num_microbatches: int, num_stages: int) -> SchedulePlan:
        """
        Inference pipeline scheduling. The graph should be staged into segments.

        An illustration of scheduling schema (the number is micro-batch index):
        ```
        f0 f1 f2 f3
           f0 f1 f2 f3
              f0 f1 f2 f3
                 f0 f1 f2 f3
        ```
        """
        fsegs: List[IRSegment] = graph.select(ntype=IRSegment, flatten=False)
        assert all(seg.isfw() for seg in fsegs), f"Detect backward. The predefined scheduling only applies for inference"
        assert len(fsegs) == num_stages, "Mismatch of forward segment number with num_stages"
        # describe schedule
        sched = SchedulePlan(graph, num_microbatches)
        fwait_steps = [sid for sid in range(num_stages)]
        total_steps = num_microbatches + num_stages - 1
        for step in range(total_steps):
            for sid in range(num_stages):
                segment = fsegs[sid]
                mb_idx = step - fwait_steps[sid]
                if mb_idx < 0 or mb_idx >= num_microbatches: continue
                sched.add_segment(segment, mb_idx, step)
        sched.finish()
        return sched
