#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""
Common scheduling descriptions
"""

from typing import List

from nnscaler.graph.schedule.schedplan import ScheduleAction, SchedulePlan
from nnscaler.graph.graph import IRGraph
from nnscaler.graph.segment import IRSegment


class PredefinedSched:

    @staticmethod
    def sched_zero_bubble(graph: IRGraph, num_microbatches: int, num_stages: int) -> SchedulePlan:
        """ZB1P scheduling with separate input- and weight-backward actions.

        Like interleaved 1F1B, each pipeline device group can own multiple
        local stages. Stage `i` is assigned to device group
        `i % pp_group_size`. Ready weight-backward work is used to fill slots
        that would otherwise be pipeline bubbles.

        In the diagram below, eight stages are interleaved over four device
        groups: group 0 owns stages 0 and 4, group 1 owns stages 1 and 5,
        group 2 owns stages 2 and 6, and group 3 owns stages 3 and 7. `xFy`,
        `xIy`, and `xWy` denote the forward, input-backward, and
        weight-backward actions of stage `x` on microbatch `y`. With 24
        microbatches, `num_rounds` is 6 and `microbatches_per_round` is 4:

        ```
        step:      0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35  36  37  38  39  40  41  42  43  44  45  46  47  48  49  50  51  52  53  54  55  56  57  58  59  60  61  62  63  64  65  66  67  68  69  70  71  72  73  74  75  76  77  78  79  80  81  82  83  84  85  86  87  88  89  90  91  92  93  94  95  96  97  98  99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146
        group 0: 0F0 0F1 0F2 0F3 4F0 4F1 4F2 --- --- --- 4F3 4I0 4W0 0F4 4I1 4W1 0F5 4I2 4W2 0F6 4I3 4W3 0F7 0I0 0W0 4F4 0I1 0W1 4F5 0I2 0W2 4F6 0I3 0W3 4F7 4I4 4W4 0F8 4I5 4W5 0F9 4I6 4W6 0F10 4I7 4W7 0F11 0I4 0W4 4F8 0I5 0W5 4F9 0I6 0W6 4F10 0I7 0W7 4F11 4I8 4W8 0F12 4I9 4W9 0F13 4I10 4W10 0F14 4I11 4W11 0F15 0I8 0W8 4F12 0I9 0W9 4F13 0I10 0W10 4F14 0I11 0W11 4F15 4I12 4W12 0F16 4I13 4W13 0F17 4I14 4W14 0F18 4I15 4W15 0F19 0I12 0W12 4F16 0I13 0W13 4F17 0I14 0W14 4F18 0I15 0W15 4F19 4I16 4W16 0F20 4I17 4W17 0F21 4I18 4W18 0F22 4I19 4W19 0F23 0I16 0W16 4F20 0I17 0W17 4F21 0I18 0W18 4F22 0I19 0W19 4F23 4I20 4W20 4I21 4W21 4I22 4W22 4I23 4W23 0I20 0W20 0I21 0W21 0I22 0W22 0I23 0W23
        group 1: --- 1F0 1F1 1F2 1F3 5F0 5F1 --- --- 5F2 5I0 5F3 5I1 5W0 1F4 5I2 5W1 1F5 5I3 5W2 1F6 1I0 5W3 1F7 1I1 1W0 5F4 1I2 1W1 5F5 1I3 1W2 5F6 5I4 1W3 5F7 5I5 5W4 1F8 5I6 5W5 1F9 5I7 5W6 1F10 1I4 5W7 1F11 1I5 1W4 5F8 1I6 1W5 5F9 1I7 1W6 5F10 5I8 1W7 5F11 5I9 5W8 1F12 5I10 5W9 1F13 5I11 5W10 1F14 1I8 5W11 1F15 1I9 1W8 5F12 1I10 1W9 5F13 1I11 1W10 5F14 5I12 1W11 5F15 5I13 5W12 1F16 5I14 5W13 1F17 5I15 5W14 1F18 1I12 5W15 1F19 1I13 1W12 5F16 1I14 1W13 5F17 1I15 1W14 5F18 5I16 1W15 5F19 5I17 5W16 1F20 5I18 5W17 1F21 5I19 5W18 1F22 1I16 5W19 1F23 1I17 1W16 5F20 1I18 1W17 5F21 1I19 1W18 5F22 5I20 1W19 5F23 5I21 5W20 5I22 5W21 5I23 5W22 1I20 5W23 1I21 1W20 1I22 1W21 1I23 1W22 1W23
        group 2: --- --- 2F0 2F1 2F2 2F3 6F0 --- 6F1 6I0 6F2 6I1 6F3 6I2 6W0 2F4 6I3 6W1 2F5 2I0 6W2 2F6 2I1 6W3 2F7 2I2 2W0 6F4 2I3 2W1 6F5 6I4 2W2 6F6 6I5 2W3 6F7 6I6 6W4 2F8 6I7 6W5 2F9 2I4 6W6 2F10 2I5 6W7 2F11 2I6 2W4 6F8 2I7 2W5 6F9 6I8 2W6 6F10 6I9 2W7 6F11 6I10 6W8 2F12 6I11 6W9 2F13 2I8 6W10 2F14 2I9 6W11 2F15 2I10 2W8 6F12 2I11 2W9 6F13 6I12 2W10 6F14 6I13 2W11 6F15 6I14 6W12 2F16 6I15 6W13 2F17 2I12 6W14 2F18 2I13 6W15 2F19 2I14 2W12 6F16 2I15 2W13 6F17 6I16 2W14 6F18 6I17 2W15 6F19 6I18 6W16 2F20 6I19 6W17 2F21 2I16 6W18 2F22 2I17 6W19 2F23 2I18 2W16 6F20 2I19 2W17 6F21 6I20 2W18 6F22 6I21 2W19 6F23 6I22 6W20 6I23 6W21 2I20 6W22 2I21 6W23 2I22 2W20 2I23 2W21 2W22 2W23
        group 3: --- --- --- 3F0 3F1 3F2 3F3 7F0 7I0 7F1 7I1 7F2 7I2 7F3 7I3 7W0 3F4 3I0 7W1 3F5 3I1 7W2 3F6 3I2 7W3 3F7 3I3 3W0 7F4 7I4 3W1 7F5 7I5 3W2 7F6 7I6 3W3 7F7 7I7 7W4 3F8 3I4 7W5 3F9 3I5 7W6 3F10 3I6 7W7 3F11 3I7 3W4 7F8 7I8 3W5 7F9 7I9 3W6 7F10 7I10 3W7 7F11 7I11 7W8 3F12 3I8 7W9 3F13 3I9 7W10 3F14 3I10 7W11 3F15 3I11 3W8 7F12 7I12 3W9 7F13 7I13 3W10 7F14 7I14 3W11 7F15 7I15 7W12 3F16 3I12 7W13 3F17 3I13 7W14 3F18 3I14 7W15 3F19 3I15 3W12 7F16 7I16 3W13 7F17 7I17 3W14 7F18 7I18 3W15 7F19 7I19 7W16 3F20 3I16 7W17 3F21 3I17 7W18 3F22 3I18 7W19 3F23 3I19 3W16 7F20 7I20 3W17 7F21 7I21 3W18 7F22 7I22 3W19 7F23 7I23 7W20 3I20 7W21 3I21 7W22 3I22 7W23 3I23 3W20 3W21 3W22 3W23
        ```

        Each group switches between its two local stages throughout the
        schedule. Ready weight-backward actions fill slots without delaying
        the input-gradient critical path.

        The schedule is constructed in two phases:

        1. Build a local ZB1P action order for every device group. The stream
           serializes all local stages on that group, prioritizes forward and
           input-backward actions, and uses ready weight-backward actions to
           fill available slots. Remaining weight-backward actions are drained
           at the end.
        2. Align the local orders into global steps while enforcing the pipeline
           dependencies. `F(s, m)` waits for `F(s - 1, m)`, and `I(s, m)` waits
           for `I(s + 1, m)`. In contrast, `W(s, m)` only waits for `I(s, m)`;
           it does not produce a gradient needed by another stage, so it can be
           moved away from the input-gradient critical path.

        Actions completed in the current global step become visible in the next
        step. This models one unit of execution time per action and prevents two
        dependent stages from being scheduled in the same step.
        """
        if num_microbatches <= 0:
            raise ValueError(f'expected num_microbatches > 0, but got {num_microbatches}')

        segments: List[IRSegment] = graph.select(ntype=IRSegment, flatten=False)
        fsegs = [seg for seg in segments if seg.isfw()]
        if len(fsegs) != num_stages:
            raise ValueError(
                f'Mismatch of forward segment number ({len(fsegs)}) with num_stages ({num_stages})'
            )
        devs2segs = {}
        for segment in fsegs:
            cur_devs = tuple(segment.device)
            if not cur_devs:
                raise ValueError(f'Pipeline stage {fsegs.index(segment)} has no assigned device')
            for devs in devs2segs:
                if set(devs) & set(cur_devs) and devs != cur_devs:
                    raise ValueError(f'find illegal device assignment: {devs} vs {cur_devs}')
            devs2segs.setdefault(cur_devs, []).append(segment)

        pp_group_size = len(devs2segs)
        if num_stages % pp_group_size != 0:
            raise ValueError(
                f'num_stages ({num_stages}) must be divisible by '
                f'pp_group_size ({pp_group_size})'
            )
        n_local_stages = num_stages // pp_group_size
        device_groups = list(devs2segs)
        for stage, segment in enumerate(fsegs):
            expected_group = device_groups[stage % pp_group_size]
            if tuple(segment.device) != expected_group:
                raise ValueError(
                    'zero_bubble requires interleaved round-robin stage placement: '
                    f'stage {stage} expected {expected_group}, got {tuple(segment.device)}'
                )

        num_rounds = max(1, num_microbatches // pp_group_size)
        if num_microbatches % num_rounds != 0:
            raise ValueError(
                f'num_microbatches ({num_microbatches}) must be divisible by '
                f'num_rounds ({num_rounds})'
            )

        class ScheduleInfo:
            def __init__(self):
                self.pp_group_size = pp_group_size
                self.n_local_stages = n_local_stages
                self.num_of_rounds = num_rounds
                # when `num_microbatches % pp_group_size == 0`,
                # `microbatches_per_round` will be `pp_group_size`
                # this can make sure
                # when the round-k in first stages of all device groups is done
                # the round-k+1 in the subsequent stages can start immediately
                # see above diagram for illustration
                # (0~3) is first round
                # (4~7) is second round, when 0F3 is done, the first stage of last device group will finish (3F0),
                # we can start 4F0 immediately
                #  step:      0   1   2   3   4   5   6 ...
                #   group 0: 0F0 0F1 0F2 0F3 4F0 4F1 4F2 ...
                self.microbatches_per_round = num_microbatches // num_rounds
                self._n_microbatches = num_microbatches

        from nnscaler.graph.schedule.interleaved_1f1b import (
            _add_bubbles_to_actions,
            _calculate_single_rank_operations,
        )

        # Per-group local action streams. Each entry is an _Action describing
        # F/I/W for one local stage and microbatch, or None for an intentional
        # bubble. These streams are dependency-aligned into global steps below.
        schedule_info = ScheduleInfo()
        actions = {
            rank: _calculate_single_rank_operations(
                schedule_info,
                rank,
                enable_zero_bubble=True,
            )
            for rank in range(pp_group_size)
        }
        pipeline_order = _add_bubbles_to_actions(
            actions,
            pp_group_size,
            num_stages,
        )

        schedule = SchedulePlan(graph, num_microbatches)
        for rank in range(pp_group_size):
            for step, op in enumerate(pipeline_order[rank]):
                if op is None:
                    continue

                action = str(op.computation_type)
                segment = fsegs[op.stage_index]
                schedule_action = None
                if action == 'I':
                    segment = segment.mirror
                    schedule_action = ScheduleAction.BACKWARD_INPUT
                elif action == 'W':
                    segment = segment.mirror
                    schedule_action = ScheduleAction.BACKWARD_WEIGHT
                schedule.add_segment(
                    segment,
                    op.microbatch_index,
                    step,
                    action=schedule_action,
                )

        schedule.finish()
        return schedule

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
    def sched_1f1b_interleaved(graph: IRGraph, num_microbatches: int, num_stages: int) -> SchedulePlan:
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

        sched = SchedulePlan(graph, num_microbatches)
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
