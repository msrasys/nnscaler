"""
Common scheduling descriptions
"""

from typing import List

from cube.graph.schedule.schedplan import SchedulePlan
from cube.graph.graph import IRGraph
from cube.graph.segment import IRSegment


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
        segments: List[IRSegment] = graph.select(ntype=IRSegment, flatten=False)
        fsegs = [seg for seg in segments if seg.isfw()]
        assert len(fsegs) == num_stages, f"Mismatch of forward segement number ({len(fsegs)}) with num_stages ({len(num_stages)})"

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
        segments: List[IRSegment] = graph.select(ntype=IRSegment, flatten=False)
        fsegs = [seg for seg in segments if seg.isfw()]
        assert len(fsegs) == num_stages, "Mismatch of forward segement number with num_stages"
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
        assert len(fsegs) == num_stages, "Mismatch of forward segement number with num_stages"
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
