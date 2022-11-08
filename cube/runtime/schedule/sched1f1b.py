from typing import Callable, Iterable, List
import torch

from cube.runtime.schedule.strategy import ScheduleABC


class Schedule1F1B(ScheduleABC):

    @staticmethod
    def run(segment: Callable,   # forward body
            rfadapter: Callable, # recv_forward adapter
            sfadapter: Callable, # send_forward adapter
            rbadapter: Callable, # recv_backward adapter
            sbadapter: Callable, # send_backward adapter
            dataloader: Iterable,
            stage_id: int,
            num_stages: int,
            num_microbatch: int,
            reducers: List[Callable], # weight reducers
            recompute=False):

        # special case: num_stages == 1: use gradient accum
        if num_stages == 1:
            for _ in range(num_microbatch):
                inputs = Schedule1F1B.dataloader_step(dataloader)
                outputs = Schedule1F1B.forward_step(segment, *inputs)
                input_grads = Schedule1F1B.backward_step(inputs, outputs, (None,))
            for reducer in reducers:
                reducer()
            return

        num_warmup_microbatches = num_stages - 1 - stage_id
        num_warmup_remaining = num_microbatch - num_warmup_microbatches

        # warmup
        for _ in range(num_warmup_microbatches):
            # recv forward
            # print(f'rank[{torch.distributed.get_rank()}]: line26: recving forward')
            inputs = Schedule1F1B.adapter_step(rfadapter, True)
            inputs = Schedule1F1B.dataloader_step(dataloader) if len(inputs) == 0 else inputs
            # forward
            Schedule1F1B.push_tail('inputs', inputs)
            if recompute:
                with torch.no_grad():
                    outputs = Schedule1F1B.forward_step(segment, *inputs)
                    Schedule1F1B.push_tail('outputs', None)
            else:
                # print(f'rank[{torch.distributed.get_rank()}]: line36: forward')
                outputs = Schedule1F1B.forward_step(segment, *inputs)
                Schedule1F1B.push_tail('outputs', outputs)
            # send forward
            # print(f'rank[{torch.distributed.get_rank()}]: line40 send forward')
            Schedule1F1B.adapter_step(sfadapter, True, *outputs)

        if num_warmup_remaining > 0:
            # print(f'rank[{torch.distributed.get_rank()}]: line44 recv forward')
            inputs = Schedule1F1B.adapter_step(rfadapter, True)
            inputs = Schedule1F1B.dataloader_step(dataloader) if len(inputs) == 0 else inputs

        # steady
        for i in range(num_warmup_remaining):
            # forward
            Schedule1F1B.push_tail('inputs', inputs)
            if recompute:
                with torch.no_grad():
                    outputs = Schedule1F1B.forward_step(segment, *inputs)
                    Schedule1F1B.push_tail('outputs', None)
            else:
                # print(f'rank[{torch.distributed.get_rank()}]: line 57 forward')
                outputs = Schedule1F1B.forward_step(segment, *inputs)
                Schedule1F1B.push_tail('outputs', outputs)
    
            # send forward recv backward
            # print(f'rank[{torch.distributed.get_rank()}]: line62 send forward recv backward')
            grads = Schedule1F1B.exchange(sfadapter, rbadapter, stage_id, (True, False), *outputs)
            grads = (None,) if len(grads) == 0 else grads
    
            # backward
            inputs, outputs = Schedule1F1B.pop_head('inputs'), Schedule1F1B.pop_head('outputs')
            if recompute:
                assert outputs is None
                outputs = Schedule1F1B.forward_step(segment, *inputs)
            # print(f'rank[{torch.distributed.get_rank()}]: line71 backward')
            input_grads = Schedule1F1B.backward_step(inputs, outputs, grads)

            # send backward recv forward
            if i != num_warmup_remaining - 1:
                # print(f'rank[{torch.distributed.get_rank()}]: line77 send backward recv forward')
                inputs = Schedule1F1B.exchange(sbadapter, rfadapter, stage_id, (False, True), *input_grads)
                inputs = Schedule1F1B.dataloader_step(dataloader) if len(inputs) == 0 else inputs
            else:
                # send backward
                # print(f'rank[{torch.distributed.get_rank()}]: line82 send backward')
                Schedule1F1B.adapter_step(sbadapter, False, *input_grads)

        # cooldown
        for i in range(num_warmup_microbatches):
            inputs, outputs = Schedule1F1B.pop_head('inputs'), Schedule1F1B.pop_head('outputs')
            # recv backward
            # print(f'rank[{torch.distributed.get_rank()}]: line89 recv backward')
            grads = Schedule1F1B.adapter_step(rbadapter, False)
            grads = (None,) if len(grads) == 0 else grads
            # backward
            if recompute:
                assert outputs is None
                outputs = Schedule1F1B.forward_step(segment, *inputs)
            # print(f'rank[{torch.distributed.get_rank()}]: line96 backward')
            input_grads = Schedule1F1B.backward_step(inputs, outputs, grads)
            # send backward
            # print(f'rank[{torch.distributed.get_rank()}]: line99 send backward') 
            Schedule1F1B.adapter_step(sbadapter, False, *input_grads)
        
        # allreduce gradient
        for reducer in reducers:
            reducer()

        Schedule1F1B.assert_empty()
        # print(f'rank[{torch.distributed.get_rank()}]: ok here')
