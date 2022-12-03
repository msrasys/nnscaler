"""
Schedule Plan tailored for AlphaFold

The scheduling follows forward-backward pattern.
In steady phase, each forward will perform a single forward at `recycle+1`
micro-batches, with one keeping activation while others no activation.

"""
from typing import Callable, Iterable, List, Tuple
from functools import partial
import torch

from cube.runtime.schedule.strategy import ScheduleABC


def first_stage_rfadapter(shapes: Tuple[List[int]], dtypes: Tuple[List[torch.dtype]], dataloader):
    return next(dataloader)

def last_stage_sfadapter(msa_repr: torch.Tensor, pair_repr: torch.Tensor):
    pass


class ScheduleNF1B(ScheduleABC):

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
            recycle: int,
            reducers: List[Callable]):

        assert num_microbatch >= num_stages

        # special case: num_stages == 1: use gradient accum
        if num_stages == 1:
            for _ in range(num_microbatch):
                inputs = ScheduleNF1B.dataloader_step(dataloader)
                for _ in range(recycle):
                    # FIXME: a simulation as output will be loss
                    with torch.no_grad():
                        _ = ScheduleNF1B.forward_step(segment, *inputs)
                outputs = ScheduleNF1B.forward_step(segment, *inputs)
                input_grads = ScheduleNF1B.backward_step(inputs, outputs, (None,))
            for reducer in reducers:
                reducer()
            return

        # =============================== recycle ====================================
        if stage_id == 0:
            assert rfadapter is None
            shapes, dtypes = [], []
            for data in ScheduleNF1B.dataloader_step(dataloader):
                shapes.append(list(data.size()))
                dtypes.append(data.dtype)
            rfadapter = partial(first_stage_rfadapter, shapes=shapes, dtypes=dtypes, dataloader=dataloader)
        # if stage_id == num_stages - 1:
        #     assert sfadapter is None
        #     sfadapter = last_stage_sfadapter    
        
        for rid in range(recycle):
            for mid in range(num_microbatch):
                # recv forward
                if stage_id == 0 and rid == 0:
                    inputs = ScheduleNF1B.dataloader_step(dataloader)
                else:
                    inputs = ScheduleNF1B.adapter_step(rfadapter, require_grad=(rid == recycle-1))
                # forward
                with torch.no_grad():
                    outputs = ScheduleNF1B.forward_step(segment, *inputs)
                # FIXME: a simulation
                if stage_id == num_stages - 1:
                    outputs = ScheduleNF1B.dataloader_step(dataloader)
                # send forward
                ScheduleNF1B.adapter_step(sfadapter, False, *outputs)
        # recv forward batches TODO: optmize with async
        datas = []
        if stage_id == 0:
            for mid in range(num_microbatch):
                inputs = ScheduleNF1B.adapter_step(rfadapter, require_grad=False)
                datas.append(inputs)
        # ==========================================================================
        
        # 1F1B schedule
        if stage_id == 0: rfadapter = None
        if stage_id == num_stages - 1: sfadapter = None
        num_warmup_microbatches = num_stages - 1 - stage_id
        num_warmup_remaining = num_microbatch - num_warmup_microbatches

        # warmup
        for _ in range(num_warmup_microbatches):
            # recv forward
            # print(f'rank[{torch.distributed.get_rank()}]: line26: recving forward')
            inputs = ScheduleNF1B.adapter_step(rfadapter, True)
            inputs = datas.pop(0) if inputs == (None,) else inputs
            # forward
            ScheduleNF1B.push_tail('inputs', inputs)
            outputs = ScheduleNF1B.forward_step(segment, *inputs)
            ScheduleNF1B.push_tail('outputs', outputs)
            # send forward
            # print(f'rank[{torch.distributed.get_rank()}]: line40 send forward')
            ScheduleNF1B.adapter_step(sfadapter, True, *outputs)

        if num_warmup_remaining > 0:
            # print(f'rank[{torch.distributed.get_rank()}]: line44 recv forward')
            inputs = ScheduleNF1B.adapter_step(rfadapter, True)
            inputs = datas.pop(0) if inputs == (None,) else inputs

        # steady
        for i in range(num_warmup_remaining):
            # forward
            ScheduleNF1B.push_tail('inputs', inputs)
            # print(f'rank[{torch.distributed.get_rank()}]: line 57 forward')
            outputs = ScheduleNF1B.forward_step(segment, *inputs)
            ScheduleNF1B.push_tail('outputs', outputs)
    
            # send forward recv backward
            # print(f'rank[{torch.distributed.get_rank()}]: line62 send forward recv backward')
            grads = ScheduleNF1B.exchange(sfadapter, rbadapter, stage_id, (True, False), *outputs)
            grads = (None,) if len(grads) == 0 else grads
    
            # backward
            inputs, outputs = ScheduleNF1B.pop_head('inputs'), ScheduleNF1B.pop_head('outputs')
            # print(f'rank[{torch.distributed.get_rank()}]: line71 backward')
            input_grads = ScheduleNF1B.backward_step(inputs, outputs, grads)

            # send backward recv forward
            if i != num_warmup_remaining - 1:
                # print(f'rank[{torch.distributed.get_rank()}]: line77 send backward recv forward')
                inputs = ScheduleNF1B.exchange(sbadapter, rfadapter, stage_id, (False, True), *input_grads)
                inputs = datas.pop(0) if inputs == (None,) else inputs
            else:
                # send backward
                # print(f'rank[{torch.distributed.get_rank()}]: line82 send backward')
                ScheduleNF1B.adapter_step(sbadapter, False, *input_grads)

        # cooldown
        for i in range(num_warmup_microbatches):
            inputs, outputs = ScheduleNF1B.pop_head('inputs'), ScheduleNF1B.pop_head('outputs')
            # recv backward
            # print(f'rank[{torch.distributed.get_rank()}]: line89 recv backward')
            grads = ScheduleNF1B.adapter_step(rbadapter, False)
            grads = (None,) if len(grads) == 0 else grads
            # backward
            # print(f'rank[{torch.distributed.get_rank()}]: line96 backward')
            input_grads = ScheduleNF1B.backward_step(inputs, outputs, grads)
            # send backward
            # print(f'rank[{torch.distributed.get_rank()}]: line99 send backward') 
            ScheduleNF1B.adapter_step(sbadapter, False, *input_grads)
        
        # allreduce gradient
        for reducer in reducers:
            reducer()

        assert len(datas) == 0
        ScheduleNF1B.assert_empty()
