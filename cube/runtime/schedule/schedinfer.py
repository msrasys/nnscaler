from typing import Callable, Iterable, List, Optional
import torch

from cube.runtime.schedule.strategy import ScheduleABC


class ScheduleInfer(ScheduleABC):

    @staticmethod
    def run(segment: Callable, # forward body
            rfadapter: Optional[Callable],  # recv forward adapter
            sfadapter: Optional[Callable],  # send forward adapter
            dataloader: Iterable,
            num_microbatch: int):
        
        for _ in range(num_microbatch):
            # recv forward
            inputs = ScheduleInfer.adapter_step(rfadapter, False)
            inputs = ScheduleInfer.dataloader_step(dataloader) if inputs == (None,) else inputs
            # forward
            outputs = ScheduleInfer.forward_step(segment, *inputs)
            # send forward
            ScheduleInfer.adapter_step(sfadapter, True, *outputs)

        ScheduleInfer.assert_empty()
