from typing import Any, Callable, Dict, Iterable, List
import torch

from cube.profiler.timer import CudaTimer


class ScheduleABC:

    status: Dict[str, List[torch.Tensor]] = dict()

    @staticmethod
    def forward_step(segment: Callable, *args, **kwargs):
        """
        forward pass
        """
        CudaTimer().start('forward')
        outputs = segment(*args, **kwargs)
        CudaTimer().stop('forward')
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        return outputs

    @staticmethod
    def backward_step(itensors: List[torch.Tensor],
                      otensors: List[torch.Tensor],
                      otensor_grads: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        backward pass
        """
        for tensor in itensors:
            if torch.is_tensor(tensor) and tensor.requires_grad:
                tensor.retain_grad()
        CudaTimer().start("backward")
        torch.autograd.backward(otensors, grad_tensors=otensor_grads)
        CudaTimer().stop("backward")
        itensor_grads = []
        for tensor in itensors:
            if torch.is_tensor(tensor) and tensor.requires_grad:
                itensor_grads.append(tensor.grad)
            else:
                itensor_grads.append(None)
        return tuple(itensor_grads)

    @staticmethod
    def dataloader_step(dataloader: Iterable):
        data = next(dataloader)
        if not isinstance(data, tuple):
            data = (data,)
        return data

    @staticmethod
    def adapter_step(adapter: Callable, *args):
        """
        adapter pass
        """
        if adapter is None: return ()
        CudaTimer().start('adapter')
        outputs = adapter(*args)
        CudaTimer().stop('adapter')
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        return outputs

    @staticmethod
    def exchange(sadapter: Callable, radapter: Callable, stage_id: int, *args):
        """
        send adapter and recv adapter
        """
        # TODO: optimize with batch operators
        if stage_id % 2 == 0:
            ScheduleABC.adapter_step(sadapter, *args)
            outs = ScheduleABC.adapter_step(radapter)
        else:
            outs = ScheduleABC.adapter_step(radapter)
            ScheduleABC.adapter_step(sadapter, *args)
        return outs

    @staticmethod
    def push_tail(name: str, val: Any):
        if name not in ScheduleABC.status:
            ScheduleABC.status[name] = []
        ScheduleABC.status[name].append(val)

    @staticmethod
    def push_head(name: str, val: Any):
        if name not in ScheduleABC.status:
            ScheduleABC.status[name] = []
        ScheduleABC.status[name].insert(0, val)

    @staticmethod
    def pop_head(name: str):
        assert name in ScheduleABC.status, f"{name} is empty"
        out = ScheduleABC.status[name].pop(-1)
        if len(ScheduleABC.status[name]) == 0:
            del ScheduleABC.status[name]
        return out

    @staticmethod
    def pop_tail(name: str):
        assert name in ScheduleABC.status, f"{name} is empty"
        out = ScheduleABC.status[name].pop(0)
        if len(ScheduleABC.status[name]) == 0:
            del ScheduleABC.status
        return out

    @staticmethod
    def assert_empty():
        assert len(ScheduleABC.status) == 0, f"status is not empty. Got field {list(ScheduleABC.status.keys())}"
