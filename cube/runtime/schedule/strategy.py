from typing import Any, Callable, Dict, Iterable, List
import torch

from cube.runtime.executor import AsyncCommHandler
from cube.flags import CompileFlag
from cube.profiler.timer import CudaTimer


class ScheduleABC:

    status: Dict[str, List[torch.Tensor]] = dict()

    @staticmethod
    def forward_step(segment: Callable, *args, **kwargs):
        """
        forward pass
        """
        args = ScheduleABC.sync_tensors(args)
        if not CompileFlag.async_comm:
            CudaTimer().start('forward')
        outputs = segment(*args, **kwargs)
        if not CompileFlag.async_comm:
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
        otensor_grads = ScheduleABC.sync_tensors(otensor_grads)
        for tensor in itensors:
            if torch.is_tensor(tensor) and tensor.requires_grad:
                tensor.retain_grad()
        if not CompileFlag.async_comm:
            CudaTimer().start("backward")
        otensors = [t for t in otensors if t.requires_grad]
        assert len(otensors) == len(otensor_grads), f"output tensor mismatches with gradient number"
        torch.autograd.backward(otensors, grad_tensors=otensor_grads)
        if not CompileFlag.async_comm:
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
    def adapter_step(adapter: Callable, require_grad : bool = True, *args):
        """
        Adapter pass.
        If the adapter is None, will return (None,)
        """
        if adapter is None: return (None,)
        # if adapter is None: return ()
        args = tuple(t for t in args if torch.is_tensor(t))
        if not CompileFlag.async_comm:
            CudaTimer().start('adapter')
        outputs = adapter(*args)
        if not CompileFlag.async_comm:
            CudaTimer().stop('adapter')
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        if require_grad:
            grad_dtypes = (torch.float16, torch.float32)
            outputs = tuple(t.requires_grad_() if torch.is_tensor(t) and t.dtype in grad_dtypes else t for t in outputs)
        return outputs

    @staticmethod
    def exchange(sadapter: Callable, radapter: Callable, stage_id: int, require_grads: bool, *args):
        """
        send adapter and recv adapter
        """
        # TODO: optimize with batch operators
        if stage_id % 2 == 0:
            ScheduleABC.adapter_step(sadapter, require_grads[0], *args)
            outs = ScheduleABC.adapter_step(radapter, require_grads[1])
        else:
            outs = ScheduleABC.adapter_step(radapter, require_grads[1])
            ScheduleABC.adapter_step(sadapter, require_grads[0], *args)
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
        out = ScheduleABC.status[name].pop(0)
        if len(ScheduleABC.status[name]) == 0:
            del ScheduleABC.status[name]
        return out

    @staticmethod
    def pop_tail(name: str):
        assert name in ScheduleABC.status, f"{name} is empty"
        out = ScheduleABC.status[name].pop(-1)
        if len(ScheduleABC.status[name]) == 0:
            del ScheduleABC.status
        return out
    
    @staticmethod
    def sync_tensors(tensors: List[Any]) -> List[Any]:
        """
        Wait until the finish of synchornized tensors
        """
        return [AsyncCommHandler().wait(t) if torch.is_tensor(t) else t for t in tensors]

    @staticmethod
    def assert_empty():
        assert len(ScheduleABC.status) == 0, f"status is not empty. Got field {list(ScheduleABC.status.keys())}"
