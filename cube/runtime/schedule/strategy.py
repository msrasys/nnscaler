from typing import Any, Callable, Dict, Iterable, List
import torch

from cube.profiler.timer import CudaTimer

from cube.flags import CompileFlag


def convert_fp32_to_fp16(t: Any):
    """
    A tensor with float32 will be converted to float16.
    A dtype of torch.float32 will be returned as torch.float16
    """
    if isinstance(t, torch.dtype) and t == torch.float32:
        t = torch.float16
    elif torch.is_tensor(t) and t.dtype == torch.float32:
        with torch.no_grad():
            t = t.half()
    return t


class ScheduleABC:

    status: Dict[str, List[torch.Tensor]] = dict()

    @staticmethod
    def forward_step(segment: Callable, *args, **kwargs):
        """
        forward pass
        """
        CudaTimer().start('forward')
        with torch.autocast('cuda', torch.float16, enabled=CompileFlag.use_amp):
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
        otensors = [t for t in otensors if t.requires_grad]
        assert len(otensors) == len(otensor_grads), f"output tensor mismatches with gradient number"
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
    def adapter_step(adapter: Callable, require_grad : bool = True, *args):
        """
        Adapter pass.
        If the adapter is None, will return (None,)
        """
        if adapter is None: return (None,)
        # if adapter is None: return ()
        args = tuple(t for t in args if torch.is_tensor(t))
        if CompileFlag.use_amp:
            args = tuple(convert_fp32_to_fp16(t) for t in args)
        CudaTimer().start('adapter')
        outputs = adapter(*args)
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
    def assert_empty():
        assert len(ScheduleABC.status) == 0, f"status is not empty. Got field {list(ScheduleABC.status.keys())}"
