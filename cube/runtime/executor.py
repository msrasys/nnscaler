r"""
Executor for runtime
"""
import atexit

from typing import Tuple, Any, Callable, List, Dict
import torch
import warnings

from cube.flags import CompileFlag


if CompileFlag.use_amp:
    warnings.warn(
        "Detected auto mixed precision (AMP) is enabled. It's an "
        "experimental feature that is only for benchmark. "
        "torch.cdua.amp.GradScalerr is not enabled for loss "
        "and optimizer, which may lead to gradient loss. The tensors "
        "and dtypes arguments in adapter will be automatically converted to "
        "torch.float16, if they are in float32 precision or torch.float32 dtype."
    )


def debug_id(tensors, msg: str, rank: int):
    if torch.distributed.get_rank() == rank:
        if torch.is_tensor(tensors):
            print(f'[{torch.distributed.get_rank()}] {msg}: [{id(tensors)}]')
        else:
            print(f'[{torch.distributed.get_rank()}] {msg}: {[id(t) for t in tensors]}')


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


TensorPairs = List[Tuple[int, torch.Tensor]]


class Executor:

    # We consider each segment as an isolated graph. By
    # executing the forward of graph, the input tensors will be detached
    # from previous graph and saved for backward.
    # Each graph has its name, and multiple call for the graph will append
    # (instant id -> detached) input tensor pairs for backward reference.
    _detach: Dict[str, List[TensorPairs]] = dict()

    @staticmethod
    def fexecute(name: str, subgraph: Callable, *input_tensors: Tuple[Any], requires_grad=True):
        """
        forward the sub-graph.
        """
        if not requires_grad:
            with torch.no_grad():
                outputs = subgraph(*input_tensors)
            return outputs

        # everytime forward a segment, detach the tensor from previous graph
        mapping: Dict[int, torch.Tensor] = dict()
        for itensor in input_tensors:
            if torch.is_tensor(itensor) and itensor.requires_grad:
                mapping[id(itensor)] = itensor.detach().requires_grad_()
        input_dtensors = tuple(mapping[id(t)] if id(t) in mapping else t for t in input_tensors)
        
        saved_pairs = [(id(itensor), dtensor) for itensor, dtensor in zip(input_tensors, input_dtensors)]
        Executor._detach.setdefault(name, []).append(saved_pairs)  
        
        outputs = subgraph(*input_dtensors)
        return outputs

    @staticmethod
    def aexecute(subgraph: Callable, *input_tensors: Tuple[Any], requires_grad=True):
        """
        execute adapter
        """
        if CompileFlag.use_amp:
            input_tensors = tuple(convert_fp32_to_fp16(t) for t in input_tensors)

        if not requires_grad:
            with torch.no_grad():
                outputs = subgraph(*input_tensors)
        else:
            outputs = subgraph(*input_tensors)
            allow_grad_dtypes = (torch.float32, torch.float16)
            if torch.is_tensor(outputs) and outputs.dtype in allow_grad_dtypes:
                outputs = outputs.requires_grad_()
            else:
                outputs = (t.requires_grad_() if t.dtype in allow_grad_dtypes else t for t in outputs)
        return outputs

    @staticmethod
    def backward(name: str,
                 input_tensors: List[torch.Tensor],
                 output_tensors: List[torch.Tensor],
                 output_tensor_grads: List[torch.Tensor]) -> Tuple[torch.Tensor]:
        """
        Backward Procedure.

        @param input_tensors List[torch.Tensor]
            tensors that their gradient need to be computed, including parameters.
            Correspoinding forward input tensors.

        @param output_tensors List[torch.Tensor]
            tensors that start for gradient backward computation.
            Corresponding to forward output tensors.

        @param output_tensor_grads List[torch.Tensor]:
            gradient tensors corresponding to output_tensors.

        @return gradients List[torch.Tensor]:
            gradient tensors corresponding to input_tensors.
        """
        if len(output_tensors) == 0: return None

        saved_pairs = Executor._detach[name].pop(0)
        tensor_ids: List[int] = [pair[0] for pair in saved_pairs]
        dtensors: List[torch.Tensor] = [pair[1] for pair in saved_pairs]
        for t in input_tensors:
            if id(t) not in tensor_ids:
                warnings.warn("input doesn't match. Make sure in scheduling that earlier forward perform earlier backward")

        input_tensors = []
        for t in dtensors:
            if torch.is_tensor(t) and t.requires_grad:
                t.retain_grad()
                input_tensors.append(t)

        torch.autograd.backward(
            output_tensors,
            grad_tensors=output_tensor_grads,
        )
        grads = tuple(t.grad for t in input_tensors)
        assert all(grad is not None for grad in grads), "RuntimeError: got gradient None"

        if    len(grads) == 0: return None
        elif  len(grads) == 1: return grads[0]
        else: return grads

    @staticmethod
    def clear():
        Executor._detach = dict()

    @staticmethod
    def check_clear():
        for name, npairs in Executor._detach.items():
            assert len(npairs) == 0, \
                f"Fine remaining segment needs backward: {name}, remaining times: {len(npairs)}"


fexecute = Executor.fexecute
aexecute = Executor.aexecute
backward = Executor.backward

# register checking for normal exit
atexit.register(Executor.check_clear)


### =================== Experimental Feature =======================

# import queue
# 
# 
# class MessageManager:
#     """
#     message manager to make send as async calls.
#     """
# 
#     class __MessageManager:
#         def __init__(self):
#             self._reqs = queue.Queue(maxsize=128)
# 
#     instance = None
# 
#     def __init__(self):
#         if not MessageManager.instance:
#             MessageManager.instance = MessageManager.__MessageManager()
# 
#     def __getattr__(self, name):
#         return getattr(self.instance, name)
#     
#     def push(self, req):
#         self.instance._reqs.put(req, block=True, timeout=None)
#     
#     def pull(self):
#         return self.instance._reqs.get(block=True, timeout=None)
