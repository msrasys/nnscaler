#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

r"""
Executor for runtime
"""
import atexit

from typing import Tuple, Any, Callable, List, Dict, Optional
import torch
import logging
import os

_logger = logging.getLogger(__name__)

_ALLOW_GRAD_DTYPES = (torch.double, torch.float32, torch.float16, torch.bfloat16)


def debug_id(tensors, msg: str, rank: int):
    if torch.distributed.get_rank() == rank:
        if torch.is_tensor(tensors):
            print(f'[{torch.distributed.get_rank()}] {msg}: [{id(tensors)}]')
        else:
            print(f'[{torch.distributed.get_rank()}] {msg}: {[id(t) for t in tensors]}')


class AsyncCommHandler:

    class __AsyncCommHandler:
        def __init__(self):
            self._works: Dict[int, List] = {}
            self._callbacks: Dict[int, Callable] = {}

    instance = None

    def __init__(self) -> None:
        if not AsyncCommHandler.instance:
            AsyncCommHandler.instance = AsyncCommHandler.__AsyncCommHandler()

    def __getattr__(self, name):
        return getattr(self.instance, name)

    def wait(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Wait until the finish of the communication

        @param tensor torch.Tensor
        @return tensor torch.Tensor
        """
        if id(tensor) not in self._works:
            return tensor
        works = self._works.pop(id(tensor))
        for work in works:
            work.wait()
        callback = self._callbacks.pop(id(tensor))
        if callback is not None:
            tensor = callback(tensor)
        return tensor
    
    def submit(self, tensor: torch.Tensor, works: List, callback: Optional[Callable] = None):
        """
        Submit an async communication
        """
        self._works[id(tensor)] = works
        self._callbacks[id(tensor)] = callback

    def clear(self):
        AsyncCommHandler.instance = AsyncCommHandler.__AsyncCommHandler()

    def check_clear(self):
        assert len(self._works) == 0 and len(self._callbacks) == 0


TensorPairs = List[Tuple[int, torch.Tensor]]

_TIMING_EVENTS = []


def _timing_enabled() -> bool:
    return os.getenv('NNSCALER_EXECUTOR_TIMING', os.getenv('LLM_TRAIN_EXECUTOR_TIMING', '')).lower() in {'1', 'true', 'yes', 'on'}


def _timing_start(kind: str, name: str):
    if not _timing_enabled() or not torch.cuda.is_available():
        return None
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    return kind, name, start_event, end_event


def _timing_end(record) -> None:
    if record is None:
        return
    kind, name, start_event, end_event = record
    end_event.record()
    _TIMING_EVENTS.append((kind, name, start_event, end_event))


def get_and_clear_executor_timing_stats(max_entries: int = 100):
    global _TIMING_EVENTS
    if not _TIMING_EVENTS:
        return None
    torch.cuda.synchronize()
    stats = {}
    for kind, name, start_event, end_event in _TIMING_EVENTS:
        label = f'{kind}:{name}'
        elapsed_s = start_event.elapsed_time(end_event) / 1000.0
        stat = stats.setdefault(label, {'count': 0, 'sum_s': 0.0, 'max_s': 0.0})
        stat['count'] += 1
        stat['sum_s'] += elapsed_s
        stat['max_s'] = max(stat['max_s'], elapsed_s)
    _TIMING_EVENTS = []
    if max_entries and len(stats) > max_entries:
        stats = dict(sorted(stats.items(), key=lambda item: item[1]['sum_s'], reverse=True)[:max_entries])
    return stats


class Executor:

    # We consider each segment as an isolated graph. By
    # executing the forward of graph, the input tensors will be detached
    # from previous graph and saved for backward.
    # Each graph has its name, and multiple call for the graph will append
    # (instant id -> detached) input tensor pairs for backward reference.
    _detach: Dict[str, List[TensorPairs]] = dict()
    _backward_pre_hook: Optional[Callable] = None

    @staticmethod
    def fexecute(name: str, subgraph: Callable, *input_tensors: Tuple[Any], requires_grad=True):
        """
        forward the sub-graph.
        """
        timing_record = _timing_start('fw', name)
        input_tensors = Executor.sync_tensors(input_tensors)

        if not requires_grad:
            with torch.no_grad():
                outputs = subgraph(*input_tensors)
            _timing_end(timing_record)
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
        _timing_end(timing_record)
        return outputs

    @staticmethod
    def aexecute(subgraph: Callable, *input_tensors: Tuple[Any], requires_grad=True):
        """
        execute adapter
        """
        timing_record = _timing_start('adapter', getattr(subgraph, '__name__', subgraph.__class__.__name__))
        if not requires_grad:
            with torch.no_grad():
                outputs = subgraph(*input_tensors)
        else:
            outputs = subgraph(*input_tensors)
            if isinstance(outputs, tuple):
                outputs = (t.requires_grad_() if torch.is_tensor(t) and t.dtype in _ALLOW_GRAD_DTYPES else t for t in outputs)
            elif torch.is_tensor(outputs) and outputs.dtype in _ALLOW_GRAD_DTYPES:
                outputs = outputs.requires_grad_()
        _timing_end(timing_record)
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
        timing_record = _timing_start('bw', name)
        output_tensor_grads = Executor.sync_tensors(output_tensor_grads)

        saved_pairs = Executor._detach[name].pop(0)
        tensor_ids: List[int] = [pair[0] for pair in saved_pairs]
        dtensors: List[torch.Tensor] = [pair[1] for pair in saved_pairs]
        for t in input_tensors:
            if id(t) not in tensor_ids:
                import traceback
                _logger.warning(
                    f"rank {torch.distributed.get_rank()}: input {name} doesn't match. "
                    f"Make sure in scheduling, earlier forward perform earlier backward. "
                    f"Remain {len(Executor._detach[name])} segments.\n"
                    f"{''.join(traceback.format_stack())}"
                )
        
        if len(output_tensors) == 0: return None

        input_tensors = []
        for t in dtensors:
            if torch.is_tensor(t) and t.requires_grad:
                t.retain_grad()
                input_tensors.append(t)

        visited = set()
        dedup_output_tensors = []
        dedup_output_tensor_grads = []
        for t, g in zip(output_tensors, output_tensor_grads):
            # filter out duplicated output tensor and its grad.
            pair = (id(t), id(g))
            if pair not in visited:
                visited.add(pair)
                dedup_output_tensors.append(t)
                dedup_output_tensor_grads.append(g)

        # apply hook before backward
        if Executor._backward_pre_hook is not None:
            input_tensors, dedup_output_tensors, dedup_output_tensor_grads = \
                Executor._backward_pre_hook(
                    input_tensors,
                    dedup_output_tensors,
                    dedup_output_tensor_grads
                )

        torch.autograd.backward(
            dedup_output_tensors,
            grad_tensors=dedup_output_tensor_grads,
        )
        grads = tuple(t.grad for t in input_tensors)
        assert all(grad is not None for grad in grads), "RuntimeError: got gradient None"

        _timing_end(timing_record)
        if len(grads) == 0:
            return None
        if len(grads) == 1:
            return grads[0]
        return grads

    @staticmethod
    def sync_tensors(tensors: List[Any]) -> List[Any]:
        """
        Wait until the finish of synchornized tensors
        """
        return [AsyncCommHandler().wait(t) if torch.is_tensor(t) else t for t in tensors]


    @staticmethod
    def register_backward_pre_hook(hook: Optional[Callable]):
        """Register a backward hook for the right before the backward executor.

        The backward hook will be called with the following arguments:
            hook(input_tensors, output_tensors, output_tensor_grads) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]

        The backward hook mainly serves for the scenarios like loss scaling.

        Notes:
            Users can only register one backward pre_hook. If there was a hook
            registered before, it will be overwritten.

        Args:
            hook (Callable or None): the backward hook to be registered. The hook takes
                input_tensors (List[torch.Tensor]),
                output_tensors (List[torch.Tensor]),
                output_tensor_grads (List[torch.Tensor]) as inputs and returns the
                same format of updated tensors.
        """
        Executor._backward_pre_hook = hook
    
    @staticmethod
    def clear():
        Executor._detach = dict()
        Executor._backward_pre_hook = None

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
atexit.register(AsyncCommHandler().check_clear)
