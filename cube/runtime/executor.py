r"""
Executor for runtime
"""

from typing import Tuple, Any, Callable, List, Dict
import torch


def debug_id(tensors, msg: str, rank: int):
    if torch.distributed.get_rank() == rank:
        if torch.is_tensor(tensors):
            print(f'[{torch.distributed.get_rank()}] {msg}: [{id(tensors)}]')
        else:
            print(f'[{torch.distributed.get_rank()}] {msg}: {[id(t) for t in tensors]}')


class Executor:

    _detach: Dict[str, Dict[torch.Tensor, torch.Tensor]] = dict()

    @staticmethod
    def fexecute(name: str, subgraph: Callable, *input_tensors: Tuple[Any], requires_grad=True):
        """
        forward the sub-graph.
        """
        if not requires_grad:
            with torch.no_grad():
                outputs = subgraph(*input_tensors)
        else:
            # everytime forward a segment, detach the tensor from previous graph
            # debug_id(input_tensors, 'outside fexecute args', 0)
            assert name not in Executor._detach
            Executor._detach[name] = dict()
            for itensor in input_tensors:
                if torch.is_tensor(itensor) and itensor.requires_grad:
                    if itensor not in Executor._detach[name]:
                        Executor._detach[name][itensor] = itensor.detach().requires_grad_()
            input_tensors = tuple(
                Executor._detach[name][t] if t in Executor._detach[name] else t for t in input_tensors
            )
            # debug_id(input_tensors, 'inside fexecute args', 0)
            outputs = subgraph(*input_tensors)
            # debug_id(outputs, 'fexecute result', 0)
        # print('forwarding... ')
        return outputs

    @staticmethod
    def aexecute(subgraph: Callable, *input_tensors: Tuple[Any], requires_grad=True):
        """
        execute adapter
        """
        if not requires_grad:
            with torch.no_grad():
                outputs = subgraph(*input_tensors)
        else:
            outputs = subgraph(*input_tensors)
            outputs = outputs.requires_grad_() if torch.is_tensor(outputs) else (t.requires_grad_() for t in outputs)
        return outputs

    @staticmethod
    def backward(name: str,
                 input_tensors: List[torch.Tensor],
                 output_tensors: List[torch.Tensor],
                 output_tensor_grads: List[torch.Tensor]) -> Tuple[torch.Tensor]:
        """
        Backward Procedure.

        input_tensors: List[torch.Tensor]:
            tensors that their gradient need to be computed, including parameters.
            Correspoinding forward input tensors.

        output_tensors:
            tensors that start for gradient backward computation.
            Corresponding to forward output tensors.

        output_tensor_grads:
            gradient tensors corresponding to output_tensors.

        Returns:
            gradient in order of non-parameter tensors in input_tensors.
            (Note parameter tnesors already have gradient accumulated at .grad attribute)
        """
        if len(output_tensors) == 0:
            return None

        assert name in Executor._detach, f"forward graph: {name} not run before"
        input_tensors = [t for t in input_tensors if torch.is_tensor(t) and not isinstance(t, torch.nn.Parameter)]
        input_tensors = [t for t in input_tensors if t.requires_grad]
        input_tensors = [Executor._detach[name][t] if t in Executor._detach[name] else t for t in input_tensors]
        for t in input_tensors:
            t.retain_grad()
        torch.autograd.backward(
            output_tensors,
            grad_tensors=output_tensor_grads,
        )
        grads = tuple(t.grad for t in input_tensors)
        assert all(grad is not None for grad in grads), "RuntimeError: got gradient None"
        del Executor._detach[name]
        if    len(grads) == 0: return None
        elif  len(grads) == 1: return grads[0]
        else: return tuple(grads)

    @staticmethod
    def clear():
        Executor._detach = dict()

    @staticmethod
    def check_clear():
        assert len(Executor._detach) == 0, \
            f"Find remain not consumed sub-graph: {tuple(Executor._detach.keys())}"


fexecute = Executor.fexecute
aexecute = Executor.aexecute
backward = Executor.backward


# def backward(input_tensors : List[torch.Tensor],
#              output_tensors: List[torch.Tensor],
#              output_tensor_grads: List[torch.Tensor]):
#     """
#     Backward Procedure.
# 
#     input_tensors: List[torch.Tensor]:
#         tensors that their gradient need to be computed, including parameters.
#         Correspoinding forward input tensors.
#     
#     output_tensors:
#         tensors that start for gradient backward computation.
#         Corresponding to forward output tensors.
# 
#     output_tensor_grads:
#         gradient tensors corresponding to output_tensors.
# 
#     Returns:
#         gradient in order of non-parameter tensors in input_tensors.
#         (Note parameter tnesors already have gradient accumulated at .grad attribute)
#     """
#     if len(input_tensors) == 0:
#         return None
#     grads = list()
#     in_grads = torch.autograd.grad(
#         outputs = output_tensors,
#         inputs  = input_tensors,
#         grad_outputs = output_tensor_grads,
#         allow_unused=True
#     )
#     for tensor, grad in zip(input_tensors, in_grads):
#         if isinstance(tensor, torch.nn.Parameter):
#             if tensor.grad is not None:
#                 tensor.grad += grad
#             else:
#                 tensor.grad = grad
#         else:
#             grads.append(grad)
#     if    len(grads) == 0: return None
#     elif  len(grads) == 1: return grads[0]
#     else: return tuple(grads)


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
