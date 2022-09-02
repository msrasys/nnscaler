r"""
Executor for runtime
"""

from typing import Tuple, Any, Callable, List, Dict
import torch



class Executor:

    _detach: Dict[torch.Tensor, torch.Tensor] = dict()

    @staticmethod
    def fexecute(subgraph: Callable, *input_tensors: Tuple[Any], requires_grad=True):
        """
        forward the sub-graph.
        """
        if not requires_grad:
            with torch.no_grad():
                outputs = subgraph(*input_tensors)
        else:
            # everytime forward a segment, detach the tensor from previous graph
            for itensor in input_tensors:
                if torch.is_tensor(itensor) and itensor.requires_grad:
                    assert itensor not in Executor._detach
                    Executor._detach[itensor] = itensor.detach().requires_grad_()
            input_tensors = tuple(
                Executor._detach[t] if t in Executor._detach else t for t in input_tensors
            )
            outputs = subgraph(*input_tensors)
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
        # print('forwarding... ')
        return outputs

    @staticmethod
    def backward(input_tensors: List[torch.Tensor],
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
        inputs = list()
        # everytime backward a input tensor, remove it from _detach
        input_tensors = [Executor._detach.pop(t) if t in Executor._detach else t for t in input_tensors]
        for input_ in input_tensors:
            if torch.is_tensor(input_) and not isinstance(input_, torch.nn.Parameter):
                if input_.requires_grad:
                    input_.retain_grad()
                    inputs.append(input_)
        torch.autograd.backward(
            output_tensors,
            grad_tensors=output_tensor_grads,
        )
        grads = tuple(input_.grad for input_ in inputs)
        if    len(grads) == 0: return None
        elif  len(grads) == 1: return grads[0]
        else: return tuple(grads)


fexecute = Executor.fexecute
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
