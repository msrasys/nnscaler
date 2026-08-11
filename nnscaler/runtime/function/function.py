#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""
The functions in this file might be inserted as node to graph, to ensure that the inserted node can generate the correct code,
please following the assumption:
  - should execute under default context (not under for example, torch.no_grad) no matter what the producer and consumer context are.
"""

from contextlib import contextmanager
from typing import Callable, Optional, List, Tuple, Union, Any
import torch
import torch.nn.functional as TorchF
import operator
import datetime
from nnscaler.flags import CompileFlag, RuntimeFlag


def identity(tensor: torch.Tensor) -> torch.Tensor:
    """
    identity forward
    """
    return tensor


def ifexpr(cond: bool, true_value: Any, false_value: Any) -> Any:
    """
    if expression
    Please note there is no short-circuit evaluation in this function.
    """
    return true_value if cond else false_value


def anchor(name: str):
    """
    anchor operation for graph navigation
    """
    return None


@contextmanager
def constant_folding(constant_folding: bool = True):
    """
    Context manager to enable/disable constant folding.
    You can put it inside your forward function to control the constant folding behavior.
    Please note as we don't set it as leaf function in tracer,
    it will not be present in the traced graph.
    """
    from nnscaler.graph.tracer.metadata import _GLOBAL_OP_CONTEXT

    old_constant_folding = _GLOBAL_OP_CONTEXT.constant_folding
    _GLOBAL_OP_CONTEXT.constant_folding = constant_folding
    try:
        yield
    finally:
        _GLOBAL_OP_CONTEXT.constant_folding = old_constant_folding


def no_constant_folding():
    """
    Context manager to disable constant folding.
    """
    return constant_folding(constant_folding=False)


def fold_constant(a: Any) -> Any:
    """
    Fold a constant(non-tensor) if constant folding is enabled.

    Please note this should be only used in `constant_folding` block
    to make sure the input to a `constant_folding` block is not wrapped in an IRObject in the graph.

    Example:
    ```
    a = some_func()  # the value is wrapped in IRObject in graph
    with constant_folding():
        a = fold_constant(a)  # unwrap value
        torch.add(t, a)       #  in graph a is a constant
    ```
    """
    return a


def multiref(tensor: torch.Tensor, times: int, *, clone_level: int = 0) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
    """
    identity forward. Create multiple same tensor.
    Args:
        tensor (torch.Tensor): input tensor
        times (int): number of same tensor to create
        clone_level (int): 0: no clone, 1: clone once for all, 2: clone each time
    Returns:
        Union[torch.Tensor, Tuple[torch.Tensor]]:
            if times==1, return tensor; else return tuple of tensors
    """
    if clone_level == 0:
        return tensor if times == 1 else tuple([tensor] * times)
    elif clone_level == 1:
        cloned_tensor = tensor.clone()
        return cloned_tensor if times == 1 else tuple([cloned_tensor] * times)
    else:  # clone_level == 2
        return tensor.clone() if times == 1 else tuple([tensor.clone() for _ in range(times)])


def to(tensor: torch.Tensor, dtype_or_device: Union[torch.device, torch.dtype]) -> torch.Tensor:
    # deprecated
    # keep it only for backward compatibility
    return tensor.to(dtype_or_device)


def accum(*tensors: Tuple[torch.Tensor]) -> torch.Tensor:
    """
    accumulate tensors in to one tensor
    """
    if len(tensors) == 2:
        return tensors[0] + tensors[1]
    else:
        return torch.sum(torch.stack(tensors, dim=0), dim=0)


def fullslice(input: torch.Tensor, *slicers: Union[None, slice, int, torch.Tensor]):
    """Slice tensors

    Note:
    1) `None` will always extend a dimension at current position.
    2) `slice(None, None, None)` equals to `:`,
        meaning select every element at its dimension.

    Args:
        input (torch.Tensor): input tensor
        slicers (Union[None | slicer | int | torch.Tensor]): slicers for input

    Returns:
        torch.Tensor: sliced tensor
    """
    return input[tuple(slicers)]


def conv2d(input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor],
           stride: int, padding: List[int], dilation, groups: int = 1):
    """
    input:  N  iC H  W
    weight: oC iC dH dW
    bias:   oC
    padding: List[int, int, int, int]: [Htop, Hbottom, Wtop, Wbottom] or
             List[int, int]: [Hside, Wside]
    """
    # switch H and W to match torch.nn.functional.pad
    padding = padding[len(padding) // 2:] + padding[0:len(padding) // 2]
    input = TorchF.pad(input, padding, 'constant', 0)
    return TorchF.conv2d(input, weight, bias, stride=stride, dilation=dilation, groups=groups)


def conv3d(input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor],
           stride: int, padding: List[int], dilation, groups: int = 1):
    """
    input:  N iC D H W,
    weight: oC iC dH dW, oC
    bias:   oC
    padding: List[int, int, int, int]: [Htop, Hbottom, Wtop, Wbottom] or
             List[int, int]: [Hside, Wside]

    output: N oC oD oH oW
    """
    # switch D, H and W to match torch.nn.functional.pad
    pad_padding = [padding[-1 - (i // 2)] for i in range(len(padding) * 2)]
    input = TorchF.pad(input, pad_padding, 'constant', 0)
    return TorchF.conv3d(input, weight, bias, stride=stride, dilation=dilation, groups=groups)


@torch.no_grad()
def _accumulate_embedding_grad(
    weight: torch.Tensor,
    masked_input: torch.Tensor,
    input_mask: torch.Tensor,
    grad_output: torch.Tensor,
    padding_idx: Optional[int],
) -> bool:
    """Accumulate an embedding dWeight without a vocabulary-sized temporary."""
    if not (
        isinstance(weight, torch.nn.Parameter)
        and weight.is_leaf
        and weight.requires_grad
    ):
        return False

    from nnscaler.runtime.adapter.reducer import has_reducer_grad_accumulator
    if has_reducer_grad_accumulator(weight):
        return False

    from nnscaler.runtime.utils import get_grad_dtype
    if weight.grad is None:
        weight.grad = torch.zeros_like(weight, dtype=get_grad_dtype(weight))

    indices = masked_input.reshape(-1)
    valid = ~input_mask.reshape(-1)
    if padding_idx is not None:
        valid = valid & (indices != padding_idx)
    source = grad_output.reshape(-1, grad_output.shape[-1])

    # Work through fixed-size vocabulary chunks. Boolean indexing and
    # torch.unique both produce dynamic-size CUDA outputs and therefore
    # synchronize the host once per microbatch. A reserved padding row maps
    # every out-of-chunk token to an ignored index while preserving the native
    # embedding reduction (and its deterministic summation order) for the
    # real rows. Only one bounded BF16 chunk exists at a time; no
    # vocabulary-sized dWeight is materialized.
    chunk_rows = 32768
    for row_start in range(0, weight.shape[0], chunk_rows):
        row_end = min(row_start + chunk_rows, weight.shape[0])
        in_chunk = valid & (indices >= row_start) & (indices < row_end)
        chunk_indices = torch.where(
            in_chunk,
            indices - row_start + 1,
            0,
        )
        chunk_grad = torch.ops.aten.embedding_dense_backward(
            source,
            chunk_indices,
            row_end - row_start + 1,
            0,
            False,
        )
        weight.grad[row_start:row_end].add_(chunk_grad[1:])
    return True


def _embedding_dense_backward(
    grad_output: torch.Tensor,
    masked_input: torch.Tensor,
    input_mask: torch.Tensor,
    num_weights: int,
    padding_idx: Optional[int],
) -> torch.Tensor:
    if input_mask.any():
        grad_output = grad_output.masked_fill(input_mask.unsqueeze(-1), 0)
    return torch.ops.aten.embedding_dense_backward(
        grad_output,
        masked_input,
        num_weights,
        -1 if padding_idx is None else padding_idx,
        False,
    )


class _Embedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, padding_idx, start, stop):
        input = input.long()
        input_mask = (input < start) | (input >= stop)
        masked_input = input.clone() - start
        masked_input[input_mask] = 0
        local_padding_idx = (
            padding_idx - start
            if padding_idx is not None and start <= padding_idx < stop
            else None
        )
        output = TorchF.embedding(
            masked_input,
            weight,
            local_padding_idx,
            None,
            2.0,
            False,
            False,
        )
        output[input_mask, :] = 0.0
        ctx.save_for_backward(masked_input, input_mask)
        ctx.weight = weight
        ctx.padding_idx = local_padding_idx
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if RuntimeFlag.fbw_phase == "input":
            return None, None, None, None, None

        masked_input, input_mask = ctx.saved_tensors
        if (
            RuntimeFlag.fbw_phase == "native_weight"
            and _accumulate_embedding_grad(
                ctx.weight,
                masked_input,
                input_mask,
                grad_output,
                ctx.padding_idx,
            )
        ):
            return None, None, None, None, None

        grad_weight = _embedding_dense_backward(
            grad_output,
            masked_input,
            input_mask,
            ctx.weight.shape[0],
            ctx.padding_idx,
        )
        return None, grad_weight, None, None, None


def embedding(input: torch.Tensor, weight: torch.Tensor, padding_idx: Optional[int], start: int, stop: int):
    """
    add start/stop to make vocab dim partitionable.

    for example, if the vocab size is 100, and partition the weigth on vocab dim to 4 part,
    then on each part, it will have different start/stop:
        1: [start=0, stop=25]
        2: [start=25, stop=50]
        3: [start=50, stop=75]
        4: [start=75, stop=100]
    before do embedding, the input index outside the range will be masked,
    and directly assign 0.0 to the masked position on the output.

    If vocab dim is partitioned, the results are summed to ensure the correctness of the final result.

    Inputs:
        input: torch.Tensor [*]
        weight: [vocab size, embed size]
        start: int, the weight split start index on vocab dim
        stop: int, the weight split stop index on vocab dim

    Outputs:
        output: [*, embed_size]
    """
    return _Embedding.apply(input, weight, padding_idx, start, stop)


def layer_norm(input: torch.Tensor,
               weight: Optional[torch.Tensor], bias: Optional[torch.Tensor],
               normalized_shape: List[int], eps: float = 1e-05) -> torch.Tensor:
    """
    LayerNorm
    """
    return torch.nn.functional.layer_norm(input, normalized_shape, weight, bias, eps)


# 'torch.select_scatter' isn't supported by Torch2ONNX yet.
# Implement it with 'torch.masked_scatter' which is supported with ONNX opset=11.
def select_scatter(input:torch.Tensor, src:torch.Tensor, dim:int, index:int):
    # e.g. [..., 1, -1, 1, ...]
    shape = [1] * input.ndim
    shape[dim] = -1

    d = input.shape[dim]
    mask = torch.zeros([d], dtype=torch.bool, device=input.device)
    mask[index] = True
    mask = mask.reshape(shape)

    return torch.masked_scatter(input, mask, src)


def tensor(data, *, dtype=None, requires_grad=False, pin_memory=False):
    """
    force set the device to torch.cuda.current_device()
    """
    return torch.tensor(
        data, dtype=dtype,
        device=torch.cuda.current_device(),
        requires_grad=requires_grad, pin_memory=pin_memory
    )


def empty(size: Tuple[int], dtype=None, requires_grad=False, pin_memory=False):
    """
    force set the device to torch.cuda.current_device()
    """
    return torch.empty(
        size, dtype=torch.get_default_dtype() if dtype is None else dtype,
        device=torch.cuda.current_device(),
        requires_grad=requires_grad, pin_memory=pin_memory
    )


def zeros(size: Tuple[int], dtype=None, requires_grad=False):
    """
    force set the device to torch.cuda.current_device()
    """
    return torch.zeros(
        size, dtype=torch.get_default_dtype() if dtype is None else dtype,
        device=torch.cuda.current_device(),
        requires_grad=requires_grad
    )


def ones(size: Tuple[int], dtype=None, requires_grad=False):
    """
    force set the device to torch.cuda.current_device()
    """
    return torch.ones(
        size, dtype=torch.get_default_dtype() if dtype is None else dtype,
        device=torch.cuda.current_device(),
        requires_grad=requires_grad
    )


def rand(size: Tuple[int], dtype=None, requires_grad=False, pin_memory=False):
    """
    force set the device to torch.cuda.current_device()
    """
    return torch.rand(
        size, dtype=torch.get_default_dtype() if dtype is None else dtype,
        device=torch.cuda.current_device(),
        pin_memory=pin_memory,
        requires_grad=requires_grad,
    )


def randn(size: Tuple[int], dtype=None, requires_grad=False, pin_memory=False):
    """
    force set the device to torch.cuda.current_device()
    """
    return torch.randn(
        size, dtype=torch.get_default_dtype() if dtype is None else dtype,
        device=torch.cuda.current_device(),
        pin_memory=pin_memory,
        requires_grad=requires_grad,
    )


def full(size: Tuple[int], fill_value, dtype=None, requires_grad=False):
    """
    force set the device to torch.cuda.current_device()
    """
    return torch.full(
        size, fill_value, dtype=dtype, requires_grad=requires_grad,
        device=torch.cuda.current_device()
    )


def arange(start: int, end: int, step: int, dtype: torch.dtype, requires_grad=False):
    return torch.arange(start=start, end=end, step=step,
                        dtype=dtype, requires_grad=requires_grad,
                        device=torch.cuda.current_device())


def linspace(start: Union[int, torch.Tensor], end: Union[int, torch.Tensor],
             steps: int, dtype: torch.dtype, requires_grad=False):
    return torch.linspace(start, end, steps, dtype=dtype, requires_grad=requires_grad,
                          device=torch.cuda.current_device())


def eye(n: int, m: Optional[int]=None, requires_grad=False, dtype: torch.dtype=torch.float32) -> torch.Tensor:
    return torch.eye(n, m=m, dtype=dtype, device=torch.cuda.current_device(), requires_grad=requires_grad)


def index_select(input: torch.Tensor, index: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.index_select(input, dim, index)


def einsum(*operands, equation=None) -> torch.Tensor:
    return torch.einsum(equation, *operands)


def stack(*tensors, dim=0) -> torch.Tensor:
    return torch.stack(tensors, dim)


def cat(*tensors, dim=0) -> torch.Tensor:
    return torch.cat(tensors, dim)


def nndropout(input: torch.Tensor, p=0.5, inplace=False):
    return torch.nn.Dropout(p, inplace)(input)


def setitem(__a, *__bc):
    """
    If __bc has more than 2 elements, that means idxs are flatten becasue idxs contains tensor.
    In this runtime function, idxs will be structured as a tuple if they are flatten,
    and return __a to make this inplace operation trackable.
    """
    if len(__bc) < 2:
        raise ValueError(f'at least two arguments needed, but get __bc={__bc}')
    elif len(__bc) == 2:
        __b, __c = __bc[0], __bc[1]
    else:
        __b, __c = __bc[:-1], __bc[-1]
    operator.setitem(__a, __b, __c)
    return __a


def dict_keys(d: dict):
    return tuple(d.keys())


def dict_values(d: dict):
    return tuple(d.values())


def dict_items(d: dict):
    return tuple(d.items())


def print_time(content: str):
    if not CompileFlag.line_timer:
        return
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else -1
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f"line timer: {rank} - {datetime.datetime.now()} - {content}")


class _BackwardHook(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, backward_hook: Callable[[], None]):
        ctx.save_for_backward()
        ctx.backward_hook = backward_hook
        return x

    @staticmethod
    def backward(ctx, grad_output):
        ctx.backward_hook()
        return grad_output, None


def insert_backward_hook(x: torch.Tensor, backward_hook: Optional[Callable[[], None]]) -> torch.Tensor:
    if backward_hook is None:
        # no need to add hook
        return x
    return _BackwardHook.apply(x, backward_hook)
