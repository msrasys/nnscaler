from typing import List, Tuple
import torch

from cube.profiler.timer import CudaTimer, print_each_rank
from cube.profiler.memory import memory_summary
from cube.runtime.device import DeviceGroup

from handcraft.module.stage import PipeStage

io_input = input

def forward_step(model, *args, **kwargs):
    """
    Forward pass
    """
    CudaTimer().start("forward")
    outputs = model(*args, **kwargs)
    if not isinstance(outputs, tuple):
        outputs = (outputs, )
    CudaTimer().stop("forward")
    return outputs


def backward_step(input_tensors: List[torch.Tensor],
                  output_tensors: List[torch.Tensor],
                  output_tensor_grads: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    Backward pass
    """
    for tensor in input_tensors:
        if torch.is_tensor(tensor) and tensor.requires_grad:
            tensor.retain_grad()
    CudaTimer().start("backward")
    torch.autograd.backward(output_tensors, grad_tensors=output_tensor_grads)
    CudaTimer().stop("backward")
    input_tensor_grads = []
    for tensor in input_tensors:
        if torch.is_tensor(tensor) and tensor.requires_grad:
            input_tensor_grads.append(tensor.grad)
        else:
            input_tensor_grads.append(None)
    return input_tensor_grads


def recv_forward(model: PipeStage, prev_rank: int) -> List[torch.Tensor]:
    shapes, dtypes = model.inputs_info
    assert len(shapes) == len(dtypes)
    assert isinstance(prev_rank, int), "Expected prev_rank to be int"
    # print(f'rank {DeviceGroup().rank} recving forward: {shapes}, {dtypes}')
    if len(shapes) == 0: return ()

    CudaTimer().start(field_name='comm')
    tensors = [
        torch.empty(
            shape, requires_grad=True, dtype=dtype,
            device=torch.cuda.current_device()
        ) for shape, dtype in zip(shapes, dtypes)
    ]
    recv_ops = [
        torch.distributed.P2POp(
            torch.distributed.irecv, tensor, prev_rank
        ) for tensor in tensors
    ]
    reqs = torch.distributed.batch_isend_irecv(recv_ops)
    for req in reqs:
        req.wait()
    torch.cuda.synchronize()
    CudaTimer().stop(field_name='comm')
    return tensors


def recv_backward(model: PipeStage, next_rank: int) -> List[torch.Tensor]:
    shapes, dtypes = model.outputs_info
    assert len(shapes) == len(dtypes)
    assert isinstance(next_rank, int), "Expected next_rank to be int"
    # print(f'rank {DeviceGroup().rank} recving backward: {shapes}')
    if len(shapes) == 0: return ()

    CudaTimer().start(field_name='comm')
    tensors = [
        torch.empty(
            shape, requires_grad=False, dtype=dtype,
            device=torch.cuda.current_device()
        ) for shape, dtype in zip(shapes, dtypes)
    ]
    recv_ops = [
        torch.distributed.P2POp(
            torch.distributed.irecv, tensor, next_rank
        ) for tensor in tensors
    ]
    reqs = torch.distributed.batch_isend_irecv(recv_ops)
    for req in reqs:
        req.wait()
    torch.cuda.synchronize()
    CudaTimer().stop(field_name='comm')
    return tensors


def send_forward(outputs: List[torch.Tensor], next_rank: int):
    assert all([torch.is_tensor(out) for out in outputs]), "Expected List[Tensor]"
    assert isinstance(next_rank, int), "Expected next_rank to be int"
    if len(outputs) == 0: return
    # print(f'rank {DeviceGroup().rank} sending forward: {[tuple(t.size()) for t in outputs]}')
    
    CudaTimer().start(field_name='comm')
    send_ops = [
        torch.distributed.P2POp(
            torch.distributed.isend, tensor, next_rank
        ) for tensor in outputs
    ]
    reqs = torch.distributed.batch_isend_irecv(send_ops)
    for req in reqs:
        req.wait()
    torch.cuda.synchronize()
    CudaTimer().stop(field_name='comm')


def send_backward(grads: List[torch.Tensor], prev_rank: int):
    assert all([torch.is_tensor(grad) for grad in grads]), "Expected List[Tensor]"
    assert isinstance(prev_rank, int), "Expected prev_rank to be int"
    if len(grads) == 0: return
    CudaTimer().start(field_name='comm')
    # print(f'rank {DeviceGroup().rank} sending backward: {[tuple(t.size()) for t in grads]}')

    send_ops = [
        torch.distributed.P2POp(
            torch.distributed.isend, tensor, prev_rank
        ) for tensor in grads
    ]
    reqs = torch.distributed.batch_isend_irecv(send_ops)
    for req in reqs:
        req.wait()
    torch.cuda.synchronize()
    CudaTimer().stop(field_name='comm')


def send_forward_recv_backward(outputs, model: PipeStage, next_rank: int) -> List[torch.Tensor]:
    assert all([torch.is_tensor(out) for out in outputs]), "Expected List[Tensor]"
    assert isinstance(next_rank, int), "Expected next_rank to be int"
    shapes, dtypes = model.outputs_info
    assert len(shapes) == len(dtypes)
    # print(f'rank {DeviceGroup().rank} sending forward: {[tuple(t.size()) for t in outputs]} recving backward {shapes}')

    CudaTimer().start(field_name='comm')
    ops = list()
    # send forward outputs
    send_ops = [
        torch.distributed.P2POp(
            torch.distributed.isend, tensor, next_rank
        ) for tensor in outputs
    ]
    ops += send_ops
    # recv backward inputs
    tensors = [
        torch.empty(
            shape, requires_grad=True, dtype=dtype,
            device=torch.cuda.current_device()
        ) for shape, dtype in zip(shapes, dtypes)
    ]
    recv_ops = [
        torch.distributed.P2POp(
            torch.distributed.irecv, tensor, next_rank
        ) for tensor in tensors
    ]
    ops += recv_ops
    reqs = torch.distributed.batch_isend_irecv(ops)
    for req in reqs:
        req.wait()
    torch.cuda.synchronize()
    CudaTimer().stop(field_name='comm')
    return tensors


def send_backward_recv_forward(grads, model: PipeStage, prev_rank: int) -> List[torch.Tensor]:
    assert all([torch.is_tensor(grad) for grad in grads]), "Expected List[Tensor]"
    assert isinstance(prev_rank, int), "Expected prev_rank to be int"
    shapes, dtypes = model.inputs_info
    assert len(shapes) == len(dtypes)
    # print(f'rank {DeviceGroup().rank} sending backward: {[tuple(t.size()) for t in grads]} recving forward {shapes}')

    CudaTimer().start(field_name='comm')
    ops = list()
    # send backward gradients
    send_ops = [
        torch.distributed.P2POp(
            torch.distributed.isend, tensor, prev_rank
        ) for tensor in grads
    ]
    ops += send_ops
    # recv forward inputs
    tensors = [
        torch.empty(
            shape, requires_grad=True, dtype=dtype,
            device=torch.cuda.current_device()
        ) for shape, dtype in zip(shapes, dtypes)
    ]
    recv_ops = [
        torch.distributed.P2POp(
            torch.distributed.irecv, tensor, prev_rank
        ) for tensor in tensors
    ]
    ops += recv_ops
    reqs = torch.distributed.batch_isend_irecv(ops)
    for req in reqs:
        req.wait()
    torch.cuda.synchronize()
    CudaTimer().stop(field_name='comm')
    return tensors



def schedule_naive(model: PipeStage, dataloader, num_microbatch: int):
    """
    neighbors: (prev_rank: int, next_rank: int)
    """
    prev_rank = model.prev_stage_global_grank
    next_rank = model.next_stage_global_rank

    for _ in range(num_microbatch):
        model.data = next(dataloader)
        # print(f'rank {rank} recving forward input...')
        inputs = () if model.is_first_stage else recv_forward(model, prev_rank)
        # forward
        outputs = forward_step(model, *inputs)
        # send forward
        if not model.is_last_stage:
            # print(f'rank {rank} sending forward output...')
            send_forward(outputs, next_rank)
        # recv backward
        # print(f'rank {rank} recving backward input...')
        output_grads = (None,) if model.is_last_stage else recv_backward(model, next_rank)
        # backward
        input_grads = backward_step(inputs, outputs, output_grads)
        # send backward
        if not model.is_first_stage:
            # print(f'rank {rank} sending backward output...')
            send_backward(input_grads, prev_rank)


def schedule_1f1b(model: PipeStage,
                  dataloader,
                  num_microbatch: int,
                  recompute=False):

    num_stage = model.num_stages
    prev_rank = model.prev_stage_global_grank
    next_rank = model.next_stage_global_rank

    num_warmup_microbatches = num_stage - 1 - model.stage_local_rank
    num_warmup_microbatches = min(num_warmup_microbatches, num_microbatch)
    num_warmup_remaining = num_microbatch - num_warmup_microbatches

    # warmup
    for i in range(num_warmup_microbatches):
        model.data = next(dataloader)
        # recv forward
        inputs = () if model.is_first_stage else recv_forward(model, prev_rank)
        # forward
        model.push(inputs, 'inputs')
        if recompute:
            with torch.no_grad():
                outputs = forward_step(model, *inputs)
                model.push(None, 'outputs')
        else:
            outputs = forward_step(model, *inputs)
            model.push(outputs, 'outputs')
        # send forward
        send_forward(outputs, next_rank)

    # before running 1f1b: need to recv first forward tensor
    if num_warmup_remaining > 0:
        model.data = next(dataloader)
        inputs = () if model.is_first_stage else recv_forward(model, prev_rank)

    # run 1f1b
    for i in range(num_warmup_remaining):
        model.data = next(dataloader)
        # forward
        model.push(inputs, 'inputs')
        if recompute:
            with torch.no_grad():
                outputs = forward_step(model, *inputs)
                model.push(None, 'outputs')
        else:
            outputs = forward_step(model, *inputs)
            model.push(outputs, 'outputs')

        # send forward recv backward
        grads = (None,)
        if not model.is_last_stage:
            grads = send_forward_recv_backward(outputs, model, next_rank)

        # backward
        inputs, outputs = model.pop('inputs'), model.pop('outputs')
        if recompute:
            assert outputs is None
            outputs = forward_step(model, *inputs)
        input_grads = backward_step(inputs, outputs, grads)
        
        # send backward
        inputs = ()
        if not model.is_first_stage:
            if i != (num_warmup_remaining-1):
                # send backward recv forward
                inputs = send_backward_recv_forward(input_grads, model, prev_rank)
            else:
                # send backward
                send_backward(input_grads, prev_rank)

    # cooldown
    for i in range(num_warmup_microbatches):
        inputs, outputs = model.pop('inputs'), model.pop('outputs')
        # recv backward
        grads = (None,) if model.is_last_stage else recv_backward(model, next_rank)
        # backward
        if recompute:
            assert outputs is None
            outputs = forward_step(model, *inputs)
        input_grads = backward_step(inputs, outputs, grads)
        # send backward
        if not model.is_first_stage:
            send_backward(input_grads, prev_rank)

    model.assert_empty_cached()