from typing import List, Tuple
import torch

from cube.profiler.timer import CudaTimer, print_each_rank
from cube.profiler.memory import memory_summary
import cube.runtime.adapter.collectives as coll
from cube.runtime.device import DeviceGroup

io_input = input

def forward_step(model, *args, **kwargs):
    """
    Forward pass
    """
    CudaTimer().start("forward")
    output = model(*args, **kwargs)
    CudaTimer().stop("forward")
    return output


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


def recv_forward(model, prev_rank: int) -> List[torch.Tensor]:
    CudaTimer().start(field_name='comm')
    shapes = model.input_shape()
    dtypes = model.input_dtype()
    if len(shapes) == 0:
        return ()
    # print(f'rank {DeviceGroup().rank} recving forward: {shapes}')
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


def recv_backward(model, next_rank: int) -> List[torch.Tensor]:
    CudaTimer().start(field_name='comm')
    shapes = model.output_shape()
    dtypes = model.output_dtype()
    if len(shapes) == 0:
        return ()
    # print(f'rank {DeviceGroup().rank} recving backward: {shapes}')
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
    if len(outputs) == 0:
        return
    CudaTimer().start(field_name='comm')
    # print(f'rank {DeviceGroup().rank} sending forward: {[tuple(t.size()) for t in outputs]}')
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
    if len(grads) == 0:
        return
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


def send_forward_recv_backward(outputs, model, next_rank: int) -> List[torch.Tensor]:
    CudaTimer().start(field_name='comm')
    shapes = model.output_shape()
    dtypes = model.output_dtype()
    # print(f'rank {DeviceGroup().rank} sending forward: {[tuple(t.size()) for t in outputs]} recving backward {shapes}')
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
    reqs = torch.distributed.batch_isend_irecv(send_ops)
    for req in reqs:
        req.wait()
    torch.cuda.synchronize()
    CudaTimer().stop(field_name='comm')
    return tensors


def send_backward_recv_forward(grads, model, prev_rank: int) -> List[torch.Tensor]:
    CudaTimer().start(field_name='comm')
    shapes = model.input_shape()
    dtypes = model.input_dtype()
    # print(f'rank {DeviceGroup().rank} sending backward: {[tuple(t.size()) for t in grads]} recving forward {shapes}')
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
    reqs = torch.distributed.batch_isend_irecv(send_ops)
    for req in reqs:
        req.wait()
    torch.cuda.synchronize()
    CudaTimer().stop(field_name='comm')
    return tensors



def schedule_naive(model, num_microbatch: int, neighbors: Tuple[int, int]):
    """
    neighbors: (prev_rank: int, next_rank: int)
    """
    rank = DeviceGroup().rank
    prev_rank, next_rank = neighbors
    
    is_first_stage = rank < prev_rank
    is_last_stage = rank > next_rank

    for step in range(num_microbatch):
        # print(f'rank {rank} recving forward input...')
        inputs = () if is_first_stage else recv_forward(model, prev_rank)
        # forward
        outputs = forward_step(model, *inputs)
        # send forward
        if not is_last_stage:
            # print(f'rank {rank} sending forward output...')
            send_forward(outputs, next_rank)
        # recv backward
        # print(f'rank {rank} recving backward input...')
        output_grads = (None,) if is_last_stage else recv_backward(model, next_rank)
        # backward
        input_grads = backward_step(inputs, outputs, output_grads)
        # send backward
        if not is_first_stage:
            # print(f'rank {rank} sending backward output...')
            send_backward(input_grads, prev_rank)

        # memory_summary()
        # if rank == 0:
        #     io_input(f'{step}>>>')
        # torch.distributed.barrier()


def schedule_tp_1f1b(model: torch.nn.Module,
                     first_stage_model: torch.nn.Module,
                     dataloader,
                     num_microbatch: int,
                     num_stage: int):
    rank = DeviceGroup().rank
    next_rank = (DeviceGroup().rank + 1) % DeviceGroup().world_size
    prev_rank = (DeviceGroup().rank - 1) % DeviceGroup().world_size

    input_tensors = list()
    output_tensors = list()

    input_1st_tensors = list()
    output_1st_tensors = list()

    gather_list = list(range(num_stage))
    gather_list[0], gather_list[1] = gather_list[1], gather_list[0]

    def tp_forward(fmodel, dataloader) -> torch.Tensor:
        input = next(dataloader)
        output = forward_step(fmodel, input)
        input_1st_tensors.append(input)
        output_1st_tensors.append(output)
        # gather
        outputs = coll.gather([output], None, None, gather_list)
        if rank == 1:
            with torch.no_grad():
                outputs[0], outputs[1] = outputs[1], outputs[0]
                output = torch.cat(tuple(outputs), dim=-1)
            output = output.requires_grad_()
        else:
            output = None
        return output

    def tp_backward(grad: torch.Tensor):
        if rank == 1:
            with torch.no_grad():
                grads = list(grad.chunk(num_stage, dim=-1))
                grads[0], grads[1] = grads[1], grads[0]
        else:
            grads = None
        input_1st, output_1st = input_1st_tensors.pop(0), output_1st_tensors.pop(0)
        grad_1st = coll.scatter(grads, [output_1st.size()], [output_1st.dtype], gather_list)
        backward_step([input_1st], [output_1st], [grad_1st])[0]

    fofst = [0] + [-(step // 2) for step in range(num_stage-1)]
    bofst = [0] + [-(num_stage - 2 - (step // 2)) for step in range(num_stage-1)]
    # print(fofst)
    # print(bofst)
    fofst = fofst[rank]
    bofst = bofst[rank]
    last_backward = None
    last_forward = None
    for step in range(num_microbatch + num_stage - 2):
        torch.distributed.barrier()
        # print_each_rank(f'=========begin rank {rank}=========')
        fmid, bmid = step + fofst, step + bofst
        do_backward = 0 <= bmid and bmid <= num_microbatch - 1
        do_forward = 0 <= fmid and fmid <= num_microbatch - 1
    
        # step1: tp forward
        if 0 <= step and step <= num_microbatch - 1:
            # print(f'rank {rank} forward tp model ')
            output_1st = tp_forward(first_stage_model, dataloader)

        # step2: backward + forward
        if rank == 0:
            pass

        if rank % 2 == 0 and rank != 0:
            # inter-barrier
            if do_backward and last_forward is not None:
                # print(f'rank {rank} recv backward grad + send forward output ')
                output_grad = coll.sendrecv(
                    [last_forward], [model.output_shape()], [model.output_dtype()],
                    [next_rank], [next_rank]
                )[0]
            elif do_backward:
                # print(f'rank {rank} recv backward grad ')
                output_grad = coll.recv(model.output_shape(), next_rank, model.output_dtype())
            elif last_forward is not None:
                # print(f'rank {rank} send forward output ')
                coll.send(last_forward, next_rank)

            # backward
            if do_backward:
                input, output = input_tensors.pop(0), output_tensors.pop(0)
                # backward
                input_grad = backward_step([input], [output], [output_grad])[0]
            
            # intra-barrier
            if do_backward and do_forward:
                # print(f'rank {rank} send backward grad + recv forward output ')
                input = coll.sendrecv(
                    [input_grad], [model.input_shape()], [model.input_dtype()],
                    [prev_rank], [prev_rank]
                )[0]
            elif do_backward:
                # print(f'rank {rank} send backward grad ')
                coll.send(input_grad, prev_rank)
            elif do_forward:
                # print(f'rank {rank} recv forward output ')
                input = coll.recv(model.input_shape(), prev_rank, model.input_dtype())

            # forward
            last_forward = None
            if do_forward:
                # forward step
                output = forward_step(model, input)
                input_tensors.append(input)
                output_tensors.append(output)
                last_forward = output
            
        if rank % 2 == 1:
            # inter-barrier
            if rank == 1:
                input = output_1st
            else:
                if do_forward and last_backward is not None:
                    # print(f'rank {rank} send backward grad + recv forward output ')
                    input = coll.sendrecv(
                        [input_grad], [model.input_shape()], [model.input_dtype()],
                        [prev_rank], [prev_rank]
                    )[0]
                elif do_forward:
                    # print(f'rank {rank} recv forward output ')
                    input = coll.recv(model.input_shape(), prev_rank, model.input_dtype())
                elif last_backward is not None:
                    # print(f'rank {rank} send backward grad ')
                    coll.send(last_backward, prev_rank)

            # forward
            if do_forward:
                output = forward_step(model, input)
                input_tensors.append(input)
                output_tensors.append(output)

            # intra-barrier send recv
            output_grad = None
            if (do_forward and not is_last_stage()) and (do_backward and not is_last_stage()):
                # send forward recv backward
                # print(f'rank {rank} recv backward grad + send forward output ')
                output_grad = coll.sendrecv(
                    [output], [output.size()], [output.dtype],
                    [next_rank], [next_rank]
                )[0]
            elif do_forward and not is_last_stage():
                # print(f'rank {rank} send forward output ')
                coll.send(output, next_rank)
            elif do_backward and not is_last_stage():
                # print(f'rank {rank} recv backward grad ')
                output_grad = coll.recv(model.output_shape(), next_rank, model.output_dtype())

            # backward + forward
            last_backward = None
            if do_backward:
                input, output = input_tensors.pop(0), output_tensors.pop(0)
                input_grad = backward_step([input], [output], [output_grad])[0]
                last_backward = input_grad

        # step3: tp backward
        if 0 <= (step-num_stage+2) and (step-num_stage+2) <= num_microbatch - 1:
            # print(f'rank {rank} backward tp model ')
            tp_backward(last_backward)

        # if rank == 0:
        #     io_input(f'{step}>>>')
        # torch.distributed.barrier()

    assert len(input_tensors) == 0
    assert len(output_tensors) == 0
    assert len(input_1st_tensors) == 0
    assert len(output_1st_tensors) == 0

        # print_each_rank(f'=========end rank {rank}=========')


def schedule_tp_1f1b_pack(model: torch.nn.Module,
                     first_stage_model: torch.nn.Module,
                     dataloader,
                     num_microbatch: int,
                     num_stage: int):
    rank = DeviceGroup().rank
    next_rank = (DeviceGroup().rank + 1) % DeviceGroup().world_size
    prev_rank = (DeviceGroup().rank - 1) % DeviceGroup().world_size

    input_tensors = list()
    output_tensors = list()

    input_1st_tensors = list()
    output_1st_tensors = list()

    def tp_forward(fmodel, dataloader) -> torch.Tensor:
        input = next(dataloader)
        #TODO: gather
        output = forward_step(fmodel, input)
        input_1st_tensors.append(input)
        output_1st_tensors.append(output)
        output = output.detach().requires_grad_()
        return output

    def tp_backward(grad: torch.Tensor):
        input_1st, output_1st = input_1st_tensors.pop(0), output_1st_tensors.pop(0)
        if rank != 0:
            grad = torch.empty_like(output_1st)
        torch.distributed.broadcast(grad, src=0)
        backward_step([input_1st], [output_1st], [grad])[0]

    fofst = [-(step // 2) for step in range(num_stage)]
    bofst = [-(num_stage - 1 - (step // 2)) for step in range(num_stage)]
    # print(fofst)
    # print(bofst)
    fofst = fofst[rank]
    bofst = bofst[rank]
    last_backward = None
    last_forward = None
    for step in range(num_microbatch + num_stage - 1):
        torch.distributed.barrier()
        # print_each_rank(f'=========begin rank {rank}=========')
        fmid, bmid = step + fofst, step + bofst
        do_backward = 0 <= bmid and bmid <= num_microbatch - 1
        do_forward = 0 <= fmid and fmid <= num_microbatch - 1
    
        # step1: tp forward
        if 0 <= step and step <= num_microbatch - 1:
            # print(f'rank {rank} forward tp model ')
            output_1st = tp_forward(first_stage_model, dataloader)

        # forward + backward
        if rank % 2 == 0:
            # inter-barrier
            if rank == 0:
                input = output_1st
            else:
                if do_forward and last_backward is not None:
                    # print(f'rank {rank} send backward grad + recv forward output ')
                    input = coll.sendrecv(
                        [input_grad], [model.input_shape()], [model.input_dtype()],
                        [prev_rank], [prev_rank]
                    )[0]
                elif do_forward:
                    # print(f'rank {rank} recv forward output ')
                    input = coll.recv(model.input_shape(), prev_rank, model.input_dtype())
                elif last_backward is not None:
                    # print(f'rank {rank} send backward grad ')
                    coll.send(last_backward, prev_rank)

            # forward
            if do_forward:
                output = forward_step(model, input)
                input_tensors.append(input)
                output_tensors.append(output)

            # mem = torch.cuda.max_memory_allocated()
            # print(f'rank {rank}: {mem / 1024 / 1024 / 1024} GB forward')

            # intra-barrier send recv
            output_grad = None
            if (do_forward and not is_last_stage()) and (do_backward and not is_last_stage()):
                # send forward recv backward
                # print(f'rank {rank} recv backward grad + send forward output ')
                output_grad = coll.sendrecv(
                    [output], [output.size()], [output.dtype],
                    [next_rank], [next_rank]
                )[0]
            elif do_forward and not is_last_stage():
                # print(f'rank {rank} send forward output ')
                coll.send(output, next_rank)
            elif do_backward and not is_last_stage():
                # print(f'rank {rank} recv backward grad ')
                output_grad = coll.recv(model.output_shape(), next_rank, model.output_dtype())

            # backward
            last_backward = None
            if do_backward:
                input, output = input_tensors.pop(0), output_tensors.pop(0)
                input_grad = backward_step([input], [output], [output_grad])[0]
                last_backward = input_grad

        # backward + forward
        if rank % 2 == 1:
            # inter-barrier
            if is_last_stage():
                output_grad = None
            else:
                if do_backward and last_forward is not None:
                    # print(f'rank {rank} recv backward grad + send forward output ')
                    output_grad = coll.sendrecv(
                        [last_forward], [model.output_shape()], [model.output_dtype()],
                        [next_rank], [next_rank]
                    )[0]
                elif do_backward:
                    # print(f'rank {rank} recv backward grad ')
                    output_grad = coll.recv(model.output_shape(), next_rank, model.output_dtype())
                elif last_forward is not None:
                    # print(f'rank {rank} send forward output ')
                    coll.send(last_forward, next_rank)

            # backward
            if do_backward:
                input, output = input_tensors.pop(0), output_tensors.pop(0)
                # backward
                input_grad = backward_step([input], [output], [output_grad])[0]
            
            # intra-barrier
            if do_backward and do_forward:
                # print(f'rank {rank} send backward grad + recv forward output ')
                input = coll.sendrecv(
                    [input_grad], [model.input_shape()], [model.input_dtype()],
                    [prev_rank], [prev_rank]
                )[0]
            elif do_backward:
                # print(f'rank {rank} send backward grad ')
                coll.send(input_grad, prev_rank)
            elif do_forward:
                # print(f'rank {rank} recv forward output ')
                input = coll.recv(model.input_shape(), prev_rank, model.input_dtype())

            # forward
            last_forward = None
            if do_forward:
                # forward step
                output = forward_step(model, input)
                input_tensors.append(input)
                output_tensors.append(output)
                last_forward = output

        # step3: tp backward
        if 0 <= (step-num_stage+1) and (step-num_stage+1) <= num_microbatch - 1:
            # print(f'rank {rank} backward tp model ')
            tp_backward(last_backward)

        # memory_summary()
        # if rank == 0:
        #     io_input(f'{step}>>>')
        # torch.distributed.barrier()
        # print_each_rank(f'=========end rank {rank}: {step}=========')

    assert len(input_tensors) == 0
    assert len(output_tensors) == 0
    assert len(input_1st_tensors) == 0
    assert len(output_1st_tensors) == 0

        # print_each_rank(f'=========end rank {rank}=========')