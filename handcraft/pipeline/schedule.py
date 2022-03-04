from typing import List
import torch

from cube.profiler.timer import CudaTimer, print_each_rank
import cube.runtime.adapter.collectives as coll
from cube.runtime.device import DeviceGroup


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
                  output_tensor_grads: List[torch.Tensor]):
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


def is_first_stage():
    return DeviceGroup().rank == 0


def is_last_stage():
    return DeviceGroup().rank == DeviceGroup().world_size - 1


def recv_input(model, dataloader, prev_rank: int):
    if is_first_stage():
        return next(dataloader)
    else:
        return coll.recv(model.input_shape(), prev_rank, model.input_dtype())


def schedule_naive(model, dataloader, num_microbatch: int):
    rank = DeviceGroup().rank
    next_rank = (DeviceGroup().rank + 1) % DeviceGroup().world_size
    prev_rank = (DeviceGroup().rank - 1) % DeviceGroup().world_size
    for _ in range(num_microbatch):
        # recv forward
        if is_first_stage():
            input = next(dataloader)
        else:
            # print(f'rank {rank} recving forward input...')
            input = coll.recv(model.input_shape(), prev_rank, model.input_dtype())
        # forward
        output = forward_step(model, input)
        # send forward
        if not is_last_stage():
            # print(f'rank {rank} sending forward output...')
            coll.send(output, next_rank)
        # recv backward
        output_grad = None
        if not is_last_stage():
            # print(f'rank {rank} recving backward input...')
            output_grad = coll.recv(output.size(), next_rank, output.dtype)
        # backward
        input_grad = backward_step([input], [output], [output_grad])[0]
        # send backward
        if not is_first_stage():
            # print(f'rank {rank} sending backward output...')
            coll.send(input_grad, prev_rank)


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
    for step in range(num_microbatch + 2):
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

        if rank != 0 and rank % 2 == 0:
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

        if rank == 1:

            # forward
            if do_forward:
                input = output_1st
                output = forward_step(model, input)
                input_tensors.append(input)
                output_tensors.append(output)

            # intra-barrier send recv
            if do_forward and do_backward:
                # send forward recv backward
                # print(f'rank {rank} recv backward grad + send forward output ')
                output_grad = coll.sendrecv(
                    [output], [output.size()], [output.dtype],
                    [next_rank], [next_rank]
                )[0]
            elif do_forward:
                # print(f'rank {rank} send forward output ')
                coll.send(output, next_rank)
            elif do_backward:
                # print(f'rank {rank} recv backward grad ')
                output_grad = coll.recv(model.output_shape(), next_rank, model.output_dtype())
            
            # backward
            if do_backward:
                input, output = input_tensors.pop(0), output_tensors.pop(0)
                input_grad = backward_step([input], [output], [output_grad])[0]
                last_backward = input_grad
            
        if rank != 1 and rank % 2 == 1:

            # inter-barrier
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
            if do_backward:
                input, output = input_tensors.pop(0), output_tensors.pop(0)
                input_grad = backward_step([input], [output], [output_grad])[0]
                last_backward = input_grad

        # step3: tp backward
        if 0 <= (step-num_stage+2) and (step-num_stage+2) <= num_microbatch - 1:
            # print(f'rank {rank} backward tp model ')
            tp_backward(last_backward)

        # print_each_rank(f'=========end rank {rank}=========')


def schedule_1f1b(model: torch.nn.Module,
                  dataloader,
                  num_microbatch: int):
    group = list(range(DeviceGroup().world_size))
    rank = DeviceGroup().rank
    next_rank = (DeviceGroup().rank + 1) % DeviceGroup().world_size
    prev_rank = (DeviceGroup().rank - 1) % DeviceGroup().world_size

    input_tensors = list()
    output_tensors = list()

    # warmup
    num_warmup_microbatch = DeviceGroup().world_size - 1 - rank
    for mid in range(num_warmup_microbatch):
        # recv forward
        input = recv_input(model, dataloader, prev_rank)
        # forward
        output = forward_step(model, input)
        # send forward
        coll.send(output, next_rank)
        input_tensors.append(input)
        output_tensors.append(output)

    num_warmup_remaining = num_microbatch - num_warmup_microbatch
    if num_warmup_remaining > 0:
        input = recv_input(model, dataloader, prev_rank)

    # steady
    for i in range(num_warmup_microbatch):
        # forward
        output = forward_step(model, input)
        # send forward + recv backward
        grad = coll.sendrecv(
            [output],
            [list(output.size())], [output.dtype],
            [next_rank], [next_rank]
        )[0]
        input_tensors.append(input)
        output_tensors.append(output)
        # backward
        input, output = input_tensors.pop(0), output_tensors.pop(0)
        input_grad = backward_step([input], [output], [grad])
        # send backward recv forward
        if i != (num_warmup_remaining-1):
            input = coll.sendrecv(
                [input_grad],
                (list(input.size()),), (input.dtype,),
                [prev_rank], [prev_rank]
            )
        else:
            # send backward
            coll.send(input_grad, prev_rank)

    # cooldown
    for i in range(num_warmup_microbatch):
        input, output = input_tensors.pop(0), output_tensors.pop(0)
        # recv backward
        grad = coll.recv(list(output.size()), next_rank, dtype=output.dtype)
        # backward
        grad = backward_step([input], [output], [grad])
        # send backward
        coll.send(grad, prev_rank)

