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
    reqs = torch.distributed.batch_isend_irecv(ops)
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
    reqs = torch.distributed.batch_isend_irecv(ops)
    for req in reqs:
        req.wait()
    torch.cuda.synchronize()
    CudaTimer().stop(field_name='comm')
    return tensors



def schedule_naive(model, dataloader, num_microbatch: int, neighbors: Tuple[int, int]):
    """
    neighbors: (prev_rank: int, next_rank: int)
    """
    rank = DeviceGroup().rank
    prev_rank, next_rank = neighbors
    
    is_first_stage = rank < prev_rank
    is_last_stage = rank > next_rank

    for step in range(num_microbatch):
        model.set_inputs(*next(dataloader))
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


def schedule_tp_1f1b_pack(model: torch.nn.Module,
                          dataloader,
                          num_microbatch: int,
                          num_stage: int,
                          neighbors: Tuple[int, int]):
    rank = DeviceGroup().rank
    prev_rank, next_rank = neighbors

    is_first_stage = rank < prev_rank
    # FIXME: only work for pure pipeline
    is_first_decoder_stage = (rank == num_stage // 2)
    is_last_stage = rank > next_rank
    last_stage = torch.distributed.get_world_size() - 1

    input_tensors = list()
    output_tensors = list()

    input_head_tensors = list()
    output_head_tensors = list()

    def tp_head_forward() -> torch.Tensor:
        src_tokens, prev_output_tokens = next(dataloader)
        model.set_inputs(*(src_tokens, prev_output_tokens))
        enc = model.forward_encoder_preprocess(dst=0)[0]
        dec = model.forward_decoder_preprocess(dst=num_stage // 2)[0]
        input_head_tensors.append((src_tokens, prev_output_tokens))
        output_head_tensors.append((enc, dec))
        enc = enc.detach().requires_grad_()
        dec = dec.detach().requires_grad_()
        # FIXME: this will change decoder input
        if is_first_stage:
            model.set_preprocess(enc=enc)
        if is_first_decoder_stage:
            model.set_preprocess(dec=dec)
        if is_first_stage:
            return (enc,)
        if is_first_decoder_stage:
            return (dec,)
        else:
            return ()

    def tp_head_backward(grads: Tuple[torch.Tensor]):
        inputs_head, outputs_head = input_head_tensors.pop(0), output_head_tensors.pop(0)
        # encoder backward
        enc, dec = outputs_head
        if not is_first_stage:
            grads = (torch.empty_like(enc),)
        # decoder backward
        backward_step((), (enc,), grads)
        #FIXME: grads is using enc gradient!!!
        if not is_first_decoder_stage:
            grads = (torch.empty_like(dec),)
        backward_step((), (dec,), grads)

    def tp_tail_forward_backward(outputs: Tuple[torch.Tensor]):
        dec = None
        if is_last_stage:
            assert len(outputs) == 1
            dec = outputs[0]
            dec = dec.detach().requires_grad_()
        loss = model.forward_postprocess(dec, src=last_stage)
        grads = backward_step((dec,), loss, (None,))
        return grads

    fofst = [-(step // 2) for step in range(num_stage)]
    bofst = [-(num_stage - 1 - (step // 2)) for step in range(num_stage)]
    # print(fofst)
    # print(bofst)
    fofst = fofst[rank]
    bofst = bofst[rank]
    last_backward = (None,)
    last_forward = (None,)
    tail_grads = (None,)
    for step in range(num_microbatch + num_stage - 1):
        torch.distributed.barrier()
        # print_each_rank(f'=========begin rank {rank}=========')
        fmid, bmid = step + fofst, step + bofst
        do_backward = 0 <= bmid and bmid <= num_microbatch - 1
        do_forward = 0 <= fmid and fmid <= num_microbatch - 1
    
        # step1: tp forward
        if 0 <= step and step <= num_microbatch - 1:
            # print(f'rank {rank} forward tp model ')
            inputs = tp_head_forward()

        # forward + backward
        if rank % 2 == 0:
            # inter-barrier
            if is_first_stage:
                inputs = inputs
            else:
                if do_forward and last_backward != (None,):
                    # print(f'rank {rank} send backward grad + recv forward output ')
                    inputs = send_backward_recv_forward(last_backward, model, prev_rank)
                    # input = coll.sendrecv(
                    #     [input_grad], [model.input_shape()], [model.input_dtype()],
                    #     [prev_rank], [prev_rank]
                    # )[0]
                elif do_forward:
                    # print(f'rank {rank} recv forward output ')
                    inputs = recv_forward(model, prev_rank)
                    # input = coll.recv(model.input_shape(), prev_rank, model.input_dtype())
                elif last_backward != (None,):
                    # print(f'rank {rank} send backward grad ')
                    send_backward(last_backward, prev_rank)
                    # coll.send(last_backward, prev_rank)

            # forward
            if do_forward:
                input_tensors.append(inputs)
                if is_first_stage:
                    inputs = ()
                outputs = forward_step(model, *inputs)
                output_tensors.append(outputs)

            # mem = torch.cuda.max_memory_allocated()
            # print(f'rank {rank}: {mem / 1024 / 1024 / 1024} GB forward')

            # intra-barrier send recv
            output_grads = (None,)
            if (do_forward and not is_last_stage) and (do_backward and not is_last_stage):
                # send forward recv backward
                # print(f'rank {rank} recv backward grad + send forward output ')
                output_grads = send_forward_recv_backward(outputs, model, next_rank)
                # output_grads = coll.sendrecv(
                #     [output], [output.size()], [output.dtype],
                #     [next_rank], [next_rank]
                # )[0]
            elif do_forward and not is_last_stage:
                # print(f'rank {rank} send forward output ')
                send_forward(outputs, next_rank)
                # coll.send(output, next_rank)
            elif do_backward and not is_last_stage:
                # print(f'rank {rank} recv backward grad ')
                output_grads = recv_backward(model, next_rank)
                # output_grad = coll.recv(model.output_shape(), next_rank, model.output_dtype())

            # backward
            last_backward = (None,)
            if do_backward:
                inputs, outputs = input_tensors.pop(0), output_tensors.pop(0)
                input_grads = backward_step(inputs, outputs, output_grads)
                last_backward = input_grads

        # backward + forward
        if rank % 2 == 1:
            # inter-barrier
            if is_last_stage:
                output_grads = tail_grads
            else:
                if do_backward and last_forward != (None,):
                    # print(f'rank {rank} recv backward grad + send forward output ')
                    output_grads = send_forward_recv_backward(last_forward, model, next_rank)
                    # output_grad = coll.sendrecv(
                    #     [last_forward], [model.output_shape()], [model.output_dtype()],
                    #     [next_rank], [next_rank]
                    # )[0]
                elif do_backward:
                    # print(f'rank {rank} recv backward grad ')
                    output_grads = recv_backward(model, next_rank)
                    # output_grad = coll.recv(model.output_shape(), next_rank, model.output_dtype())
                elif last_forward != (None,):
                    # print(f'rank {rank} send forward output ')
                    send_forward(last_forward, next_rank)
                    # coll.send(last_forward, next_rank)

            # backward
            last_backward = (None,)
            if do_backward:
                inputs, outputs = input_tensors.pop(0), output_tensors.pop(0)
                # backward
                input_grads = backward_step(inputs, outputs, output_grads)
                last_backward = input_grads
            
            # intra-barrier
            if do_backward and do_forward:
                # print(f'rank {rank} send backward grad + recv forward output ')
                inputs = send_backward_recv_forward(input_grads, model, prev_rank)
                # input = coll.sendrecv(
                #     [input_grad], [model.input_shape()], [model.input_dtype()],
                #     [prev_rank], [prev_rank]
                # )[0]
            elif do_backward:
                # print(f'rank {rank} send backward grad ')
                send_backward(input_grads, prev_rank)
                # coll.send(input_grad, prev_rank)
            elif do_forward:
                # print(f'rank {rank} recv forward output ')
                inputs = recv_forward(model, prev_rank)
                # input = coll.recv(model.input_shape(), prev_rank, model.input_dtype())

            # forward
            last_forward = (None,)
            if do_forward:
                # forward step
                outputs = forward_step(model, *inputs)
                input_tensors.append(inputs)
                output_tensors.append(outputs)
                last_forward = outputs

        # tp tail forward-backward
        last_stage_mid = step - (num_stage - 1) // 2
        if 0 <= last_stage_mid and last_stage_mid <= num_microbatch - 1:
            tail_grads = tp_tail_forward_backward(last_forward) 

        # step 4: tp encoder and decoder backward
        encoder_mid = step + 1 - num_stage
        if 0 <= encoder_mid and encoder_mid <= num_microbatch - 1:
            tp_head_backward(last_backward)

        # memory_summary()
        # if rank == 0:
        #     io_input(f'{step}>>>')
        # torch.distributed.barrier()
        # print_each_rank(f'=========end rank {rank}: {step}=========')

    assert len(input_tensors) == 0
    assert len(output_tensors) == 0
    assert len(input_head_tensors) == 0
    assert len(output_head_tensors) == 0

        # print_each_rank(f'=========end rank {rank}=========')