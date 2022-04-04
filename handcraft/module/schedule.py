from typing import List
import torch

from cube.profiler.timer import CudaTimer, print_each_rank

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
                # correctness checkprint
                # if model.is_last_stage:
                #     print(outputs)
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


def schedule_tp1f1b_pp2(model: PipeStage,
                        dataloader,
                        num_microbatch: int,
                        recompute=False):
    def tp_encoder_preprocess(model: PipeStage) -> torch.Tensor:
        model.data = next(dataloader)
        enc = model.forward_encoder_shard()
        return (enc,)

    def tp_decoder_preprocess(model: PipeStage) -> torch.Tensor:
        model.data = next(dataloader)
        dec = model.forward_decoder_shard()
        return (dec,)

    def tp_encoder_backward(model: PipeStage):
        enc = model.pop('encoder_sharding_output')
        if model.stage_local_rank == model.first_encoder_stage:
            grads = model.pop('encoder_sharding_grad')
        else:
            grads = (torch.empty_like(enc),)
        backward_step((), (enc,), grads)

    def tp_decoder_backward(model: PipeStage):
        dec = model.pop('decoder_sharding_output')
        if model.stage_local_rank == model.first_decoder_stage:
            grads = model.pop('decoder_sharding_grad')
        else:
            grads = (torch.empty_like(dec),)
        backward_step((), (dec,), grads)

    num_stage = model.num_stages
    rank = model.stage_local_rank
    prev_rank = model.prev_stage_global_grank
    next_rank = model.next_stage_global_rank
    
    output_grads = (None,)
    inputs = ()
    for step in range(num_microbatch * 2 + 2):

        encoder_fmid = step // 2
        encoder_bmid = step - 2
        decoder_fmid = step - 1
        decoder_bmid = step - 3

        # step1: forward sharding 0
        if step % 2 == 0:
            encoder_fmid = step // 2
            encoder_inputs = None
            if 0 <= encoder_fmid and encoder_fmid <= num_microbatch - 1:
                encoder_inputs = tp_encoder_preprocess(model)
        # step1: forward sharding 1
        if step % 2 == 1:
            decoder_fmid = (step - 1) // 2
            decoder_inputs = None
            if 0 <= decoder_fmid and decoder_fmid <= num_microbatch - 1:
                decoder_inputs = tp_decoder_preprocess(model)

        if rank % 2 == 0:
            # do forward
            if step % 2 == 0:
                fmid = step // 2
                do_forward = 0 <= fmid and fmid <= num_microbatch - 1
                if do_forward:
                    model.push(encoder_inputs, 'inputs')
                    if recompute:
                        with torch.no_grad():
                            outputs = forward_step(model, *(), recompute=True)
                        model.push(None, 'outputs')
                    else:
                        outputs = forward_step(model, *())
                        model.push(outputs, 'outputs')
    
                # recompute
                next_bmid = (step + 1 - 3) // 2 if step+1 >= 3 else -1
                do_next_backward = 0 <= next_bmid and next_bmid <= num_microbatch - 1
                if recompute and do_next_backward :
                    outputs_bp = model.pop('outputs')
                    assert outputs_bp is None
                    outputs_bp = forward_step(model, *())
                    model.push_ahead(outputs_bp, 'outputs')

                # send forward recv backward
                if do_forward and do_next_backward:
                    # print(f'rank {rank}: step {step}: send forward recv backward')
                    output_grads = send_forward_recv_backward(outputs, model, next_rank)
                elif do_next_backward:
                    # print(f'rank {rank}: step {step}: recv backward')
                    output_grads = recv_backward(model, next_rank)
                elif do_forward:
                    # print(f'rank {rank}: step {step}: send forward')
                    send_forward(outputs, next_rank)

            # do backward
            else:
                bmid = (step - 3) // 2 if step >= 3 else -1
                if 0 <= bmid and bmid <= num_microbatch - 1:
                    inputs, outputs = model.pop('inputs'), model.pop('outputs')
                    input_grads = backward_step(inputs, outputs, output_grads)
                    output_grads = (None,)
                    assert len(input_grads) == 1
                    model.push(input_grads, 'encoder_sharding_grad')

        if rank % 2 == 1:
            # do backward
            if step % 2 == 0:
                bmid = (step - 2) // 2 if step >= 2 else -1
                do_backward = 0 <= bmid and bmid <= num_microbatch - 1

                # backward
                if do_backward:
                    inputs, outputs = model.pop('inputs'), model.pop('outputs')
                    assert output_grads == (None,)
                    input_grads = backward_step(inputs, outputs, output_grads)
                    assert len(inputs) == 2
                    model.push((input_grads[1],), 'decoder_sharding_grad')
                    input_grads = (input_grads[0],)

                # send backward recv forward
                next_fmid = (step + 1 - 1) // 2
                do_next_forward = 0 <= next_fmid and next_fmid <= num_microbatch - 1
                if do_backward and do_next_forward:
                    # print(f'rank {rank}: step {step}: send backward recv forward')
                    inputs = send_backward_recv_forward(input_grads, model, prev_rank)
                elif do_next_forward:
                    # print(f'rank {rank}: step {step}: recv forward')
                    inputs = recv_forward(model, prev_rank)
                elif do_backward:
                    # print(f'rank {rank}: step {step}: send backward')
                    send_backward(input_grads, prev_rank)
            # do forward
            else:
                # forward
                fmid = (step - 1) // 2
                if 0 <= fmid and fmid <= num_microbatch - 1:
                    assert inputs != ()
                    model.push((inputs[0], decoder_inputs[0]), 'inputs')
                    if recompute:
                        with torch.no_grad():
                            outputs = forward_step(model, *inputs, recompute=True)
                        model.push(None, 'outputs')
                    else:
                        outputs = forward_step(model, *inputs)
                        model.push(outputs, 'outputs')
                
                    # recompute
                    if recompute:
                        inputs, outputs = model.pop('inputs'), model.pop('outputs')
                        assert outputs is None
                        outputs = forward_step(model, *inputs)
                        model.push_ahead(inputs, 'inputs')
                        model.push_ahead(outputs, 'outputs')


        # step3: backward sharding 1
        if step % 2 == 0:
            decoder_bmid = (step - 2) // 2
            if 0 <= decoder_bmid and decoder_bmid <= num_microbatch - 1:
                tp_decoder_backward(model)
        
        # step3: backward sharding 0
        if step % 2 == 1:
            encoder_bmid = (step - 3) // 2
            if 0 <= encoder_bmid and encoder_bmid <= num_microbatch - 1:
                tp_encoder_backward(model)

    model.assert_empty_cached()


def schedule_tp1f1b(model: PipeStage,
                    dataloader,
                    num_microbatch: int,
                    recompute=False):
    # special cases for pipeline stage == 2
    if model.num_stages == 2:
        return schedule_tp1f1b_pp2(model, dataloader, num_microbatch, recompute)
    
    def tp_encoder_preprocess(model: PipeStage) -> torch.Tensor:
        model.data = next(dataloader)
        enc = model.forward_encoder_shard()
        return (enc,)

    def tp_decoder_preprocess(model: PipeStage) -> torch.Tensor:
        model.data = next(dataloader)
        dec = model.forward_decoder_shard()
        return (dec,)

    def tp_encoder_backward(model: PipeStage):
        enc = model.pop('encoder_sharding_output')
        if model.stage_local_rank == model.first_encoder_stage:
            grads = model.pop('encoder_sharding_grad')
        else:
            grads = (torch.empty_like(enc),)
        backward_step((), (enc,), grads)

    def tp_decoder_backward(model: PipeStage):
        dec = model.pop('decoder_sharding_output')
        if model.stage_local_rank == model.first_decoder_stage:
            grads = model.pop('decoder_sharding_grad')
        else:
            grads = (torch.empty_like(dec),)
        backward_step((), (dec,), grads)

    num_stage = model.num_stages
    rank = model.stage_local_rank
    prev_rank = model.prev_stage_global_grank
    next_rank = model.next_stage_global_rank
    fofst = [-(step // 2) for step in range(num_stage)]
    bofst = [-(num_stage - 1 - (step // 2)) for step in range(num_stage)]

    fofst = fofst[model.stage_local_rank]
    bofst = bofst[model.stage_local_rank]
    last_backward = (None,)
    last_forward = (None,)

    for step in range(num_microbatch + num_stage - 1):
        fmid, bmid = step + fofst, step + bofst
        encoder_fmid = step
        decoder_fmid = step - num_stage // 2 // 2
        encoder_bmid = step + 1 - num_stage // 2 * 2
        decoder_bmid = step + 1 - int(num_stage // 2 * 1.5)
        do_backward = 0 <= bmid and bmid <= num_microbatch - 1
        do_forward = 0 <= fmid and fmid <= num_microbatch - 1

        # step1: tp encoder forward
        encoder_inputs = None
        if 0 <= encoder_fmid and encoder_fmid <= num_microbatch - 1:
            encoder_inputs = tp_encoder_preprocess(model)
        # step2: tp decoder forward
        decoder_inputs = None
        if 0 <= decoder_fmid and decoder_fmid <= num_microbatch - 1:
            decoder_inputs = tp_decoder_preprocess(model)

        # step 3: forward + backward
        if rank % 2 == 0:
            # inter-barrier
            inputs = ()
            if not model.is_first_stage:
                if do_forward and last_backward != (None,):
                    # print(f'rank {rank} send backward grad + recv forward output ')
                    inputs = send_backward_recv_forward(last_backward, model, prev_rank)
                elif do_forward:
                    # print(f'rank {rank} recv forward output ')
                    inputs = recv_forward(model, prev_rank)
                elif last_backward != (None,):
                    # print(f'rank {rank} send backward grad ')
                    send_backward(last_backward, prev_rank)

            # forward
            if do_forward:

                if model.stage_local_rank == model.first_encoder_stage and encoder_inputs is not None:
                    model.push(encoder_inputs, 'inputs')
                elif model.stage_local_rank == model.first_decoder_stage and decoder_inputs is not None:
                    assert len(inputs) == 1 and len(decoder_inputs) == 1
                    model.push((inputs[0], decoder_inputs[0]), 'inputs')
                else:
                    model.push(inputs, 'inputs')
    
                if recompute:
                    with torch.no_grad():
                        outputs = forward_step(model, *inputs, recompute=True)
                    model.push(None, 'outputs')
                else:
                    outputs = forward_step(model, *inputs)
                    model.push(outputs, 'outputs')

            # recompute if backward is needed
            if do_backward:
                inputs, outputs_bp = model.pop('inputs'), model.pop('outputs')
                if recompute:
                    assert outputs_bp is None
                    outputs_bp = forward_step(model, *inputs)

            # intra-barrier send recv
            output_grads = (None,)
            if (do_forward and not model.is_last_stage) and (do_backward and not model.is_last_stage):
                # send forward recv backward
                # print(f'rank {rank} recv backward grad + send forward output ')
                output_grads = send_forward_recv_backward(outputs, model, next_rank)
            elif do_forward and not model.is_last_stage:
                # print(f'rank {rank} send forward output ')
                send_forward(outputs, next_rank)
            elif do_backward and not model.is_last_stage:
                # print(f'rank {rank} recv backward grad ')
                output_grads = recv_backward(model, next_rank)

            # backward
            last_backward = (None,)
            if do_backward:
                # inputs, outputs = input_tensors.pop(0), output_tensors.pop(0)
                input_grads = backward_step(inputs, outputs_bp, output_grads)

                if model.stage_local_rank == model.first_encoder_stage:
                    assert len(input_grads) == 1
                    model.push(input_grads, 'encoder_sharding_grad')
                elif model.stage_local_rank == model.first_decoder_stage:
                    assert len(input_grads) == 2
                    model.push((input_grads[1],), 'decoder_sharding_grad')
                    input_grads = (input_grads[0],)
                last_backward = input_grads

        # step 3: backward + forward
        if rank % 2 == 1:
            # inter-barrier
            if model.is_last_stage:
                output_grads = (None,)
            else:
                if do_backward and last_forward != (None,):
                    # print(f'rank {rank} recv backward grad + send forward output ')
                    output_grads = send_forward_recv_backward(last_forward, model, next_rank)
                elif do_backward:
                    # print(f'rank {rank} recv backward grad ')
                    output_grads = recv_backward(model, next_rank)
                elif last_forward != (None,):
                    # print(f'rank {rank} send forward output ')
                    send_forward(last_forward, next_rank)

            # backward
            last_backward = (None,)
            if do_backward:
                inputs, outputs_bp = model.pop('inputs'), model.pop('outputs')
                # backward
                input_grads = backward_step(inputs, outputs_bp, output_grads)
                last_backward = input_grads
            
            # intra-barrier
            if do_backward and do_forward:
                # print(f'rank {rank} send backward grad + recv forward output ')
                inputs = send_backward_recv_forward(input_grads, model, prev_rank)
            elif do_backward:
                # print(f'rank {rank} send backward grad ')
                send_backward(input_grads, prev_rank)
            elif do_forward:
                # print(f'rank {rank} recv forward output ')
                inputs = recv_forward(model, prev_rank)

            # forward
            last_forward = (None,)
            if do_forward:
                # forward step
                model.push(inputs, 'inputs')
                if recompute:
                    with torch.no_grad():
                        outputs = forward_step(model, *inputs, recompute=True)
                        model.push(None, 'outputs')
                        # correctness check print
                        # if model.is_last_stage:
                        #     print(outputs)
                else:
                    outputs = forward_step(model, *inputs)
                    model.push(outputs, 'outputs')
                last_forward = outputs

            next_backward = 0 <= (bmid+1) and (bmid+1) <= num_microbatch - 1
            if next_backward:
                if recompute:
                    inputs, outputs_bp = model.pop('inputs'), model.pop('outputs')
                    assert outputs_bp is None
                    outputs = forward_step(model, *inputs)
                    model.push_ahead(inputs, 'inputs')
                    model.push_ahead(outputs, 'outputs')

        # step 4: sharding decoder backward
        if 0 <= decoder_bmid and decoder_bmid <= num_microbatch - 1:
            tp_decoder_backward(model)

        # step 5: sharding encoder backward
        if 0 <= encoder_bmid and encoder_bmid <= num_microbatch - 1:
            tp_encoder_backward(model)

    model.assert_empty_cached()
