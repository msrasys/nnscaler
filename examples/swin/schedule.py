import torch

from cube.profiler.timer import CudaTimer


def is_last_stage():
    return torch.distributed.get_rank() == torch.distributed.get_world_size() - 1


#================= WhatToDO functions ==================#

def forward_step(model, image, trans_input=None):
    CudaTimer().start("forward")
    output = model(image, trans_input)
    CudaTimer().stop("forward")
    return output


def backward_step(feature_map, output_tensor, output_tensor_grad):
    """
    Calculate input tensor gradient
    """
    if feature_map is not None and feature_map.requires_grad:
        feature_map.retain_grad()
    CudaTimer().start("backward")
    torch.autograd.backward(output_tensor, grad_tensors=output_tensor_grad)
    CudaTimer().stop("backward")
    input_tensor_grad = None
    if feature_map is not None and feature_map.requires_grad:
        input_tensor_grad = feature_map.grad
    return input_tensor_grad

#================= WhatToDO functions ==================#

#================= Between Stage functions ==================#

def send(tensors, to_rank):
    """
    send tensor to the target rank
    """
    if to_rank < 0 or to_rank >= torch.distributed.get_world_size():
        return None
    assert isinstance(tensors, list) or isinstance(tensors, tuple)
    CudaTimer().start("send")
    reqs = list()
    for tensor in tensors:
        if tensor is None:
            continue
        elif torch.is_tensor(tensor):
            send_op = torch.distributed.P2POp(
                torch.distributed.isend, tensor, to_rank
            )
            reqs.append(send_op)
        else:
            raise RuntimeError("Expected tensor or None")
    reqs = torch.distributed.batch_isend_irecv(reqs)
    for req in reqs:
        req.wait()
    torch.cuda.synchronize()
    CudaTimer().stop("send")


def recv(shapes, from_rank, dtype=torch.float):
    if from_rank < 0 or from_rank >= torch.distributed.get_world_size():
        return [None] * len(shapes)
    assert isinstance(shapes, list) or isinstance(shapes, tuple)
    CudaTimer().start("recv")
    reqs = list()
    recved_tensors = list()
    for shape in shapes:
        if shape is None:
            recved_tensors.append(None)
            continue
        tensor = torch.empty(
            shape, requires_grad=True, device=torch.cuda.current_device(),
            dtype=dtype
        )
        recved_tensors.append(tensor)
        recv_op = torch.distributed.P2POp(
            torch.distributed.irecv, tensor, from_rank
        )
        reqs.append(recv_op)
    reqs = torch.distributed.batch_isend_irecv(reqs)
    for req in reqs:
        req.wait()
    torch.cuda.synchronize()
    CudaTimer().stop("recv")
    return recved_tensors


def send_and_recv(send_tensors, recv_shapes, rank, dtype=torch.float):
    if rank < 0 or rank >= torch.distributed.get_world_size():
        return [None] * len(recv_shapes)
    assert isinstance(send_tensors, list) or isinstance(send_tensors, tuple)
    assert isinstance(recv_shapes, list) or isinstance(recv_shapes, tuple)
    CudaTimer().start("send_recv")
    reqs = list()
    recved_tensors = list()
    for tensor in send_tensors:
        if tensor is None:
            continue
        send_op = torch.distributed.P2POp(
            torch.distributed.isend, tensor, rank
        )
        reqs.append(send_op)
    for shape in recv_shapes:
        if shape is None:
            recved_tensors.append(None)
            continue
        recv_tensor = torch.empty(
            shape, requires_grad=True, device=torch.cuda.current_device(),
            dtype=dtype
        )
        recv_op = torch.distributed.P2POp(
            torch.distributed.irecv, recv_tensor, rank
        )
        recved_tensors.append(recv_tensor)
        reqs.append(recv_op)
    reqs = torch.distributed.batch_isend_irecv(reqs)
    for req in reqs:
        req.wait()
    torch.cuda.synchronize()
    CudaTimer().stop("send_recv")
    return recved_tensors

#================= Between Stage functions ==================#

def split_batch(inputs, num_microbatches):
    """
    Split a mini-batch to micro-batches
    """
    assert isinstance(inputs, list) or isinstance(inputs, tuple)
    input_chunks = list()
    for feature_map in inputs:
        if torch.is_tensor(feature_map):
            feature_map = torch.chunk(feature_map, chunks=num_microbatches, dim=0)
        else:
            feature_map = [feature_map] * num_microbatches
        input_chunks.append(feature_map)
    micro_batches = list()
    for micro_data in zip(*tuple(input_chunks)):
        micro_batches.append(micro_data)
    return micro_batches


#================= Scheduling ==================#

def scheduling_1f1b(model, inputs, bs, micro_bs, dtype=torch.float):
    myrank = torch.distributed.get_rank()

    num_microbatches = int(bs / micro_bs)
    num_warmup_microbatches = \
        (torch.distributed.get_world_size() - 
         torch.distributed.get_rank() - 1)
    num_warmup_microbatches = min(num_warmup_microbatches, num_microbatches)
    num_warmup_remaining = num_microbatches - num_warmup_microbatches
    
    input_tensors = list()
    output_tensors = list()

    inputs = split_batch(inputs, num_microbatches)

    # warmup forward pass
    for i in range(num_warmup_microbatches):
        # recv forward
        # print('[warmup] rank {}: step-{}: recving forward...'.format(myrank, i))
        feature_map = recv(
            (torch.Size([micro_bs] + model.in_size),), myrank-1, dtype
        )[0]
        image = inputs[i][0]
        # forward
        output_tensor = forward_step(model, image, feature_map)
        # send forward
        # print('[warmup] rank {}: step-{}: sending forward...'.format(myrank, i))
        send((output_tensor,), myrank+1)

        input_tensors.append(feature_map)
        output_tensors.append(output_tensor)

    # before running 1F1B, need to recieve first forward tensor
    if num_warmup_remaining > 0:
        # recv forward
        # print('[1f1b] rank {}: step-{}: recving forward...'.format(myrank, 0))
        feature_map = recv(
            (torch.Size([micro_bs] + model.in_size),), myrank-1, dtype
        )[0]
        image = inputs[num_warmup_microbatches][0]

    # run 1F1B
    for i in range(num_warmup_remaining):
        # forward
        output_tensor = forward_step(model, image, feature_map)
        # send forward + recv backward grads
        # print('[1f1b] rank {}: step-{}: sending forward + recving backward...'.format(myrank, i))
        output_tensor_grad = send_and_recv(
            (output_tensor,),
            (torch.Size([micro_bs] + model.out_size),),
            myrank+1, dtype
        )[0]
        input_tensors.append(feature_map)
        output_tensors.append(output_tensor)
        # backward
        feature_map, output_tensor = input_tensors.pop(0), output_tensors.pop(0)
        input_tensor_grad = backward_step(feature_map, output_tensor, output_tensor_grad)
        if i != (num_warmup_remaining-1):
            # send backward grads + recv forward results
            # print('[1f1b] rank {}: step-{}: sending backward + recving forward...'.format(myrank, i))
            feature_map = send_and_recv(
                (input_tensor_grad,),
                (torch.Size([micro_bs] + model.in_size),),
                myrank-1, dtype
            )[0]
            image = inputs[num_warmup_microbatches+i+1][0]
        else:   # last iteration - no more inputs
            feature_map = None
            # send backward grads
            # print('[1f1b] rank {}: step-{}: sending backward...'.format(myrank, i))
            send((input_tensor_grad,), myrank-1)
    
    # cooldown gradient trans back
    for i in range(num_warmup_microbatches):
        feature_map = input_tensors.pop(0)
        output_tensor = output_tensors.pop(0)
        # recv backward gradients
        output_tensor_grad = recv(
            (torch.Size([micro_bs] + model.out_size),), myrank+1, dtype
        )[0]
        # backward
        input_tensor_grad = backward_step(feature_map, output_tensor, output_tensor_grad)
        # send backward gradients
        # print('[cooldown] rank {}: step-{}: sending backward...'.format(myrank, i))
        send((input_tensor_grad,), myrank-1)

#================= Scheduling ==================#