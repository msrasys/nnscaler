"""Example Usage

python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=62000 \
    --use_env \
    examples/case_study/ultimate/pipeline_linear.py
"""

import torch
from torch import nn
import os


class Linears(nn.Module):
    """
    Note in model creation, it will only construct model chunks
    that belong to this rank
    """

    def __init__(self, features, op_num=4):
        super().__init__()
        self.ops = nn.ModuleList([])

        myrank = torch.distributed.get_rank()
        ngpus = torch.distributed.get_world_size()
        op_num_per_rank = int(op_num / ngpus)

        for _ in range(op_num_per_rank):
            self.ops.append(nn.Linear(features, features))
    
    def forward(self, x):
        out = x
        for op in self.ops:
            out = op(out)
        return out


def is_first_stage():
    return torch.distributed.get_rank() == 0


def is_last_stage():
    return torch.distributed.get_rank() == torch.distributed.get_world_size() - 1


#================= WhatToDO functions ==================#

def forward_step(model, input_tensor):
    output_tensor = model(input_tensor)
    # last stage: calcuate loss
    if is_last_stage():
        output_tensor = torch.sum(output_tensor)
    return output_tensor


def backward_step(input_tensor, output_tensor, output_tensor_grad):
    """
    Calculate input tensor gradient
    """
    if input_tensor is not None and input_tensor.requires_grad:
        input_tensor.retain_grad()
    torch.autograd.backward(output_tensor, grad_tensors=output_tensor_grad)
    input_tensor_grad = None
    if input_tensor is not None and input_tensor.requires_grad:
        input_tensor_grad = input_tensor.grad
    return input_tensor_grad

#================= WhatToDO functions ==================#

#================= Between Stage functions ==================#

def send(tensor, to_rank):
    """
    send tensor to the target rank
    """
    if to_rank < 0 or to_rank >= torch.distributed.get_world_size():
        return None
    send_op = torch.distributed.P2POp(
        torch.distributed.isend, tensor, to_rank
    )
    reqs = torch.distributed.batch_isend_irecv([send_op])
    for req in reqs:
        req.wait()
    torch.cuda.synchronize()


def recv(shape, from_rank, boundary_tensor):
    if from_rank < 0 or from_rank >= torch.distributed.get_world_size():
        return boundary_tensor
    tensor = torch.empty(
        shape, requires_grad=True, device=torch.cuda.current_device()
    )
    recv_op = torch.distributed.P2POp(
        torch.distributed.irecv, tensor, from_rank
    )
    reqs = torch.distributed.batch_isend_irecv([recv_op])
    for req in reqs:
        req.wait()
    torch.cuda.synchronize()
    return tensor


def send_and_recv(send_tensor, recv_shape, rank, boundary_tensor):
    if rank < 0 or rank >= torch.distributed.get_world_size():
        return boundary_tensor
    recv_tensor = torch.empty(
        recv_shape, requires_grad=True, device=torch.cuda.current_device()
    )
    send_op = torch.distributed.P2POp(
        torch.distributed.isend, send_tensor, rank
    )
    recv_op = torch.distributed.P2POp(
        torch.distributed.irecv, recv_tensor, rank
    )
    reqs = torch.distributed.batch_isend_irecv([send_op, recv_op])
    for req in reqs:
        req.wait()
    torch.cuda.synchronize()
    return recv_tensor

#================= Between Stage functions ==================#


#================= Scheduling ==================#

def scheduling_1f1b(model, inputs, bs, feats, micro_bs):
    myrank = torch.distributed.get_rank()

    num_microbatches = int(bs / micro_bs)
    num_warmup_microbatches = \
        (torch.distributed.get_world_size() - 
         torch.distributed.get_rank() - 1)
    num_warmup_remaining = num_microbatches - num_warmup_microbatches
    
    input_tensors = list()
    output_tensors = list()

    if inputs is not None:
        inputs = torch.chunk(inputs, chunks=num_microbatches, dim=0)
    else:
        inputs = [None] * num_microbatches

    # warmup forward pass
    for i in range(num_warmup_microbatches):
        # recv forward
        print('[warmup] rank {}: step-{}: recving forward...'.format(myrank, i))
        input_tensor = recv(torch.Size([micro_bs, feats]), myrank-1, inputs[i])
        # forward
        output_tensor = forward_step(model, input_tensor)
        # send forward
        print('[warmup] rank {}: step-{}: sending forward...'.format(myrank, i))
        send(output_tensor, myrank+1)

        input_tensors.append(input_tensor)
        output_tensors.append(output_tensor)

    # before running 1F1B, need to recieve first forward tensor
    if num_warmup_remaining > 0:
        # recv forward
        print('[1f1b] rank {}: step-{}: recving forward...'.format(myrank, 0))
        input_tensor = recv(torch.Size([micro_bs, feats]), myrank-1, inputs[num_warmup_microbatches])

    # run 1F1B
    for i in range(num_warmup_remaining):
        # forward
        output_tensor = forward_step(model, input_tensor)
        # send forward + recv backward grads
        print('[1f1b] rank {}: step-{}: sending forward + recving backward...'.format(myrank, i))
        output_tensor_grad = send_and_recv(
            output_tensor, torch.Size([micro_bs, feats]), myrank+1, None)
        input_tensors.append(input_tensor)
        output_tensors.append(output_tensor)
        # backward
        input_tensor, output_tensor = input_tensors.pop(0), output_tensors.pop(0)
        input_tensor_grad = backward_step(input_tensor, output_tensor, output_tensor_grad)
        if i != (num_warmup_remaining-1):
            # send backward grads + recv forward results
            print('[1f1b] rank {}: step-{}: sending backward + recving forward...'.format(myrank, i))
            input_tensor = send_and_recv(
                input_tensor_grad, torch.Size([micro_bs, feats]), myrank-1, inputs[num_warmup_microbatches+i+1])
        else:   # last iteration - no more inputs
            input_tensor = None
            # send backward grads
            print('[1f1b] rank {}: step-{}: sending backward...'.format(myrank, i))
            send(input_tensor_grad, myrank-1)
    
    # cooldown gradient trans back
    for i in range(num_warmup_microbatches):
        input_tensor = input_tensors.pop(0)
        output_tensor = output_tensors.pop(0)
        # recv backward gradients
        output_tensor_grad = recv(torch.Size([micro_bs, feats]), myrank+1, None)
        # backward
        input_tensor_grad = backward_step(input_tensor, output_tensor, output_tensor_grad)
        # send backward gradients
        print('[cooldown] rank {}: step-{}: sending backward...'.format(myrank, i))
        send(input_tensor_grad, myrank-1)

#================= Scheduling ==================#


if __name__ == '__main__':

    # initialize distributed env
    local_rank = int(os.environ.get('LOCAL_RANK'))
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://',
    )
    myrank = torch.distributed.get_rank()

    bs = 32
    micro_bs = 1
    features = 10240

    model = Linears(features, op_num=4).cuda()

    if myrank == 0:
        inputs = torch.randn((bs, features)).cuda()
    else:
        inputs = None

    for _ in range(50):
        scheduling_1f1b(model, inputs, bs, features, micro_bs)
