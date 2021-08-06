"""Example Usage

python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=62000 \
    --use_env \
    examples/case_study/ultimate/model_partition_linear.py
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


def is_last_stage():
    return torch.distributed.get_rank() == torch.distributed.get_world_size() - 1

#================= WhatToDO functions ==================#

def forward_step(model, input_tensor):
    output_tensor = model(input_tensor)
    # last stage: calcuate loss
    if is_last_stage():
        output_tensor = torch.sum(output_tensor)
        print('loss: {}'.format(output_tensor))
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

#================= Between Stage functions ==================#


#================= Scheduling ==================#

def scheduling_naive(model, inputs, bs, feats):

    myrank = torch.distributed.get_rank()

    # ================  forward pass ================ #
    # recv input data
    input_tensor = recv(torch.Size([bs, feats]), myrank-1, inputs)
    # forward
    output_tensor = forward_step(model, input_tensor)
    # send forward
    send(output_tensor, myrank+1)

    # ================ backward pass ================ #
    # recv backward
    output_tensor_grad = recv(torch.Size([bs, feats]), myrank+1, None)
    # backward
    input_tensor_grad = backward_step(
        input_tensor, output_tensor, output_tensor_grad)
    # send backward
    send(input_tensor_grad, myrank-1)

    # ================ weight update ================ #
    # xxx

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
    features = 10240

    model = Linears(features, op_num=4).cuda()

    if myrank == 0:
        inputs = torch.randn((bs, features)).cuda()
    else:
        inputs = None

    scheduling_naive(model, inputs, bs, features)
