import torch
from torch import nn


class Linears(nn.Module):
    """
    Note in model creation, it will only construct model chunks
    that belong to this rank
    """

    def __init__(self, features, layers=4):
        super().__init__()
        self.ops = nn.ModuleList([])

        myrank = torch.distributed.get_rank()
        ngpus = torch.distributed.get_world_size()
        op_per_rank = int(layers / ngpus)

        for _ in range(op_per_rank):
            self.ops.append(nn.Linear(features, features))
    
    def forward(self, x):
        out = x
        for op in self.ops:
            out = op(out)
        return out


def is_last_stage():
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
    if input_tensor is not None:
        input_tensor.retain_grad()
    torch.autograd.backward(output_tensor, grad_tensors=output_tensor_grad)
    input_tensor_grad = None
    if input_tensor is not None:
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


def recv(shape, from_rank, inputs_first_stage):
    if from_rank < 0 or from_rank >= torch.distributed.get_world_size():
        return None
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

def send_and_recv(send_tensor, to_rank, recv_shape, from_rank, inputs_first_stage):
    if to_rank > torch.distributed.get_world_size() or from_rank < 0:
        return None
    recv_tensor = torch.empty(
        recv_shape, requires_grad=True, device=torch.cuda.current_device()
    )
    send_op = torch.distributed.P2POp(
        torch.distributed.isend, send_tensor, to_rank
    )
    recv_op = torch.distributed.P2POp(
        torch.distributed.irecv, recv_tensor, from_rank
    )
    reqs = torch.distributed.batch_isend_irecv([send_op, recv_op])
    for req in reqs:
        req.wait()
    torch.cuda.synchronize()
    return recv_tensor



#================= Between Stage functions ==================#



def scheduling_1f1b(model, inputs, bs, feats, micro_bs):
    myrank = torch.distributed.get_rank()

    num_microbatches = bs / micro_bs
    num_warmup_microbatches = \
        (torch.distributed.get_world_size() - 
         torch.distributed.get_rank() - 1)
    num_warmup_remaining = num_microbatches - num_warmup_microbatches
    
    input_tensors = list()
    output_tensors = list()

    if inputs is not None:
        inputs = torch.chunk(input_tensor, chunks=num_microbatches, dim=0)

    # warmup forward pass
    for i in range(num_warmup_microbatches):
        # recv forward
        input_tensor = recv(torch.Size([bs, feats]), myrank-1, inputs)
        # forward
        output_tensor = forward_step(model, input_tensor)
        # send forward
        send(output_tensor, myrank+1)

        input_tensors.append(input_tensor)
        output_tensors.append(output_tensor)

    # before running 1F1B, need to recieve first forward tensor
    if num_warmup_remaining > 0:
        # recv forward
        input_tensor = recv(torch.Size([bs, feats]), myrank-1, inputs)
        if input_tensor is None:
            input_tensor = inputs[i+num_warmup_microbatches]

    # run 1F1B
    for i in range(num_warmup_remaining):
        # forward
        output_tensor = forward_step(model, input_tensor)
        # send forward + recv backward grads
        output_tensor_grad = send_and_recv_backward(
            output_tensor, myrank+1, torch.Size([bs, feats]), myrank+1)
        input_tensors.append(input_tensor)
        output_tensors.append(output_tensor)
        # backward
        input_tensor, output_tensor = input_tensors.pop(0), output_tensors.pop(0)
        input_tensor_grad = backward_step(input_tensor, output_tensor, output_tensor_grad)
        if i != (num_warmup_remaining-1):
            # send backward grads + recv forward results
            input_tensor = send_and_recv(
                input_tensor_grad, myrank-1, torch.Size([bs, feats]), myrank-1)
        else:   # last iteration - no more inputs
            input_tensor = None
            # send backward grads
            send(input_tensor_grad, myrank - 1)
    
    # cooldown
    for i in range(num_warmup_microbatches):
        input_tensor = input_tensors.pop(0)
        output_tensor = output_tensors.pop(0)
        # recv backward gradients
        output_tensor_grad = recv(torch.Size([bs, feats]), myrank+1)
        # backward
        input_tensor_grad = backward_step(input_tensor, output_tensor, output_tensor_grad)
        # send backward gradients
        send(input_tensor_grad, myrank-1)


if __name__ == '__main__':

    # initialize distributed env
    local_rank = int(os.environ.get('LOCAL_RANK'))
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://',
    )

    batch_size = 32
    features = 1024

    torch.randn((batch_size, features))


