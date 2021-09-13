"""
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=62000 \
    --use_env \
    examples/poc/torchfx.py
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.fx import symbolic_trace

import os


local_rank = int(os.environ.get('LOCAL_RANK'))
torch.cuda.set_device(local_rank)
torch.distributed.init_process_group(
    backend='nccl',
    init_method='env://',
)

# ====================== Check for normal module ==========================
class FeedForward(nn.Module):
    def __init__(self, dim, dropout=0., mult=16, classes=1000):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim * mult)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim * mult, dim)
        self.classifier = nn.Linear(dim, classes)

    def forward(self, x):
        output = self.linear1(x)
        output = self.gelu(output)
        output = self.dropout(output)
        output = self.linear2(output)
        output = self.classifier(output)
        return output

model = FeedForward(dim=1024).cuda()
graph_module = symbolic_trace(model)
if local_rank == 0:
    print(graph_module)
    print(graph_module.code)
    print(graph_module.graph)


# ====================== Check for autograd function ==========================
class CustomOp(torch.autograd.Function):
    @staticmethod
    def symbolic(graph, input, weight):
        return torch.matmul(input, weight)
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        return torch.matmul(input, weight)
    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        return input+weight, input+weight

class CustomModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, weight):
        out = CustomOp.apply(input, weight)
        return out

custom_op = CustomModule().cuda()

input = torch.ones((1024, 1024)).cuda().requires_grad_()
weight = torch.ones((1024, 1024)).cuda().requires_grad_()

if local_rank == 0:
    custom_op_trace = symbolic_trace(custom_op)
    print(custom_op_trace)
    print(custom_op_trace.code)
    print(custom_op_trace.graph)
    # traced graph call
    out = custom_op_trace(input, weight)
    torch.sum(out).backward()
    print(out)
    print('weight grad: ', weight.grad)
    # original graph call

    out = custom_op(input, weight)
    input.grad = None
    weight.grad = None
    torch.sum(out).backward()
    print('weight grad expected: ', weight.grad)
    print(out)

torch.distributed.barrier()


# ====================== Check for function with communications ==========================
class InputAdapter(torch.autograd.Function):
    @staticmethod
    def symbolic(graph, input_):
        return input_
    @staticmethod
    def forward(ctx, input_):
        return input_
    @staticmethod
    def backward(ctx, grad_output):
        return torch.distributed.all_reduce(grad_output)


class OutputAdapter(torch.autograd.Function):
    @staticmethod
    def symbolic(graph, input_):
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
        tensor_list[rank] = input_
        torch.distributed.all_gather(tensor_list, input_)
        output = torch.cat(tensor_list, dim=-1)
        return output
    @staticmethod
    def forward(ctx, input_):
        # world_size = torch.distributed.get_world_size()
        # rank = torch.distributed.get_rank()
        # tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
        # tensor_list[rank] = input_
        # torch.distributed.all_gather(tensor_list, input_)
        # output = torch.cat(tensor_list, dim=-1)
        output = input_
        torch.distributed.all_reduce(output)
        return output
    @staticmethod
    def backward(ctx, grad_output):
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        tensor_list = torch.split(
            grad_output, grad_output.size()[-1]//world_size, dim=-1
        )
        return tensor_list[rank].contiguous()


class LinearComm(nn.Module):
    def __init__(self, input_feats, output_feats):
        super().__init__()
        self.linear = nn.Linear(input_feats, output_feats)
    def forward(self, x):
        x = InputAdapter.apply(x)
        x = self.linear(x)
        x = OutputAdapter.apply(x)
        return x

comm_linear = LinearComm(1024, 1024).cuda()
graph_comm = symbolic_trace(comm_linear)
if local_rank == 0:
    print(graph_comm.graph)
    print(graph_comm.code)

input = torch.ones((1024, 1024)).cuda().requires_grad_()
out = graph_comm(input)
out_ref = comm_linear(input)
if local_rank == 0:
    print('out: ', out)
    print('out expected: ', out_ref)
