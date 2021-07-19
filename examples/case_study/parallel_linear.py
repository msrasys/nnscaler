"""Example Usage

python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=62000 \
    --use_env \
    examples/case_study/parallel_linear.py
"""

import torch
import os
from torch.nn.parameter import Parameter
torch.manual_seed(121)

hooks = list()

# tensor parallel - split weight in column
def linear_tensor_parallel(input, weight, bias):
    ### Policy need to know ###
    devices = [0, 1, 2, 3]               # how many device to perform?

    ### Necessary information to know ###
    rank = torch.distributed.get_rank()  # which role I participate?

    ### Additional ops need to use ###
    class InputAdapter(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input_):
            return input_
        @staticmethod
        def backward(ctx, grad_output):
            return torch.distributed.all_reduce(grad_output)
    
    class OutputAdapter(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input_):
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
            tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
            tensor_list[rank] = input_
            torch.distributed.all_gather(tensor_list, input_)
            output = torch.cat(tensor_list, dim=-1)
            return output
        @staticmethod
        def backward(ctx, grad_output):
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
            tensor_list = torch.split(
                grad_output, grad_output.size()[-1]//world_size, dim=-1
            )
            return tensor_list[rank].contiguous()

    ### Input Slice ###
    weight = torch.chunk(weight, chunks=len(devices), dim=0)[rank].contiguous()
    bias = torch.chunk(bias, chunks=len(devices), dim=0)[rank].contiguous()

    ### Input Adapter ###
    input = InputAdapter.apply(input)
    
    ### Forward ###
    output = torch._C._nn.linear(input, weight, bias)

    ### Ouput Adapter ###
    # insert a forward + backward op at last (allgather - split)
    output = OutputAdapter.apply(output)
    return output


# data parallel
def linear_data_parallel(input, weight, bias):
    ### Additional ops need to use ###
    # -> torch.distributed.all_reduce at backward
    
    ### Input Adapter ###
    hw = weight.register_hook(lambda grad: torch.distributed.all_reduce(grad))
    hb = bias.register_hook(lambda grad: torch.distributed.all_reduce(grad))
    global hooks
    hooks += [hw, hb]

    ### Forward ###
    output = torch._C._nn.linear(input, weight, bias)

    ### Output Adapter ### -> no need
    return output


# tensor + data parallel
def linear_hybrid_tensor_data_parallel(input, weight, bias):
    ### Policy need to know ###
    tp_size = 2                       # how many slices? which device?
    dp_size = 2

    ### Necessary information to execute ###
    rank = torch.distributed.get_rank()  # which role I participate?

    # data parallel group
    dp_group = None
    group = torch.distributed.new_group([0,2])
    if rank in [0, 2]:
        dp_group = group
    group = torch.distributed.new_group([1,3])
    if rank in [1, 3]:
        dp_group = group

    # tensor parallel group
    tp_group = None
    group = torch.distributed.new_group([0,1])
    if rank in [0, 1]:
        tp_group = group
    group = torch.distributed.new_group([2,3])
    if rank in [2, 3]:
        tp_group = group
    tp_rank = torch.distributed.get_rank(group=tp_group)
    tp_world_size = torch.distributed.get_world_size(group=tp_group)
    print_each_rank(
        'rank global:tp:dp=[{}:{}:{}] | size global:tp:dp=[{}:{}:{}]'.format(
            torch.distributed.get_rank(),
            torch.distributed.get_rank(tp_group),
            torch.distributed.get_rank(dp_group),
            torch.distributed.get_world_size(),
            torch.distributed.get_world_size(tp_group),
            torch.distributed.get_world_size(dp_group)
        ))

    ### Additional Ops ###
    class InputAdapter(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input_, group):
            ctx.constants = group
            return input_
        @staticmethod
        def backward(ctx, grad_output):
            group = ctx.constants
            return torch.distributed.all_reduce(grad_output, group=group), None
    
    class OutputAdapter(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input_, group, dim=-1):
            world_size = torch.distributed.get_world_size(group=group)
            rank = torch.distributed.get_rank(group=group)
            tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
            tensor_list[rank] = input_
            torch.distributed.all_gather(tensor_list, input_, group=group)
            output = torch.cat(tensor_list, dim=dim)
            ctx.constants = (group, dim)
            return output
        @staticmethod
        def backward(ctx, grad_output):
            group, dim = ctx.constants
            world_size = torch.distributed.get_world_size(group=group)
            rank = torch.distributed.get_rank(group=group)
            tensor_list = torch.split(
                grad_output, grad_output.size()[-1]//world_size, dim=dim
            )
            return tensor_list[rank].contiguous(), None, None

    ### Input Adapter - Slice ###
    weight = torch.chunk(weight, chunks=tp_world_size, dim=0)[tp_rank].contiguous()
    bias = torch.chunk(bias, chunks=tp_world_size, dim=0)[tp_rank].contiguous()
    # replicate is implicitly done due to SPMD
    
    ### Input Adapter - Data Parallel ###
    weight.register_hook(lambda grad: torch.distributed.all_reduce(grad, group=dp_group))
    bias.register_hook(lambda grad: torch.distributed.all_reduce(grad, group=dp_group))

    torch.distributed.barrier()
    ### Input Adapter - Tensor Parallel ###
    input = InputAdapter.apply(input, tp_group)

    ### Forward ###
    output = torch._C._nn.linear(input, weight, bias)

    ### Output Adapter - Tensor Parallel ###
    output = OutputAdapter.apply(output, tp_group, -1)

    ### Ouput Adapter - Data Parallel ###
    ## No need

    return output



######### Utility #############
def print_each_rank(msg, selected_rank=None):
    myrank = torch.distributed.get_rank()
    for rank in range(torch.distributed.get_world_size()):
        if selected_rank is None or myrank in selected_rank:
            if myrank == rank:
                print('rank [{}]: {}\n'.format(rank, msg))
        torch.distributed.barrier()


if __name__ == '__main__':

    local_rank = int(os.environ.get('LOCAL_RANK'))
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://',
    )

    # tensor definition
    batch_size = 32
    out_features = 1024
    in_features = 1024
    weight = torch.rand((out_features, in_features)).cuda().requires_grad_()
    # print_each_rank('weight: {}'.format(weight))
    bias = torch.rand(out_features).cuda().requires_grad_()
    # print_each_rank('bias: {}'.format(bias))
    input = torch.rand((batch_size, in_features)).cuda()
    # print_each_rank('input: {}'.format(input))
    
    # model parallel
    print_each_rank('======== Model Parallel =========', [0])
    output = linear_tensor_parallel(input, weight, bias)
    loss = torch.mean(output)
    print_each_rank(loss)
    loss.backward()
    # note weight is created as transposed
    print_each_rank('weight grad: {}'.format(weight.grad.t()))
    print_each_rank('======== Model Parallel =========', [0])

    # data parallel
    weight.grad = None
    bias.grad = None
    print_each_rank('======== Data Parallel =========', [0])
    output = linear_data_parallel(input, weight, bias)
    loss = torch.mean(output)
    loss.backward()
    print_each_rank('weight grad: {}'.format(weight.grad.t()))
    print_each_rank('======== Data Parallel =========', [0])

    # hybrid tensor-data parallel
    weight.grad = None
    bias.grad = None
    for hook in hooks:
        hook.remove()
    print_each_rank('======== Data + Tensor Parallel =========', [0])
    output = linear_hybrid_tensor_data_parallel(input, weight, bias)
    loss = torch.mean(output)
    # print_each_rank(loss)
    loss.backward()
    print_each_rank('weight grad: {}'.format(weight.grad.t()))
    print_each_rank('======== Data + Tensor Parallel =========', [0])
