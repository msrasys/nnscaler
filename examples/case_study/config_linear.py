"""Example Usage

python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=6000 \
    --use_env \
    examples/case_study/config_linear.py
"""

import torch
from torch.nn.parameter import Parameter
torch.manual_seed(121)

# tensor parallel - split weight in column
def linear_tensor_parallel(input, weight, bias):
    ### Policy need to know ###
    devices = [0, 1]                       # how many device to perform?

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

    ### Input Adapter ###
    weight = torch.chunk(weight, chunks=len(devices), dim=0)[rank].contiguous()
    bias = torch.chunk(bias, chunks=len(devices), dim=0)[rank].contiguous()
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
    
    ### Input Adapter ###
    weight.register_hook(lambda grad: torch.distributed.allreduce(grad))
    bias.register_hook(lambda grad: torch.distributed.allreduce(grad))

    ### Forward ###
    output = torch._C._nn.linear(input, weight, bias)

    ### Output Adapter ### -> no need
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

    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://',
    )
    torch.cuda.set_device(torch.distributed.get_rank())

    # tensor definition
    batch_size = 32
    out_features = 1024
    in_features = 1024
    weight = Parameter(torch.rand((out_features, in_features))).cuda()
    # print_each_rank('weight: {}'.format(weight))
    bias = Parameter(torch.rand(out_features)).cuda()
    # print_each_rank('bias: {}'.format(bias))
    input = torch.rand((batch_size, in_features)).cuda()
    # print_each_rank('input: {}'.format(input))
    
    # model parallel
    print_each_rank('======== Model Parallel =========', [0])
    output = linear_tensor_parallel(input, weight, bias)
    loss = torch.mean(output)
    print_each_rank(loss)
    loss.backward()
    print_each_rank('======== Model Parallel =========', [0])

    # data parallel
    print_each_rank('======== Data Parallel =========', [0])
    print_each_rank('======== Data Parallel =========', [0])
