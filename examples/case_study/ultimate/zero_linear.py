"""
Zero Redundancy Implementation

Partition Weights / Gradients / Optimizer States across GPUs

python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=62000 \
    --use_env \
    examples/case_study/ultimate/zero_linear.py
"""
import torch
import os
torch.manual_seed(121)

tensor_map = dict()

def linear_zero(input, weight, bias):
    ### weight / bias is partitioned ###
    class ZeroLinear(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, weight, bias):

            weight_id = id(weight)
            bias_id = id(bias)
            ctx.save_for_backward(input, torch.tensor(weight_id), torch.tensor(bias_id))
            tensor_map[weight_id] = weight
            tensor_map[bias_id] = bias

            # ======= all-gather parameters ========= #
            device_num = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
            # all-gather weight
            weight_list = [torch.empty_like(weight) for _ in range(device_num)]
            weight_list[rank] = weight
            torch.distributed.all_gather(weight_list, weight)
            weight_full = torch.cat(weight_list, dim=0).contiguous()
            # all-gather bias
            bias_list = [torch.empty_like(bias) for _ in range(device_num)]
            bias_list[rank] = bias
            torch.distributed.all_gather(bias_list, bias)
            bias_full = torch.cat(bias_list, dim=0).contiguous()
            # ======= all-gather parameters ========= #

            # compute: -> use full weight / bias
            output = torch._C._nn.linear(input, weight_full, bias_full)

            return output

        @staticmethod
        def backward(ctx, grad_output):
            input, weight_id, bias_id = ctx.saved_tensors
            weight = tensor_map[weight_id.item()]
            bias = tensor_map[bias_id.item()]

            grad_input = grad_weight = grad_bas = None
            if ctx.needs_input_grad[0]:
                # ========== all-gather weight =========== #
                weight_list = [torch.empty_like(weight) for _ in range(device_num)]
                weight_list[rank] = weight
                torch.distributed.all_gather(weight_list, weight)
                weight_full = torch.cat(weight_list, dim=0).contiguous()
                # ========== all-gather weight =========== #

                grad_input = grad_output.matmul(weight_full)

            if ctx.needs_input_grad[1]:
                dim = grad_output.dim()
                if dim > 2:
                    grad_weight_full = grad\
                        .view(-1, grad_output.shape[-1])\
                        .t()\
                        .matmul(input.view(-1, input.shape[-1]))
                else:
                    grad_weight_full = grad_output.t().matmul(input)
            if ctx.needs_input_grad[2]:
                grad_bias_full = grad_output.sum(0)

            ## ========== reduce-scatter for data parallelism ========= ##
            device_num = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
            grad_weight_list = list(torch.chunk(grad_weight_full, chunks=device_num, dim=0))
            grad_weight = torch.empty_like(grad_weight_list[rank])
            torch.distributed.reduce_scatter(grad_weight, grad_weight_list)
            grad_bias_list = list(torch.chunk(grad_bias_full, chunks=device_num, dim=0))
            grad_bias = torch.empty_like(grad_bias_list[rank])
            torch.distributed.reduce_scatter(grad_bias, grad_bias_list)
            ## ========== reduce-scatter for data parallelism ========= ##

            return grad_input, grad_weight, grad_bias
    
    output = ZeroLinear.apply(input, weight, bias)
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
    devices = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()

    # tensor definition
    batch_size = 32
    out_features = 10240
    in_features = 10240  ## 100 MB weight

    # weight
    weight = torch.chunk(
        torch.rand((out_features, in_features)),
        chunks=devices,
        dim=0
    )[rank].contiguous().cuda().requires_grad_()

    # bias
    bias = torch.chunk(
        torch.rand((out_features,)),
        chunks=devices,
        dim=0
    )[rank].contiguous().cuda().requires_grad_()

    # data
    input = torch.rand((batch_size, in_features)).cuda()
    
    # op compute
    print_each_rank('======== Zero-Redundancy =======', [0])

    output = linear_zero(input, weight, bias)
    loss = torch.mean(output) * 100
    print_each_rank('loss: {}'.format(loss))
    loss.backward()

    with torch.no_grad():
        weight.data += weight.grad
        bias.data += bias.grad
    
    # finish_op_memory = (torch.cuda.memory_allocated() - init_memory) / 1024 / 1024
    # max_allocated = (torch.cuda.max_memory_allocated() - init_memory) / 1024 / 1024
    
    # allocate tensor on gpu to see if swap workds
    # after_alloc_memory = (torch.cuda.memory_allocated() - init_memory) / 1024 / 1024

    # print('Memory Consumption (MB):\n\t input-require: {:.2f}\n\t after swap weight: {:.2f}\n\t after op run {:.2f}\n\t max allocated: {:.2f}\n\t after allocate {:.2f}'.format(
    #     input_memory, weight_swap_memory, finish_op_memory, max_allocated, after_alloc_memory))

    # correctness verify
    output = linear_zero(input, weight, bias)
    loss = torch.mean(output) * 100
    print_each_rank('loss: {}'.format(loss))
    print_each_rank('======== Zero-Redundancy =======', [0])
