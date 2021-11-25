import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from cube.profiler.timer import print_each_rank
from cube.runtime.resource import EnvResource


def _reduce(input_):
    """All-reduce the input tensor across model parallel group."""

    # Bypass the function if we are using only 1 GPU.
    world_size = torch.distributed.get_world_size(group=EnvResource().tp_group)
    if world_size == 1:
        return input_
    group = EnvResource().tp_group
    torch.distributed.all_reduce(input_, group=group)
    return input_


def _split(input_):
    """Split the tensor along its last dimension and keep the
    corresponding slice."""

    world_size = torch.distributed.get_world_size(group=EnvResource().tp_group)
    rank = torch.distributed.get_rank(group=EnvResource().tp_group)
    # Bypass the function if we are using only 1 GPU.
    if world_size==1:
        return input_
    last_dim = input_.dim() - 1
    last_dim_size = input_.size()[last_dim] // world_size
    tensor_list = torch.split(input_, last_dim_size, dim=last_dim)
    output = tensor_list[rank].contiguous()
    return output


def _gather(input_):
    """Gather tensors and concatinate along the last dimension."""

    world_size = torch.distributed.get_world_size(group=EnvResource().tp_group)
    rank = torch.distributed.get_rank(group=EnvResource().tp_group)
    # Bypass the function if we are using only 1 GPU.
    if world_size==1:
        return input_
    # Size and dimension.
    last_dim = input_.dim() - 1
    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_
    group = EnvResource().tp_group
    torch.distributed.all_gather(tensor_list, input_, group=group)
    # Note: torch.cat already creates a contiguous tensor.
    output = torch.cat(tensor_list, dim=last_dim).contiguous()
    return output


class ColumnInputAdapter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_):
        return input_
    @staticmethod
    def backward(ctx, grad_output):
        return _reduce(grad_output)


class ColumnOutputAdapter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_):
        return _gather(input_)
    @staticmethod
    def backward(ctx, grad_output):
        return _split(grad_output)


class RowInputAdapter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_):
        return _split(input_)

    @staticmethod
    def backward(ctx, grad_outputs):
        return _gather(grad_outputs)


class RowOutputAdapter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_):
        return _reduce(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class ColumnParallelLinear(torch.nn.Module):

    def __init__(self, input_size, output_size, bias=True, full_input=True, full_output=False):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.full_input = full_input
        self.full_output = full_output

        world_size = torch.distributed.get_world_size(group=EnvResource().tp_group)

        # print_each_rank(f'> parallizing linear using column partition: '
        #                 f'{output_size} partitioned by {world_size} devices')

        # not if output size is smaller than world size,
        # no parallel enbaled. Each device compute the same
        if world_size > output_size:
            world_size = 1

        self.weight = Parameter(torch.empty(
            int(self.output_size // world_size),
            self.input_size,
        ))
        if bias:
            self.bias = Parameter(torch.empty(
                int(self.output_size // world_size),
            ))
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)

    def forward(self, input_):
        bias = self.bias
        if not self.full_input:
            raise RuntimeError("Expected full tensor input")
        input_parallel = ColumnInputAdapter.apply(input_)
        output_parallel = F.linear(input_parallel, self.weight, bias)
        if self.full_output:
            output = ColumnOutputAdapter.apply(output_parallel)
        else:
            output = output_parallel
        return output


class RowParallelLinear(torch.nn.Module):

    def __init__(self, input_size, output_size, bias=True, full_input=True, full_output=False):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.full_input = full_input
        self.full_output = full_output

        world_size = torch.distributed.get_world_size(group=EnvResource().tp_group)

        # print_each_rank(f'> parallizing linear using row partition: '
        #                 f'{output_size} partitioned by {world_size} devices')

        # not if output size is smaller than world size,
        # no parallel enbaled. Each device compute the same
        if world_size > output_size:
            world_size = 1

        self.weight = Parameter(torch.empty(
            self.output_size,
            int(self.input_size // world_size),
        ))
        if bias:
            self.bias = Parameter(torch.empty(self.output_size))
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)

    def forward(self, input_):
        bias = self.bias
        if self.full_input:
            input_parallel = RowInputAdapter.apply(input_)
        else:
            input_parallel = input_
        output_parallel = F.linear(input_parallel, self.weight, bias)
        if self.full_output:
            output = RowOutputAdapter.apply(output_parallel)
        else:
            output = output_parallel
        return output


class ShardEmbedding(torch.nn.Module):

    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.shard_num = torch.distributed.get_world_size(group=EnvResource().tp_group)
        self.myshard = torch.distributed.get_rank(group=EnvResource().tp_group)

        shard_num_embeddings = self.num_embeddings // self.shard_num
        self.vocab_start_index = shard_num_embeddings * self.myshard
        self.vocab_end_index = self.vocab_start_index + shard_num_embeddings 

        self.weight = torch.nn.Parameter(
            torch.empty(shard_num_embeddings, self.embedding_dim)
        )

    def forward(self, input_):
        # Build the mask.
        input_mask = (input_ < self.vocab_start_index) | \
                     (input_ >= self.vocab_end_index)
        # Mask the input.
        masked_input = input_.clone() - self.vocab_start_index
        masked_input[input_mask] = 0
        output_parallel = F.embedding(
            masked_input, self.weight,
            None, None, 2., False, False
        )
        output = RowOutputAdapter.apply(output_parallel)
        return output
