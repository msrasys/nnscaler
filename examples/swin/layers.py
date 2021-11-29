import torch
from torch import autograd
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from cube.profiler.timer import CudaTimer


def _reduce(input_, group):
    """All-reduce the input tensor across model parallel group."""

    # Bypass the function if we are using only 1 GPU.
    CudaTimer().start(field_name='tp_allreduce')
    world_size = torch.distributed.get_world_size(group)
    if world_size == 1:
        CudaTimer().stop(field_name='tp_allreduce')
        return input_
    torch.distributed.all_reduce(input_, group=group)
    CudaTimer().stop(field_name='tp_allreduce')
    return input_


def _split(input_, group, dim=-1):
    """Split the tensor along its last dimension and keep the
    corresponding slice."""

    world_size = torch.distributed.get_world_size(group=group)
    rank = torch.distributed.get_rank(group=group)
    # Bypass the function if we are using only 1 GPU.
    if world_size==1:
        return input_
    dim_size = input_.size()[dim] // world_size
    tensor_list = torch.split(input_, dim_size, dim=dim)
    output = tensor_list[rank].contiguous()
    return output


def _gather(input_, group, dim=-1):
    """Gather tensors and concatinate along the last dimension."""
    CudaTimer().start(field_name='tp_allgather')

    world_size = torch.distributed.get_world_size(group=group)
    rank = torch.distributed.get_rank(group=group)
    # Bypass the function if we are using only 1 GPU.
    if world_size==1:
        CudaTimer().stop(field_name='tp_allgather')
        return input_
    # Size and dimension.
    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_
    torch.distributed.all_gather(tensor_list, input_, group=group)
    # Note: torch.cat already creates a contiguous tensor.
    output = torch.cat(tensor_list, dim=dim).contiguous()

    CudaTimer().stop(field_name='tp_allgather')
    return output

def _scatter(input_, group, dim=0):
    """Reduce-Scatter tensor"""
    CudaTimer().start(field_name='tp_reduce_scatter')
    world_size = torch.distributed.get_world_size(group=group)
    if world_size == 1:
        CudaTimer().stop(field_name='tp_reduce_scatter')
        return input_
    rank = torch.distributed.get_rank(group=group)
    tensor_list = list(torch.chunk(input_, world_size, dim))
    # for idx, tensor in enumerate(tensor_list):
    #     tensor_list[idx] = tensor.contiguous()
    torch.distributed.reduce_scatter(tensor_list[rank], tensor_list, group=group)
    CudaTimer().stop(field_name='tp_reduce_scatter')
    return tensor_list[rank]


class ColumnInputAdapter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, group):
        ctx.group = group
        return input_
    @staticmethod
    def backward(ctx, grad_output):
        group = ctx.group
        return _reduce(grad_output, group), None


class ColumnOutputAdapter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, group):
        ctx.group = group
        return _gather(input_, group)
    @staticmethod
    def backward(ctx, grad_output):
        group = ctx.group
        return _split(grad_output, group), None


class RowInputAdapter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, group):
        ctx.group = group
        return _split(input_, group)

    @staticmethod
    def backward(ctx, grad_outputs):
        group = ctx.group
        return _gather(grad_outputs, group), None


class RowOutputAdapter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, group):
        ctx.group = group
        return _reduce(input_, group)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class DPtoTPAdapter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, group):
        """
        split
        """
        ctx.group = group
        return _gather(input_, group, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        reduce-scatter
        """
        group = ctx.group
        return _split(grad_output, group, dim=0), None


class TPtoDPAdapter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, group):
        """
        Reduce-scatter
        """
        ctx.group = group
        return _split(input_, group, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        all-gather
        """
        group = ctx.group
        return _gather(grad_output, group, dim=0), None




class ColumnParallelLinear(torch.nn.Module):

    def __init__(self, input_size, output_size, bias=True, in_adapter=True, out_adapter=True, tp_group=-1):
        super().__init__()
        assert tp_group != -1
        self.input_size = input_size
        self.output_size = output_size
        self.in_adapter = in_adapter
        self.out_adapter = out_adapter

        self.group = tp_group
        self.world_size = torch.distributed.get_world_size(group=self.group)

        # print_each_rank(f'> parallizing linear using column partition: '
        #                 f'{output_size} partitioned by {world_size} devices')

        # not if output size is smaller than world size,
        # no parallel enbaled. Each device compute the same
        if self.world_size > output_size:
            raise RuntimeError

        self.weight = Parameter(torch.empty(
            int(self.output_size // self.world_size),
            self.input_size,
        ))
        if bias:
            self.bias = Parameter(torch.empty(
                int(self.output_size // self.world_size),
            ))
        else:
            self.bias = None

    def forward(self, input_):

        if self.in_adapter and self.world_size > 1:
            input_ = ColumnInputAdapter.apply(input_, self.group)

        output = F.linear(input_, self.weight, self.bias)

        if self.out_adapter and self.world_size > 1:
            output = ColumnOutputAdapter.apply(output, self.group)

        return output


class RowParallelLinear(torch.nn.Module):

    def __init__(self, input_size, output_size, bias=True, in_adapter=True, out_adapter=True, tp_group=-1):
        super().__init__()
        assert tp_group != -1
        self.input_size = input_size
        self.output_size = output_size
        self.in_adapter = in_adapter
        self.out_adapter = out_adapter

        self.group = tp_group
        self.world_size = torch.distributed.get_world_size(group=self.group)

        # print_each_rank(f'> parallizing linear using row partition: '
        #                 f'{output_size} partitioned by {world_size} devices')

        # not if output size is smaller than world size,
        # no parallel enbaled. Each device compute the same
        if self.world_size > input_size:
            raise RuntimeError

        self.weight = Parameter(torch.empty(
            self.output_size,
            int(self.input_size // self.world_size),
        ))
        if bias:
            self.bias = Parameter(torch.empty(self.output_size))
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)

    def forward(self, input_):
        bias = self.bias
        if self.in_adapter and self.world_size > 1:
            input_ = RowInputAdapter.apply(input_, self.group)

        output = F.linear(input_, self.weight, bias)

        if self.out_adapter and self.world_size > 1:
            output = RowOutputAdapter.apply(output, self.group)

        return output


class ShardEmbedding(torch.nn.Module):

    def __init__(self, num_embeddings, embedding_dim, tp_group):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.group = tp_group
        self.shard_num = torch.distributed.get_world_size(group=self.group)
        self.myshard = torch.distributed.get_rank(group=self.group)

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
        output = RowOutputAdapter.apply(output_parallel, self.group)
        return output


class DPtoTP(torch.nn.Module):

    def __init__(self, dp_group):
        super().__init__()
        self.group = dp_group

    def forward(self, input_):
        return DPtoTPAdapter.apply(input_, self.group)


class TPtoDP(torch.nn.Module):

    def __init__(self, tp_group):
        super().__init__()
        self.group = tp_group

    def forward(self, input_):
        return TPtoDPAdapter.apply(input_, self.group)

