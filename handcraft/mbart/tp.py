from typing import Tuple
import torch


class Reduce(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, group):
        torch.distributed.all_reduce(input, group=group)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class IdentityFoward(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, group):
        ctx._group = group
        return input
    
    @staticmethod
    def backward(ctx, grad_output):
        torch.distributed.all_reduce(grad_output, group=ctx._group)
        return grad_output, None


def shard_linear_col(input, weight, bias, group):
    world_size = torch.distributed.get_world_size(group)
    if world_size == 1:
        return torch.nn.functional(input, weight, bias)
    input = IdentityFoward.apply(input, group)
    return torch.nn.functional(input, weight, bias)


def shard_linear_row(input, weight, bias, group):
    world_size = torch.distributed.get_world_size(group)
    if world_size == 1:
        return torch.nn.functional(input, weight, bias)
    out = torch.nn.functional(input, weight, bias)
    out = Reduce.apply(out, group)
    return out


class DummyModelEmbed(torch.nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int,
                 input_shape: Tuple[int, int], group = None):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.input_shape = input_shape

        self.tp_group = group
        self.tp_size = torch.distributed.get_world_size(group)

        shard_id = torch.distributed.get_rank(group)
        self.vocab_start_index = num_embeddings // self.tp_size * shard_id
        self.vocab_end_index = num_embeddings // self.tp_size * (shard_id + 1)
        self.embed_weight = torch.nn.Parameter(torch.ones((num_embeddings // self.tp_size, embedding_dim)))

    def input_shape(self):
        return self.input_shape

    def input_dtype(self):
        return torch.int64

    def output_shape(self):
        return self.input_shape + (self.embedding_dim,)

    def output_dtype(self):
        return torch.float32

    def forward(self, input: torch.Tensor):
        if self.tp_size > 1:
            mask = (input < self.vocab_start_index) | \
                        (input >= self.vocab_end_index)
            input = input.clone() - self.vocab_start_index
            input[mask] = 0
            input = torch.nn.functional.embedding(input, self.embed_weight)
            input[mask, :] = 0.0
            input = Reduce.apply(input, self.tp_group)
        else:
            input = torch.nn.functional.embedding(input, self.embed_weight)
        return input
