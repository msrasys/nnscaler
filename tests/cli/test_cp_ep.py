"""Reuse EP ranks as context shards and extend CP across scale units.

For EP=2, CP=4, and runtime_ngpus=8, the rank topology is:

* CP group 0: ranks (0, 1, 2, 3), processing input A
* CP group 1: ranks (4, 5, 6, 7), processing input B
* EP groups inside CP group 0: (0, 1) and (2, 3)
* EP groups inside CP group 1: (4, 5) and (6, 7)

Within each CP group, the outer boundary shards a sequence across its two scale
units. The EP policy then partitions both that local sequence shard and the
expert dimension, so each of the four ranks receives a distinct quarter of its
group's sequence. Expert ownership repeats across all scale units. Corresponding
expert weight shards are reduced across CP groups as data-parallel replicas.
"""

from pathlib import Path
import re
from typing import Iterator, Optional, Tuple

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

import nnscaler
from nnscaler.cli.trainer import Trainer
from nnscaler.cli.trainer_args import TrainerArgs
from nnscaler.graph.function import DimopSplit, TransformRule
from nnscaler.parallel import ComputeConfig, _load_parallel_module_class, parallelize
from nnscaler.policies import OpPartition, OpPlan, fn, get_pas_ops
from nnscaler.runtime.adapter.collectives import all_gather, all_reduce, chunk
from nnscaler.runtime.adapter.nn import alltoall_alltoall
from nnscaler.runtime.adapter.reducer import ParamBucketConfig
from nnscaler.runtime.device import DeviceGroup
from tests.launch_torchrun import launch_torchrun
from tests.parallel_module.common import assert_close
from tests.parallel_module.test_gencode import _gencode_contains
from tests.utils import replace_all_device_with


CONFIG_PATH = Path(__file__).with_name('trainer_args_cp_ep.yaml')
_SCALE_UNITS_PER_CONTEXT_GROUP: Optional[int] = None
# Test-only probe used to report the actual input length seen by routed_expert.
_LAST_EXPERT_SEQUENCE_LENGTH: Optional[int] = None


def _context_group_ranks(
    context_parallel_size: int,
    rank: Optional[int] = None,
) -> Tuple[int, ...]:
    # CP=4, runtime_ngpus=8:
    # rank 0~3 -> [0, 1, 2, 3], rank 4~7 -> [4, 5, 6, 7].
    # Each of these groups processes one independent input sequence.
    rank = dist.get_rank() if rank is None else rank
    first_rank = rank // context_parallel_size * context_parallel_size
    return tuple(range(first_rank, first_rank + context_parallel_size))


def _local_cross_scale_ranks(
    expert_parallel_size: int,
    context_parallel_size: int,
    rank: Optional[int] = None,
) -> Tuple[int, ...]:
    # The outer CP split happens between scale units at the same EP position.
    # For CP=4 and EP=2:
    #   CP group [0, 1, 2, 3] -> [0, 2] and [1, 3]
    #   CP group [4, 5, 6, 7] -> [4, 6] and [5, 7]
    rank = dist.get_rank() if rank is None else rank
    first_rank = rank // context_parallel_size * context_parallel_size
    expert_lane = rank % expert_parallel_size
    return tuple(
        range(
            first_rank + expert_lane,
            first_rank + context_parallel_size,
            expert_parallel_size,
        )
    )


def _expert_shard_reducer_ranks(
    expert_parallel_size: int,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
) -> Tuple[int, ...]:
    # The same expert shard is repeated at the same EP lane of every scale unit.
    # For EP=2 and runtime_ngpus=8:
    #   even ranks own expert shard 0 -> [0, 2, 4, 6]
    #   odd ranks own expert shard 1  -> [1, 3, 5, 7]
    rank = dist.get_rank() if rank is None else rank
    world_size = dist.get_world_size() if world_size is None else world_size
    expert_lane = rank % expert_parallel_size
    return tuple(range(expert_lane, world_size, expert_parallel_size))


def _expert_parallel_ranks(expert_parallel_size: int) -> Tuple[int, ...]:
    # EP communication never crosses a scale-unit boundary:
    # [0, 1], [2, 3], [4, 5], or [6, 7].
    rank = dist.get_rank()
    first_rank = rank // expert_parallel_size * expert_parallel_size
    return tuple(range(first_rank, first_rank + expert_parallel_size))


def init_cp_ep_groups(trainer: Trainer) -> None:
    global _SCALE_UNITS_PER_CONTEXT_GROUP

    compute_config = trainer.train_args.compute_config
    expert_parallel_size = compute_config.plan_ngpus
    context_parallel_size = trainer.train_args.model.args['context_parallel_size']
    world_size = dist.get_world_size()
    if world_size != compute_config.runtime_ngpus:
        raise ValueError(f'world size {world_size} does not match runtime_ngpus {compute_config.runtime_ngpus}')
    if world_size % expert_parallel_size:
        raise ValueError('runtime_ngpus must be divisible by plan_ngpus')
    if context_parallel_size % expert_parallel_size:
        raise ValueError('context_parallel_size must be divisible by plan_ngpus')
    if world_size % context_parallel_size:
        raise ValueError('runtime_ngpus must be divisible by context_parallel_size')

    # CP=4 and EP=2 means one input spans two scale units. This value is also
    # the number of duplicate full-input gradients in the non-CP baseline.
    _SCALE_UNITS_PER_CONTEXT_GROUP = context_parallel_size // expert_parallel_size

    # Groups used by all-to-all dispatch/combine inside each EP unit.
    for first_rank in range(0, world_size, expert_parallel_size):
        DeviceGroup().get_group(tuple(range(first_rank, first_rank + expert_parallel_size)))

    # Full CP groups are used by context-dependent communication. Their
    # same-EP-lane subgroups are used by the outer sequence chunk/all-gather.
    for first_rank in range(0, world_size, context_parallel_size):
        DeviceGroup().get_group(tuple(range(first_rank, first_rank + context_parallel_size)))
        for expert_lane in range(expert_parallel_size):
            DeviceGroup().get_group(tuple(
                range(
                    first_rank + expert_lane,
                    first_rank + context_parallel_size,
                    expert_parallel_size,
                )
            ))


class ContextGroupSampler(torch.utils.data.Sampler):
    """Shard data across CP groups and replay it within each group's scale units."""

    def __init__(
        self,
        dataset,
        num_replicas: int,
        rank: int,
        context_parallel_size: int,
        expert_parallel_size: int,
        shuffle: bool = False,
        seed: int = 0,
    ) -> None:
        if context_parallel_size % expert_parallel_size:
            raise ValueError('context_parallel_size must be divisible by expert_parallel_size')
        scale_units_per_context_group = context_parallel_size // expert_parallel_size
        if num_replicas % scale_units_per_context_group:
            raise ValueError('number of scale units must be divisible by scale units per CP group')

        # all ranks in the same CP group see the same input sequence

        # Trainer passes scale-unit ranks here, not global GPU ranks. With four
        # scale units and two scale units per CP group:
        #   scale units 0/1 both read input A
        #   scale units 2/3 both read input B
        # With EP=2, global ranks [0, 3] see A and global ranks [4, 7] see B.
        self.sampler = torch.utils.data.DistributedSampler(
            dataset,
            num_replicas=num_replicas // scale_units_per_context_group,
            rank=rank // scale_units_per_context_group,
            shuffle=shuffle,
            seed=seed,
        )

    def __iter__(self) -> Iterator[int]:
        return iter(self.sampler)

    def __len__(self) -> int:
        return len(self.sampler)

    def set_epoch(self, epoch: int) -> None:
        self.sampler.set_epoch(epoch)


class _ScaleUnitContextChunk(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        expert_parallel_size: int,
        context_parallel_size: int,
    ) -> torch.Tensor:
        # Before this call, every rank in a CP group has the complete input
        # sample. For input A, ranks [0, 2] split it into two parts: rank 0 gets
        # the first half and rank 2 gets the second half. Ranks [1, 3] perform
        # the same split for the other EP lane. The EP policy then splits each
        # half once more, so every rank ultimately gets one quarter of input A.
        ctx.ranks = _local_cross_scale_ranks(
            expert_parallel_size,
            context_parallel_size,
        )
        expected_scale_units = context_parallel_size // expert_parallel_size
        if len(ctx.ranks) != expected_scale_units:
            raise ValueError(f'expected {expected_scale_units} scale units, got ranks {ctx.ranks}')
        return chunk(x, dim=1, ranks=ctx.ranks)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        # Reverse the forward chunk so the producer before the CP region receives
        # the gradient of the complete sequence.
        return all_gather(grad, dim=1, ranks=ctx.ranks), None, None


class _ScaleUnitContextAllGather(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        expert_parallel_size: int,
        context_parallel_size: int,
    ) -> torch.Tensor:
        # nnscaler has already combined the two EP sequence quarters inside each
        # scale unit. This all-gather then combines the two scale-unit halves and
        # restores the complete sequence for this CP group.
        ctx.ranks = _local_cross_scale_ranks(
            expert_parallel_size,
            context_parallel_size,
        )
        expected_scale_units = context_parallel_size // expert_parallel_size
        if len(ctx.ranks) != expected_scale_units:
            raise ValueError(f'expected {expected_scale_units} scale units, got ranks {ctx.ranks}')
        return all_gather(x, dim=1, ranks=ctx.ranks)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        # Each scale unit only needs the gradient for the sequence half it owned
        # in forward, so backward is the matching chunk operation.
        return chunk(grad, dim=1, ranks=ctx.ranks), None, None


class _GlobalContextMix(torch.autograd.Function):
    """
    A quick replacement of ring attention to test context parallelism.
    This operation sums the input across the context parallel group.
    """
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        full_sequence_length: int,
        context_replicas: int,
        context_parallel_size: int,
    ) -> torch.Tensor:
        # ranks == [0, 1, 2, 3] or [4, 5, 6, 7]
        # for CP=4, EP=2, runtime_ngpus=8
        ctx.ranks = _context_group_ranks(context_parallel_size)
        # context_replicas == 1 for CP=4, EP=2, runtime_ngpus=8
        ctx.denominator = full_sequence_length * context_replicas
        # The input is already partitioned across the context parallel group,
        # so we only need to sum across the context parallel group
        # The output is the input plus the sum of the input across the context parallel group,
        # divided by the full sequence length and the number of context replicas
        context_sum = all_reduce(x.sum(dim=1, keepdim=True), ranks=ctx.ranks)
        return x + context_sum / ctx.denominator

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        # The forward output at every rank depends on the global context sum.
        # Sum the context branch's gradients over the same CP group, then add
        # both the local residual path and the global-context path.
        context_grad = all_reduce(grad.sum(dim=1, keepdim=True), ranks=ctx.ranks)
        return grad + context_grad / ctx.denominator, None, None, None


def _fake_context_chunk(
    x: torch.Tensor,
    expert_parallel_size: int,
    context_parallel_size: int,
    num_scale_units: int,
) -> torch.Tensor:
    # fake_fn runs only while nnscaler traces the model. It describes the local
    # output shape without issuing a distributed collective.
    if num_scale_units != context_parallel_size // expert_parallel_size:
        raise ValueError('num_scale_units does not match CP/EP sizes')
    if x.shape[1] % num_scale_units:
        raise ValueError('sequence length must be divisible by num_scale_units')
    return x.chunk(num_scale_units, dim=1)[0]


def _fake_context_all_gather(
    x: torch.Tensor,
    expert_parallel_size: int,
    context_parallel_size: int,
    num_scale_units: int,
) -> torch.Tensor:
    # Shape-only tracing counterpart of _ScaleUnitContextAllGather.forward.
    if num_scale_units != context_parallel_size // expert_parallel_size:
        raise ValueError('num_scale_units does not match CP/EP sizes')
    return torch.cat([x] * num_scale_units, dim=1)


def _fake_routed_expert(
    x: torch.Tensor,
    expert_weight: torch.Tensor,
    expert_parallel_size: int,
    full_sequence_length: int,
    context_replicas: int,
    context_parallel_size: int,
) -> torch.Tensor:
    # Preserve shape and requires_grad during tracing. The real dispatch and
    # expert computation are emitted through routed_expert below.
    return x.clone()


@nnscaler.register_op(
    'b (num_scale_units s^) h^ -> b s^ h^',
    fake_fn=_fake_context_chunk,
)
def scale_unit_context_chunk(
    x: torch.Tensor,
    expert_parallel_size: int,
    context_parallel_size: int,
    num_scale_units: int,
) -> torch.Tensor:
    # This explicit consistency check keeps the symbolic annotation
    # ``(num_scale_units s)`` aligned with the runtime CP/EP topology.
    if num_scale_units != context_parallel_size // expert_parallel_size:
        raise ValueError('num_scale_units does not match CP/EP sizes')
    return _ScaleUnitContextChunk.apply(x, expert_parallel_size, context_parallel_size)


@nnscaler.register_op(
    'b s^ h^ -> b (num_scale_units s^) h^',
    fake_fn=_fake_context_all_gather,
)
def scale_unit_context_all_gather(
    x: torch.Tensor,
    expert_parallel_size: int,
    context_parallel_size: int,
    num_scale_units: int,
) -> torch.Tensor:
    # Use the same topology check as the entry boundary so the two Autograd
    # Functions are exact inverses of one another.
    if num_scale_units != context_parallel_size // expert_parallel_size:
        raise ValueError('num_scale_units does not match CP/EP sizes')
    return _ScaleUnitContextAllGather.apply(x, expert_parallel_size, context_parallel_size)


_EP_DISPATCH_COMBINE_RULE = TransformRule(
    # One partition operation performs both parts of EP:
    #   input activation D(1): split the local sequence half across EP ranks
    #   expert weight    D(0): split experts across the same EP ranks
    #   output           D(1): preserve the per-rank sequence quarter
    [DimopSplit.D(1), DimopSplit.D(0)],
    [DimopSplit.D(1)],
)


@nnscaler.register_op(
    'b s^ h^, E^ o^ h^ -> b s^ o^',
    transform_rules=(_EP_DISPATCH_COMBINE_RULE,),
    fake_fn=_fake_routed_expert,
)
def routed_expert(
    x: torch.Tensor,
    expert_weight: torch.Tensor,
    expert_parallel_size: int,
    full_sequence_length: int,
    context_replicas: int,
    context_parallel_size: int,
) -> torch.Tensor:
    """Dispatch tokens to expert owners, compute locally, and combine results.

    nnscaler has already partitioned both inputs before this function runs:

    * ``x`` contains this rank's sequence shard.
    * ``expert_weight`` contains only the experts owned by this EP rank.

    The all-to-all exchanges tokens inside the local EP group. Every rank sends
    one token block to each peer, receives the blocks for its own experts,
    computes those experts, and sends the results back to the source ranks.

    For the test configuration ``EP=2`` and ``local_experts=2``, an input with
    sequence length 32 is viewed as four blocks of 8 tokens::

        source rank's tokens: [rank0/expert0, rank0/expert1,
                               rank1/expert0, rank1/expert1]

    After dispatch, each rank owns both source ranks' blocks for its two local
    experts. Combine performs the inverse exchange and restores the original
    per-rank token order. The returned shape is therefore identical to ``x``.
    """
    global _LAST_EXPERT_SEQUENCE_LENGTH
    # For CP=4 and EP=2 the policy has already reduced S=128 to S/4=32.
    # The value is returned to the parent pytest process as an extra runtime check.
    # expert_parallel_ranks
    # rank 0/1 -> [0, 1]
    # rank 2/3 -> [2, 3]
    # rank 4/5 -> [4, 5]
    # rank 6/7 -> [6, 7]
    expert_parallel_ranks = _expert_parallel_ranks(expert_parallel_size)
    _LAST_EXPERT_SEQUENCE_LENGTH = x.shape[1]

    # Model a context-dependent operation (similar to attention) before expert
    # routing. This communicates only within the current CP group.
    x = _GlobalContextMix.apply(
        x,
        full_sequence_length,
        context_replicas,
        context_parallel_size,
    )
    batch, local_sequence, hidden = x.shape
    local_experts = expert_weight.shape[0]
    num_routing_blocks = expert_parallel_size * local_experts
    if local_sequence % num_routing_blocks:
        raise ValueError(
            f'local sequence length {local_sequence} must be divisible by '
            f'EP size * local experts ({num_routing_blocks})'
        )
    tokens_per_expert = local_sequence // num_routing_blocks

    # Arrange the sequence as [destination EP rank, local expert, tokens].
    # all-to-all splits dimension 1 into one equally sized message per peer.
    #
    # Example: EP=2, local_experts=2, local_sequence=32, so each block has
    # tokens_per_expert=8. On every source rank, the 32 tokens are interpreted as:
    #
    #   [dst rank 0 / expert 0 / 8 tokens,
    #    dst rank 0 / expert 1 / 8 tokens,
    #    dst rank 1 / expert 0 / 8 tokens,
    #    dst rank 1 / expert 1 / 8 tokens]
    #
    # Flattening the destination-rank, local-expert, and token dimensions back
    # to sequence gives two contiguous 16-token messages. all-to-all sends the
    # first message to EP rank 0 and the second message to EP rank 1. Therefore
    # rank 0 receives:
    #
    #   [source rank 0 / experts 0,1 / 16 tokens,
    #    source rank 1 / experts 0,1 / 16 tokens]
    #
    # and rank 1 receives the corresponding blocks for experts 2 and 3.
    #
    # A real MoE would first permute tokens according to router decisions. This
    # toy example omits that permutation and treats each consecutive 8-token
    # block as if the router had already assigned it to the indicated expert.
    dispatch_input = x.reshape(
        batch,
        expert_parallel_size,
        local_experts,
        tokens_per_expert,
        hidden,
    ).reshape(batch, local_sequence, hidden)
    dispatched = alltoall_alltoall(
        dispatch_input,
        idim=1,
        odim=1,
        ranks=expert_parallel_ranks,
    )

    # Received messages are grouped by source rank. Before permute, rank 0 sees:
    #
    #   [source rank 0 / local expert 0 / 8 tokens,
    #    source rank 0 / local expert 1 / 8 tokens,
    #    source rank 1 / local expert 0 / 8 tokens,
    #    source rank 1 / local expert 1 / 8 tokens]
    #
    # The reshape gives dimensions [source rank, local expert, tokens]. Permute
    # changes them to [local expert, source rank, tokens], so rank 0 now has:
    #
    #   local expert 0: [8 tokens from source 0, 8 tokens from source 1]
    #   local expert 1: [8 tokens from source 0, 8 tokens from source 1]
    #
    # Rank 1 has the same layout for its local experts, which are global experts
    # 2 and 3 in this example.
    dispatched = dispatched.reshape(
        batch,
        expert_parallel_size,
        local_experts,
        tokens_per_expert,
        hidden,
    ).permute(0, 2, 1, 3, 4)

    # Compute only the experts stored on this rank. Each local expert concatenates
    # its two 8-token source blocks and applies one linear layer to 16 tokens:
    #
    #   expert_outputs shape = [batch, 2 local experts, 16 tokens, hidden]
    expert_outputs = torch.stack([
        F.linear(
            dispatched[:, expert_idx].reshape(
                batch,
                expert_parallel_size * tokens_per_expert,
                hidden,
            ),
            weight,
        )
        for expert_idx, weight in enumerate(expert_weight)
    ], dim=1)

    # Combine is dispatch in reverse. First change
    #
    #   [local expert, source rank, tokens]
    #
    # back to
    #
    #   [destination source rank, local expert, tokens].
    #
    # On expert-owner rank 0 this forms two 16-token messages:
    #
    #   message to source rank 0: outputs of experts 0,1 for source 0
    #   message to source rank 1: outputs of experts 0,1 for source 1
    #
    # Expert-owner rank 1 similarly sends outputs of experts 2,3. After the
    # second all-to-all, original source rank 0 receives, in order:
    #
    #   [experts 0,1 output from owner rank 0,
    #    experts 2,3 output from owner rank 1]
    #
    # This matches its pre-dispatch block order, so combined has the same shape
    # and token order as x.
    combine_input = expert_outputs.reshape(
        batch,
        local_experts,
        expert_parallel_size,
        tokens_per_expert,
        hidden,
    ).permute(0, 2, 1, 3, 4).reshape(batch, local_sequence, hidden)
    combined = alltoall_alltoall(
        combine_input,
        idim=1,
        odim=1,
        ranks=expert_parallel_ranks,
    )
    return combined


class ContextExpertBlock(torch.nn.Module):
    def __init__(self, hidden_size: int, num_experts: int) -> None:
        super().__init__()
        self.expert_weight = torch.nn.Parameter(
            torch.empty(num_experts, hidden_size, hidden_size)
        )
        torch.nn.init.normal_(self.expert_weight, std=0.1)

    def forward(
        self,
        x: torch.Tensor,
        expert_parallel_size: int,
        full_sequence_length: int,
        context_replicas: int,
        context_parallel_size: int,
    ) -> torch.Tensor:
        # Every layer performs one global-context exchange followed by real EP
        # dispatch/compute/combine. The residual keeps all layers shape-compatible.
        return x + routed_expert(
            x,
            self.expert_weight,
            expert_parallel_size=expert_parallel_size,
            full_sequence_length=full_sequence_length,
            context_replicas=context_replicas,
            context_parallel_size=context_parallel_size,
        )


class ContextExpertModel(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        sequence_length: int,
        num_experts: int,
        num_layers: int,
        expert_parallel_size: int,
        context_parallel_size: int,
        runtime_ngpus: int,
        use_context_parallel: bool,
    ) -> None:
        super().__init__()
        if num_experts % expert_parallel_size:
            raise ValueError('num_experts must be divisible by expert_parallel_size')
        if runtime_ngpus % expert_parallel_size:
            raise ValueError('runtime_ngpus must be divisible by expert_parallel_size')
        if context_parallel_size % expert_parallel_size:
            raise ValueError('context_parallel_size must be divisible by expert_parallel_size')
        if runtime_ngpus % context_parallel_size:
            raise ValueError('runtime_ngpus must be divisible by context_parallel_size')

        self.layers = torch.nn.ModuleList(
            ContextExpertBlock(hidden_size, num_experts)
            for _ in range(num_layers)
        )
        self.expert_parallel_size = expert_parallel_size
        self.num_experts = num_experts
        self.full_sequence_length = sequence_length
        self.context_parallel_size = context_parallel_size
        # CP=4 / EP=2 => two scale units collaborate on each input sequence.
        self.num_scale_units = context_parallel_size // expert_parallel_size
        self.use_context_parallel = use_context_parallel

    def forward(self, data) -> torch.Tensor:
        x = data['data']
        if self.use_context_parallel:
            # Stage 1 of sequence partitioning:
            # [0, 2] (and [1, 3]) split S into two scale-unit halves.
            # after scale_unit_context_chunk,
            # [0, 1] have the first half of the sequence
            # and [2, 3] have the second half.
            x = scale_unit_context_chunk(
                x,
                expert_parallel_size=self.expert_parallel_size,
                context_parallel_size=self.context_parallel_size,
                num_scale_units=self.num_scale_units,
            )
            # The four CP ranks now hold complementary sequence quarters after
            # routed_expert's EP TransformRule, so no duplicate context exists.
            context_replicas = 1
        else:
            # Baseline: both scale units process the same complete sequence.
            # The context all-reduce therefore sees each token twice.
            context_replicas = self.num_scale_units

        output = x
        # Keep the sequence partitioned through all context/expert layers. This
        # is the memory-saving region; no full-sequence activation is materialized.
        for layer in self.layers:
            output = layer(
                output,
                expert_parallel_size=self.expert_parallel_size,
                full_sequence_length=self.full_sequence_length,
                context_replicas=context_replicas,
                context_parallel_size=self.context_parallel_size,
            )

        if self.use_context_parallel:
            # nnscaler first combines EP quarters inside each scale unit; this
            # boundary then gathers the two scale-unit halves into full S.
            output = scale_unit_context_all_gather(
                output,
                expert_parallel_size=self.expert_parallel_size,
                context_parallel_size=self.context_parallel_size,
                num_scale_units=self.num_scale_units,
            )
        return F.mse_loss(output, data['target'])


class LongSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, hidden_size: int, sequence_length: int, size: int) -> None:
        generator = torch.Generator().manual_seed(0)
        self.data = torch.randn(size, sequence_length, hidden_size, generator=generator)
        self.target = torch.randn(size, sequence_length, hidden_size, generator=generator)

    def __getitem__(self, index: int):
        return {'data': self.data[index], 'target': self.target[index]}

    def __len__(self) -> int:
        return len(self.data)


def cp_ep_policy(graph, compute_config: ComputeConfig):
    for node in get_pas_ops(graph):
        if node.fn == routed_expert:
            # Selecting expert_weight dim 0 activates _EP_DISPATCH_COMBINE_RULE,
            # which simultaneously partitions activation sequence dim 1.
            yield OpPlan(node, partition=OpPartition(input=1, dim=0))


def baseline_param_clss_fn(parameter_fqn: str) -> ParamBucketConfig:
    if parameter_fqn.startswith('layers.') and parameter_fqn.endswith('.expert_weight'):
        if _SCALE_UNITS_PER_CONTEXT_GROUP is None:
            raise RuntimeError('init_cp_ep_groups must run before optimizer construction')
        # Baseline mode: the two scale units in one CP group process the same
        # complete input, so average their duplicate expert gradients.
        return ParamBucketConfig(reducer_nreplicas=_SCALE_UNITS_PER_CONTEXT_GROUP)
    return ParamBucketConfig()


def _check_expert_bucket(trainer: Trainer, expected_nreplicas: int):
    # Each generated rank stores one half of every layer's expert weights.
    expert_parameters = {
        parameter: trainer.model.fullmap[generated_name]
        for generated_name, parameter in trainer.model.named_parameters()
        if trainer.model.fullmap[generated_name].orig_name.startswith('layers.')
        and trainer.model.fullmap[generated_name].orig_name.endswith('.expert_weight')
    }
    assert len(expert_parameters) == trainer.train_args.model.args['num_layers']
    for metadata in expert_parameters.values():
        assert metadata.sub_shape[0] * trainer.train_args.compute_config.plan_ngpus == metadata.shape[0]

    matched_parameters = set()
    for reducer in trainer.model.reducers:
        for bucket in reducer.buckets:
            bucket_parameters = set(bucket.params)
            matched = bucket_parameters & expert_parameters.keys()
            if not matched:
                continue
            assert bucket_parameters == matched
            assert bucket.nreplicas == expected_nreplicas
            # Same expert shards synchronize over [0,2,4,6] or [1,3,5,7].
            # This spans both scale units and, for runtime=8, both CP input groups.
            assert reducer.ranks == _expert_shard_reducer_ranks(
                trainer.train_args.compute_config.plan_ngpus
            )
            matched_parameters.update(matched)

    assert matched_parameters == expert_parameters.keys()
    expert_slices = {metadata.slicers[0] for metadata in expert_parameters.values()}
    assert len(expert_slices) == 1
    return next(iter(expert_slices))


def cp_ep_worker(
    save_dir,
    use_context_parallel: bool,
    runtime_ngpus: Optional[int] = None,
):
    run_name = 'cp' if use_context_parallel else 'baseline'
    save_dir = Path(save_dir)
    checkpoint_dir = save_dir / run_name / 'checkpoints'
    args = [
        '-f', str(CONFIG_PATH),
        '--instance_name', run_name,
        '--model.args.use_context_parallel', str(use_context_parallel),
        '--gen_savedir', str(save_dir / run_name / 'generated'),
        '--checkpoint.save_dir', str(checkpoint_dir),
        '--enable_progress_bar', 'false',
    ]
    if runtime_ngpus is not None:
        args.extend([
            '--compute_config.runtime_ngpus', str(runtime_ngpus),
            '--global_batch_size', str(runtime_ngpus),
        ])
    if not use_context_parallel:
        args.extend([
            '--optimizer.param_clss_fn',
            'tests.cli.test_cp_ep.baseline_param_clss_fn',
        ])

    trainer = Trainer(args)
    trainer.run()
    # Besides checkpoint parity, inspect runtime buckets so a wrong replica
    # divisor cannot be hidden by optimizer behavior.
    expert_slice = _check_expert_bucket(
        trainer,
        expected_nreplicas=(
            # CP keeps the nreplicas=1 generated for partitioned expert shards.
            1 if use_context_parallel else _SCALE_UNITS_PER_CONTEXT_GROUP
        ),
    )

    if trainer.rank == 0:
        Trainer.merge_checkpoint(
            list((checkpoint_dir / 'last').glob('*.ckpt')),
            save_dir / f'{run_name}.pt',
        )
    dist.barrier()
    return _LAST_EXPERT_SEQUENCE_LENGTH, expert_slice.start, expert_slice.stop


@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.device_count() >= 8,
    reason='covered by the real eight-GPU test',
)
@replace_all_device_with('cpu')
def test_cp4_ep2_runtime8_static(tmp_path):
    # No eight-GPU machine is required: compile eight rank-specific modules on
    # CPU and inspect their shapes, expert slices, and emitted reducer groups.
    trainer_args = TrainerArgs.from_cli([
        '-f', str(CONFIG_PATH),
        '--compute_config.runtime_ngpus', '8',
        '--global_batch_size', '8',
        '--model.args.use_context_parallel', 'true',
    ])
    compute_config = trainer_args.compute_config
    assert compute_config.plan_ngpus == 2
    assert compute_config.runtime_ngpus == 8
    assert trainer_args.model.args['context_parallel_size'] == 4
    assert trainer_args.model.args['runtime_ngpus'] == 8

    # Two independent inputs are processed by [0,4) and [4,8).
    assert {
        rank: _context_group_ranks(4, rank)
        for rank in range(8)
    } == {
        0: (0, 1, 2, 3), 1: (0, 1, 2, 3),
        2: (0, 1, 2, 3), 3: (0, 1, 2, 3),
        4: (4, 5, 6, 7), 5: (4, 5, 6, 7),
        6: (4, 5, 6, 7), 7: (4, 5, 6, 7),
    }
    # The outer CP boundary communicates only between same-EP-lane scale units
    # inside one input group; it never mixes input A with input B.
    assert {
        rank: _local_cross_scale_ranks(2, 4, rank)
        for rank in range(8)
    } == {
        0: (0, 2), 1: (1, 3), 2: (0, 2), 3: (1, 3),
        4: (4, 6), 5: (5, 7), 6: (4, 6), 7: (5, 7),
    }
    # Weight gradients synchronize globally by expert ownership, not by input.
    assert {
        rank: _expert_shard_reducer_ranks(2, rank, 8)
        for rank in range(8)
    } == {
        0: (0, 2, 4, 6), 1: (1, 3, 5, 7),
        2: (0, 2, 4, 6), 3: (1, 3, 5, 7),
        4: (0, 2, 4, 6), 5: (1, 3, 5, 7),
        6: (0, 2, 4, 6), 7: (1, 3, 5, 7),
    }

    # Dataset samples are replicated inside each CP group and sharded between
    # CP groups: scale units 0/1 see A, while 2/3 see B.
    sampler_indices = [
        list(ContextGroupSampler(
            range(8),
            num_replicas=compute_config.runtime_ngpus // compute_config.plan_ngpus,
            rank=scale_unit_rank,
            context_parallel_size=trainer_args.model.args['context_parallel_size'],
            expert_parallel_size=compute_config.plan_ngpus,
        ))
        for scale_unit_rank in range(4)
    ]
    assert sampler_indices[0] == sampler_indices[1]
    assert sampler_indices[2] == sampler_indices[3]
    assert set(sampler_indices[0]).isdisjoint(sampler_indices[2])
    assert set(sampler_indices[0] + sampler_indices[2]) == set(range(8))

    # Capture plan-level routed_expert inputs. S=128 becomes S/4=32 because
    # outer cross-scale chunk and inner EP partition each divide by two.
    local_routed_shapes = []

    def capture_policy(graph, cfg):
        graph = fn(graph, cfg, cp_ep_policy)
        routed_nodes = graph.select(name='routed_expert')
        local_routed_shapes.extend(
            (node.device[0], node.input(0).shape, node.input(1).shape)
            for node in routed_nodes
        )
        return graph

    instance_name = 'cp4_ep2_runtime8'
    parallelize(
        trainer_args.create_model(),
        {'data': trainer_args.dummy_input},
        capture_policy,
        compute_config,
        gen_savedir=tmp_path,
        instance_name=instance_name,
        load_module=False,
        reuse='override',
    )

    sequence_length = trainer_args.get_resolved_var('sequence_length')
    hidden_size = trainer_args.get_resolved_var('hidden_size')
    num_experts = trainer_args.get_resolved_var('num_experts')
    context_parallel_size = trainer_args.model.args['context_parallel_size']
    assert sorted(local_routed_shapes) == sorted([
        (0, (trainer_args.micro_batch_size, sequence_length // context_parallel_size, hidden_size),
            (num_experts // 2, hidden_size, hidden_size)),
        (1, (trainer_args.micro_batch_size, sequence_length // context_parallel_size, hidden_size),
            (num_experts // 2, hidden_size, hidden_size)),
    ] * trainer_args.model.args['num_layers'])

    for rank in range(compute_config.runtime_ngpus):
        module_class = _load_parallel_module_class(
            ContextExpertModel,
            gen_savedir=tmp_path,
            instance_name=instance_name,
            rank=rank,
        )
        expert_metas = sorted(
            (
                metadata.orig_name,
                metadata.slicers[0].start,
                metadata.slicers[0].stop,
            )
            for metadata in module_class.attr_meta_maps[rank].values()
            if metadata.orig_name.endswith('.expert_weight')
        )
        # Expert ownership repeats in every scale unit and every CP input group.
        expected_slice = (0, 2) if rank % 2 == 0 else (2, 4)
        assert len(expert_metas) == trainer_args.model.args['num_layers']
        assert {
            (start, stop)
            for _, start, stop in expert_metas
        } == {expected_slice}

        # Generated reducers must follow the same repeated expert ownership.
        expected_ranks = '[0, 2, 4, 6]' if rank % 2 == 0 else '[1, 3, 5, 7]'
        assert _gencode_contains(
            tmp_path,
            ContextExpertModel,
            rank,
            rf'Reducer\(ranks={re.escape(expected_ranks)}',
            instance_name=instance_name,
        )


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 8,
    reason='lack of gpu devices',
)
def test_cp4_ep2_runtime8(tmp_path):
    runtime_ngpus = 8
    trainer_args = TrainerArgs.from_cli([
        '-f', str(CONFIG_PATH),
        '--compute_config.runtime_ngpus', str(runtime_ngpus),
        '--global_batch_size', str(runtime_ngpus),
    ])
    plan_ngpus = trainer_args.compute_config.plan_ngpus
    sequence_length = trainer_args.get_resolved_var('sequence_length')

    baseline_lengths = launch_torchrun(
        runtime_ngpus,
        cp_ep_worker,
        tmp_path,
        False,
        runtime_ngpus,
    )
    cp_lengths = launch_torchrun(
        runtime_ngpus,
        cp_ep_worker,
        tmp_path,
        True,
        runtime_ngpus,
    )

    assert {result[0] for result in baseline_lengths.values()} == {sequence_length // plan_ngpus}
    assert {result[0] for result in cp_lengths.values()} == {
        sequence_length // trainer_args.model.args['context_parallel_size']
    }
    assert {
        rank: result[1:]
        for rank, result in cp_lengths.items()
    } == {
        rank: (0, 2) if rank % 2 == 0 else (2, 4)
        for rank in range(runtime_ngpus)
    }

    baseline = torch.load(tmp_path / 'baseline.pt', weights_only=False)
    context_parallel = torch.load(tmp_path / 'cp.pt', weights_only=False)
    assert_close(baseline['model'], context_parallel['model'], atol=1e-6, rtol=1e-6)
    assert_close(baseline['optimizer'], context_parallel['optimizer'], atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 4,
    reason='lack of gpu devices',
)
def test_cp_across_scale_units_with_ep(tmp_path):
    trainer_args = TrainerArgs.from_yaml(str(CONFIG_PATH))
    runtime_ngpus = trainer_args.compute_config.runtime_ngpus  # 4
    plan_ngpus = trainer_args.compute_config.plan_ngpus  # 2
    sequence_length = trainer_args.get_resolved_var('sequence_length')
    # Baseline and CP runs use identical EP dispatch/combine. Only the outer
    # cross-scale sequence partition and bucket divisor differ.
    baseline_lengths = launch_torchrun(runtime_ngpus, cp_ep_worker, tmp_path, False)
    cp_lengths = launch_torchrun(runtime_ngpus, cp_ep_worker, tmp_path, True)

    assert {result[0] for result in baseline_lengths.values()} == {sequence_length // plan_ngpus}
    assert {result[0] for result in cp_lengths.values()} == {sequence_length // runtime_ngpus}
    assert {
        rank: result[1:]
        for rank, result in cp_lengths.items()
    } == {
        0: (0, 2),
        1: (2, 4),
        2: (0, 2),
        3: (2, 4),
    }

    baseline = torch.load(tmp_path / 'baseline.pt', weights_only=False)
    context_parallel = torch.load(tmp_path / 'cp.pt', weights_only=False)
    assert_close(baseline['model'], context_parallel['model'], atol=1e-6, rtol=1e-6)
    assert_close(baseline['optimizer'], context_parallel['optimizer'], atol=1e-6, rtol=1e-6)
