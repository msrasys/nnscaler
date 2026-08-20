from pathlib import Path
from typing import Optional, Tuple

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

import nnscaler
from nnscaler.cli.trainer import Trainer
from nnscaler.cli.trainer_args import TrainerArgs
from nnscaler.parallel import ComputeConfig
from nnscaler.policies import OpPartition, OpPlan, get_pas_ops
from nnscaler.runtime.adapter.collectives import all_gather, chunk
from nnscaler.runtime.adapter.reducer import ParamBucketConfig
from nnscaler.runtime.device import DeviceGroup
from tests.launch_torchrun import launch_torchrun
from tests.parallel_module.common import assert_close


CONFIG_PATH = Path(__file__).with_name('trainer_args_simulated_dp.yaml')
DP_SHARDED_CONFIG_PATH = Path(__file__).with_name('trainer_args_simulated_dp_dp_sharded.yaml')
# Test-only probes returned by each torchrun worker; they do not affect model execution.
_OPAQUE_BATCH_SIZES = set()
# Test-only probes returned by each torchrun worker; they do not affect model execution.
_DP_SHARDED_DATALOADER_INFO = {}


def _scale_unit_ranks(group_size: int) -> Tuple[int, ...]:
    rank = dist.get_rank()
    first_rank = rank // group_size * group_size
    return tuple(range(first_rank, first_rank + group_size))


def init_scale_unit_groups(trainer: Trainer) -> None:
    compute_config = trainer.train_args.compute_config
    group_size = compute_config.plan_ngpus
    world_size = dist.get_world_size()
    if world_size != compute_config.runtime_ngpus:
        raise ValueError(f'world size {world_size} does not match runtime_ngpus {compute_config.runtime_ngpus}')
    if world_size % group_size:
        raise ValueError(f'world size {world_size} must be divisible by {group_size}')
    for first_rank in range(0, world_size, group_size):
        DeviceGroup().get_group(tuple(range(first_rank, first_rank + group_size)))


class _ScaleUnitChunk(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, group_size: int) -> torch.Tensor:
        ctx.ranks = _scale_unit_ranks(group_size)
        return chunk(x, dim=0, ranks=ctx.ranks)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return all_gather(grad, dim=0, ranks=ctx.ranks), None


class _ScaleUnitAllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, group_size: int) -> torch.Tensor:
        ctx.ranks = _scale_unit_ranks(group_size)
        return all_gather(x, dim=0, ranks=ctx.ranks)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return chunk(grad, dim=0, ranks=ctx.ranks), None


def _fake_chunk(x: torch.Tensor, group_size: int) -> torch.Tensor:
    if x.shape[0] % group_size:
        raise ValueError('batch size must be divisible by the scale-unit size')
    return x.chunk(group_size, dim=0)[0]


def _fake_all_gather(x: torch.Tensor, group_size: int) -> torch.Tensor:
    return torch.cat([x] * group_size, dim=0)


@nnscaler.register_op('(group_size b) s^ h^ -> b s^ h^', fake_fn=_fake_chunk)
def scale_unit_chunk(x: torch.Tensor, group_size: int) -> torch.Tensor:
    return _ScaleUnitChunk.apply(x, group_size)


@nnscaler.register_op('b s^ h^ -> (group_size b) s^ h^', fake_fn=_fake_all_gather)
def scale_unit_all_gather(x: torch.Tensor, group_size: int) -> torch.Tensor:
    return _ScaleUnitAllGather.apply(x, group_size)


def _fake_slow_block(
    x: torch.Tensor,
    up_weight: torch.Tensor,
    up_bias: torch.Tensor,
    down_weight: torch.Tensor,
    down_bias: torch.Tensor,
) -> torch.Tensor:
    # we must make sure the return tensor has requires_grad=True, otherwise the gradient will be None and the test will fail
    return torch.randn_like(x, requires_grad=True)


@nnscaler.register_op(
    'b s^ h^, i^ h^, i^, h^ i^, h^ -> b s^ h^',
    fake_fn=_fake_slow_block,
)
def opaque_slow_block(
    x: torch.Tensor,
    up_weight: torch.Tensor,
    up_bias: torch.Tensor,
    down_weight: torch.Tensor,
    down_bias: torch.Tensor,
) -> torch.Tensor:
    _OPAQUE_BATCH_SIZES.add(x.shape[0])
    hidden = F.silu(F.linear(x, up_weight, up_bias))
    return F.linear(hidden, down_weight, down_bias)


class ProjectionModule(torch.nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.linear(x))


class DynamicShapeSubmodel(torch.nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.up = torch.nn.Linear(hidden_size, hidden_size * 2)
        self.down = torch.nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return opaque_slow_block(
            x,
            self.up.weight,
            self.up.bias,
            self.down.weight,
            self.down.bias,
        )


class SimulatedDPModel(torch.nn.Module):
    def __init__(self, hidden_size: int, scale_unit_size: int, use_scale_unit_dp: bool) -> None:
        super().__init__()
        self.pre = ProjectionModule(hidden_size)
        self.dynamic_block = DynamicShapeSubmodel(hidden_size)
        self.post = ProjectionModule(hidden_size)
        self.scale_unit_size = scale_unit_size
        self.use_scale_unit_dp = use_scale_unit_dp

    def forward(self, data) -> torch.Tensor:
        x = self.pre(data['data'])
        if self.use_scale_unit_dp:
            x = scale_unit_chunk(x, group_size=self.scale_unit_size)
        x = self.dynamic_block(x)
        if self.use_scale_unit_dp:
            x = scale_unit_all_gather(x, group_size=self.scale_unit_size)
        output = self.post(x)
        return F.mse_loss(output, data['target'])


class SimulatedDPDataset(torch.utils.data.Dataset):
    def __init__(self, hidden_size: int, sequence_length: int, size: int) -> None:
        generator = torch.Generator().manual_seed(0)
        self.data = torch.randn(size, sequence_length, hidden_size, generator=generator)
        self.target = torch.randn(size, sequence_length, hidden_size, generator=generator)

    def __getitem__(self, index: int):
        return {'data': self.data[index], 'target': self.target[index]}

    def __len__(self) -> int:
        return len(self.data)


class DPShardedSimulatedDPDataset(SimulatedDPDataset):
    """
    Reorder samples so global-rank shards reconstruct scale-unit batches.
    Note: this is only for testing (to verify that the DP-sharded model produces the same results as the baseline model).
    In practice, this dataset is not necessary.
    """

    def __init__(
        self,
        hidden_size: int,
        sequence_length: int,
        size: int,
        dp_sharded: bool,
        plan_ngpus: int,
        runtime_ngpus: int,
        logical_micro_batch_size: int,
    ) -> None:
        super().__init__(hidden_size, sequence_length, size)
        self.source_indices = list(range(size))
        if not dp_sharded:
            return

        if runtime_ngpus % plan_ngpus:
            raise ValueError('runtime_ngpus must be divisible by plan_ngpus')
        if logical_micro_batch_size % plan_ngpus:
            raise ValueError('logical_micro_batch_size must be divisible by plan_ngpus')

        num_scale_units = runtime_ngpus // plan_ngpus
        global_batch_size = logical_micro_batch_size * num_scale_units
        if size % global_batch_size:
            raise ValueError('dataset size must be divisible by the logical global batch size')

        rank_batch_size = logical_micro_batch_size // plan_ngpus
        source_indices = [None] * size
        for step in range(size // global_batch_size):
            for scale_unit_rank in range(num_scale_units):
                for plan_rank in range(plan_ngpus):
                    global_rank = scale_unit_rank * plan_ngpus + plan_rank
                    for rank_batch_index in range(rank_batch_size):
                        dataset_index = global_rank + (
                            step * rank_batch_size + rank_batch_index
                        ) * runtime_ngpus
                        source_index = scale_unit_rank + (
                            step * logical_micro_batch_size
                            + plan_rank * rank_batch_size
                            + rank_batch_index
                        ) * num_scale_units
                        source_indices[dataset_index] = source_index

        assert all(index is not None for index in source_indices)
        self.source_indices = source_indices
        self.data = self.data[source_indices]
        self.target = self.target[source_indices]


class ScaleUnitDPShardedSampler(torch.utils.data.DistributedSampler):
    """Split each scale-unit sampler batch among its plan ranks."""

    def __init__(
        self,
        dataset,
        num_replicas,
        rank,
        dp_sharded: bool,
        plan_ngpus: int,
        **kwargs,
    ) -> None:
        if dp_sharded:
            num_replicas *= plan_ngpus
            rank = dist.get_rank()
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, **kwargs)


class ScaleUnitDPShardedDataLoader(torch.utils.data.DataLoader):
    """Convert Trainer's logical micro-batch size to a rank-local batch size."""

    def __init__(
        self,
        *args,
        batch_size: int,
        plan_ngpus: int,
        dp_sharded: bool,
        stage: str,
        **kwargs,
    ) -> None:
        if dp_sharded:
            if batch_size % plan_ngpus:
                raise ValueError('micro_batch_size must be divisible by plan_ngpus')
            batch_size //= plan_ngpus
        super().__init__(*args, batch_size=batch_size, **kwargs)
        # for testing only: record the sampler indices and their corresponding source indices for each rank
        sampler_indices = tuple(iter(self.sampler))
        _DP_SHARDED_DATALOADER_INFO[stage] = (
            self.batch_size,
            self.sampler.num_replicas,
            self.sampler.rank,
            sampler_indices,
            tuple(self.dataset.source_indices[index] for index in sampler_indices),
        )


def dp_sharded_dummy_sample(trainer_args: TrainerArgs):
    """Generate the runtime-equivalent batch shape used only for tracing."""

    batch_size = trainer_args.micro_batch_size
    if trainer_args.get_resolved_var('dp_sharded'):
        plan_ngpus = trainer_args.compute_config.plan_ngpus
        if batch_size % plan_ngpus:
            raise ValueError('micro_batch_size must be divisible by plan_ngpus')
        batch_size //= plan_ngpus
    sequence_length = trainer_args.get_resolved_var('sequence_length')
    hidden_size = trainer_args.get_resolved_var('hidden_size')
    return {
        'data': torch.empty(batch_size, sequence_length, hidden_size),
        'target': torch.empty(batch_size, sequence_length, hidden_size),
    }


class LeadingSimulatedDPModel(torch.nn.Module):
    """Run the opaque block directly on rank-local data at model entry."""

    def __init__(self, hidden_size: int, scale_unit_size: int, dp_sharded: bool) -> None:
        super().__init__()
        self.dynamic_block = DynamicShapeSubmodel(hidden_size)
        self.post = ProjectionModule(hidden_size)
        self.scale_unit_size = scale_unit_size
        self.dp_sharded = dp_sharded

    def forward(self, data) -> torch.Tensor:
        x = self.dynamic_block(data['data'])
        target = data['target']
        if self.dp_sharded:
            x = scale_unit_all_gather(x, group_size=self.scale_unit_size)
            target = scale_unit_all_gather(target, group_size=self.scale_unit_size)
        output = self.post(x)
        return F.mse_loss(output, target)


def simulated_dp_policy(graph, compute_config: ComputeConfig):
    for node in get_pas_ops(graph):
        if ProjectionModule not in node.module_class_chain:
            continue
        if node.fn == F.linear:
            yield OpPlan(node, partition=OpPartition(input=1, dim=0))
        else:
            yield OpPlan(node, partition='auto')


def simulated_dp_param_clss_fn(parameter_fqn: str) -> ParamBucketConfig:
    if parameter_fqn.startswith('dynamic_block.'):
        return ParamBucketConfig(reducer_nreplicas=1)
    return ParamBucketConfig()


def dp_sharded_simulated_dp_param_clss_fn(
    trainer_args: TrainerArgs,
    parameter_fqn: str,
) -> ParamBucketConfig:
    if parameter_fqn.startswith('dynamic_block.'):
        nreplicas = 1 if trainer_args.get_resolved_var('dp_sharded') \
            else trainer_args.compute_config.plan_ngpus
        return ParamBucketConfig(reducer_nreplicas=nreplicas)
    return ParamBucketConfig()


def _check_dynamic_block_buckets(
    trainer: Trainer,
    expected_nreplicas: int,
    expected_ranks: Optional[Tuple[int, ...]] = None,
) -> None:
    target_parameters = {
        parameter
        for generated_name, parameter in trainer.model.named_parameters()
        if trainer.model.fullmap[generated_name].orig_name.startswith('dynamic_block.')
    }
    matched_parameters = set()
    for reducer in trainer.model.reducers:
        for bucket in reducer.buckets:
            bucket_parameters = set(bucket.params)
            matched = bucket_parameters & target_parameters
            if not matched:
                continue
            assert bucket_parameters == matched
            assert bucket.nreplicas == expected_nreplicas
            if expected_ranks is not None:
                assert tuple(reducer.ranks) == expected_ranks
            matched_parameters.update(matched)
    assert matched_parameters == target_parameters


def simulated_dp_worker(save_dir, use_scale_unit_dp: bool):
    run_name = 'simulated' if use_scale_unit_dp else 'baseline'
    save_dir = Path(save_dir)
    checkpoint_dir = save_dir / run_name / 'checkpoints'
    args = [
        '-f', str(CONFIG_PATH),
        '--instance_name', run_name,
        '--model.args.use_scale_unit_dp', str(use_scale_unit_dp),
        '--gen_savedir', str(save_dir / run_name / 'generated'),
        '--checkpoint.save_dir', str(checkpoint_dir),
        '--enable_progress_bar', 'false',
    ]
    if use_scale_unit_dp:
        args.extend([
            '--optimizer.param_clss_fn',
            'tests.cli.test_simulated_dp.simulated_dp_param_clss_fn',
        ])

    trainer = Trainer(args)
    trainer.run()
    _check_dynamic_block_buckets(
        trainer,
        expected_nreplicas=1 if use_scale_unit_dp else trainer.train_args.compute_config.plan_ngpus,
    )

    if trainer.rank == 0:
        Trainer.merge_checkpoint(
            list((checkpoint_dir / 'last').glob('*.ckpt')),
            save_dir / f'{run_name}.pt',
        )
    dist.barrier()


def dp_sharded_simulated_dp_worker(save_dir, dp_sharded: bool):
    run_name = 'dp_sharded' if dp_sharded else 'leading_baseline'
    save_dir = Path(save_dir)
    checkpoint_dir = save_dir / run_name / 'checkpoints'
    args = [
        '-f', str(DP_SHARDED_CONFIG_PATH),
        '--instance_name', run_name,
        '--vars.dp_sharded', str(dp_sharded),
        '--gen_savedir', str(save_dir / run_name / 'generated'),
        '--checkpoint.save_dir', str(checkpoint_dir),
        '--enable_progress_bar', 'false',
    ]
    trainer = Trainer(args)
    trainer.run()
    _check_dynamic_block_buckets(
        trainer,
        expected_nreplicas=(
            1 if dp_sharded else trainer.train_args.compute_config.plan_ngpus
        ),
        expected_ranks=tuple(range(trainer.world_size)),
    )

    if trainer.rank == 0:
        Trainer.merge_checkpoint(
            list((checkpoint_dir / 'last').glob('*.ckpt')),
            save_dir / f'{run_name}.pt',
        )
    dist.barrier()
    return (
        tuple(trainer.dummy_input['data'].shape),
        tuple(sorted(_OPAQUE_BATCH_SIZES)),
        dict(_DP_SHARDED_DATALOADER_INFO),
    )


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4, reason='lack of gpu devices')
def test_simulated_dp_cli(tmp_path):
    launch_torchrun(4, simulated_dp_worker, tmp_path, False)
    launch_torchrun(4, simulated_dp_worker, tmp_path, True)

    baseline = torch.load(tmp_path / 'baseline.pt', weights_only=False)
    simulated = torch.load(tmp_path / 'simulated.pt', weights_only=False)
    # Full- and half-batch GEMMs can use different float32 accumulation orders.
    assert_close(baseline['model'], simulated['model'], atol=1e-6, rtol=1e-6)
    assert_close(baseline['optimizer'], simulated['optimizer'], atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4, reason='lack of gpu devices')
def test_simulated_dp_with_dp_sharded_input(tmp_path):
    baseline_results = launch_torchrun(4, dp_sharded_simulated_dp_worker, tmp_path, False)
    dp_sharded_results = launch_torchrun(4, dp_sharded_simulated_dp_worker, tmp_path, True)

    assert {result[0] for result in baseline_results.values()} == {(4, 7, 16)}
    assert {result[0] for result in dp_sharded_results.values()} == {(2, 7, 16)}
    assert {result[1] for result in baseline_results.values()} == {(4,)}
    assert {result[1] for result in dp_sharded_results.values()} == {(2,)}

    baseline_loader_info = {
        rank: result[2]['train']
        for rank, result in baseline_results.items()
    }
    dp_sharded_loader_info = {
        rank: result[2]['train']
        for rank, result in dp_sharded_results.items()
    }
    assert {
        rank: info[:3]
        for rank, info in dp_sharded_loader_info.items()
    } == {
        rank: (2, 4, rank)
        for rank in range(4)
    }
    rank_indices = [set(dp_sharded_loader_info[rank][3]) for rank in range(4)]
    assert all(rank_indices[i].isdisjoint(rank_indices[j]) for i in range(4) for j in range(i + 1, 4))
    assert set().union(*rank_indices) == set(range(8))
    assert tuple(dp_sharded_loader_info[0][4] + dp_sharded_loader_info[1][4]) == baseline_loader_info[0][4]
    assert tuple(dp_sharded_loader_info[2][4] + dp_sharded_loader_info[3][4]) == baseline_loader_info[2][4]

    generated_files = list((tmp_path / 'dp_sharded' / 'generated').glob('**/gencode*.py'))
    assert len(generated_files) == 4
    for generated_file in generated_files:
        generated_code = generated_file.read_text()
        assert 'scale_unit_all_gather' in generated_code
        assert 'scale_unit_chunk' not in generated_code

    baseline = torch.load(tmp_path / 'leading_baseline.pt', weights_only=False)
    dp_sharded = torch.load(tmp_path / 'dp_sharded.pt', weights_only=False)
    assert_close(baseline['model'], dp_sharded['model'], atol=1e-6, rtol=1e-6)
    assert_close(baseline['optimizer'], dp_sharded['optimizer'], atol=1e-6, rtol=1e-6)
