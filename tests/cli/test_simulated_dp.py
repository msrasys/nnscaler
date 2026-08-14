from pathlib import Path
from typing import Tuple

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
    return x.clone()


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


def _check_dynamic_block_buckets(trainer: Trainer, expected_nreplicas: int) -> None:
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


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4, reason='lack of gpu devices')
def test_simulated_dp_cli(tmp_path):
    launch_torchrun(4, simulated_dp_worker, tmp_path, False)
    launch_torchrun(4, simulated_dp_worker, tmp_path, True)

    baseline = torch.load(tmp_path / 'baseline.pt', weights_only=False)
    simulated = torch.load(tmp_path / 'simulated.pt', weights_only=False)
    # Full- and half-batch GEMMs can use different float32 accumulation orders.
    assert_close(baseline['model'], simulated['model'], atol=1e-6, rtol=1e-6)
    assert_close(baseline['optimizer'], simulated['optimizer'], atol=1e-6, rtol=1e-6)
