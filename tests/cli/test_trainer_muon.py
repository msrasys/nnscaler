from pathlib import Path
import shutil

import torch

import pytest

from nnscaler.cli.trainer import Trainer
from tests.launch_torchrun import launch_torchrun
from nnscaler.cli.trainer import logger
from tests.utils import catch_log
from tests.parallel_module.common import assert_close, assert_equal

available_optimizers = []
try:
    from torch.optim import Muon
    MuonAvailable = True
    available_optimizers.append('muon')
except ImportError:
    MuonAvailable = False

try:
    import dion
    DionAvailable = True
    available_optimizers.append('dion')
except ImportError:
    DionAvailable = False


def get_optimizer_type(name):
    if name == 'muon':
        return 'nnscaler.runtime.muon_optimizer.Muon'
    else:
        return 'nnscaler.runtime.dion_optimizer.Muon'


def get_original_optimizer_type(name):
    if name == 'muon':
        return 'torch.optim.Muon'
    else:
        return 'dion.Muon'


def trainer_muon_worker(save_dir, config_file, name, additional_options=None):
    save_dir = Path(save_dir)
    config_path = Path(__file__).with_name(config_file).resolve()
    work_dir = save_dir / str(name)
    gen_savedir = work_dir / 'gen'
    ckpt_savedir = work_dir / 'ckpt'
    additional_options = additional_options or []

    # train first epoch
    trainer = Trainer([
        '-f', config_path,
        '--max_epochs', '1',
        '--gen_savedir', str(gen_savedir),
        '--checkpoint.save_dir', str(ckpt_savedir),
        *additional_options,
    ])
    trainer.run()
    torch.distributed.barrier()

    # create merged checkpoint
    if trainer.rank == 0:
        Trainer.merge_checkpoint(list((ckpt_savedir / 'last').glob('*.ckpt')), ckpt_savedir / 'merged.pt')

    torch.distributed.barrier()

    # train 2nd epoch, resume from merged checkpoint
    trainer = Trainer([
        '-f', config_path,
        '--max_epochs', '2',
        '--gen_savedir', str(gen_savedir),
        '--checkpoint.save_dir', str(ckpt_savedir),
        '--checkpoint.resume_from', str(ckpt_savedir / 'merged.pt'),
        *additional_options,
    ])
    trainer.run()

    torch.distributed.barrier()

    # train 3rd epoch, resume from deduped checkpoint
    trainer = Trainer([
        '-f', config_path,
        '--max_epochs', '3',
        '--gen_savedir', str(gen_savedir),
        '--checkpoint.save_dir', str(ckpt_savedir),
        '--checkpoint.resume_from', 'last',
        '--checkpoint.save_type', 'sharded',
        *additional_options,
    ])
    trainer.run()

    torch.distributed.barrier()

    # train 4th epoch, resume from sharded checkpoint
    trainer = Trainer([
        '-f', config_path,
        '--max_epochs', '4',
        '--gen_savedir', str(gen_savedir),
        '--checkpoint.save_dir', str(ckpt_savedir),
        '--checkpoint.resume_from', 'last',
        *additional_options,
    ])
    trainer.run()

    torch.distributed.barrier()

    ckpt1_savedir = save_dir / 'ckpt1'
    # train 4 epoch without resuming
    trainer = Trainer([
        '-f', config_path,
        '--max_epochs', '4',
        '--gen_savedir', str(gen_savedir),
        '--checkpoint.save_dir', str(ckpt1_savedir),
        *additional_options,
    ])
    trainer.run()

    torch.distributed.barrier()

    if trainer.rank == 0:
        for i in range(2):
            x = torch.load(ckpt_savedir / 'last' / f'{i}.ckpt', weights_only=False)
            y = torch.load(ckpt1_savedir / 'last' / f'{i}.ckpt', weights_only=False)
            for key in ['model', 'optimizer']:
                assert_equal(x[key], y[key])

    if trainer.rank == 0:
        Trainer.merge_checkpoint(list((ckpt_savedir / 'last').glob('*.ckpt')), work_dir / 'result.pt')

    torch.distributed.barrier()


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason='lack of gpu devices')
@pytest.mark.parametrize('name', available_optimizers)
def test_trainer_muon_resume_correctness_basic(tmp_path, name):
    config_file = 'trainer_args_muon.yaml'
    launch_torchrun(2, trainer_muon_worker, tmp_path, config_file, 'False', [
        '--optimizer.type', get_original_optimizer_type(name)
    ])
    launch_torchrun(2, trainer_muon_worker, tmp_path, config_file, 'True', [
        '--compute_config.zero_param_level_sharding', True,
        '--compute_config.use_zero', 1,
        '--optimizer.type', get_optimizer_type(name),
    ])

    zero0_ckpt = torch.load(tmp_path / 'False' / 'result.pt', weights_only=False)
    zero1_ckpt = torch.load(tmp_path / 'True' / 'result.pt', weights_only=False)

    assert_equal(zero0_ckpt['model'], zero1_ckpt['model'])
    assert_equal(zero0_ckpt['optimizer']['state'], zero1_ckpt['optimizer']['state'])


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4, reason='lack of gpu devices')
@pytest.mark.parametrize('name', available_optimizers)
def test_trainer_muon_resume_correctness_zero1(tmp_path, name):
    config_file = 'trainer_args_muon.yaml'
    launch_torchrun(4, trainer_muon_worker, tmp_path, config_file, '1', [
        '--compute_config.zero_param_level_sharding', True,
        '--compute_config.runtime_ngpus', 4,
        '--compute_config.use_zero', 1,
        '--optimizer.type', get_optimizer_type(name),
    ])
    launch_torchrun(4, trainer_muon_worker, tmp_path, config_file, '2', [
        '--compute_config.zero_param_level_sharding', True,
        '--compute_config.runtime_ngpus', 4,
        '--compute_config.use_zero', 0,
        '--optimizer.type', get_optimizer_type(name),
    ])

    zero0_ckpt = torch.load(tmp_path / '1' / 'result.pt', weights_only=False)
    zero1_ckpt = torch.load(tmp_path / '2' / 'result.pt', weights_only=False)

    assert_equal(zero0_ckpt['model'], zero1_ckpt['model'])
    assert_equal(zero0_ckpt['optimizer']['state'], zero1_ckpt['optimizer']['state'])


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason='lack of gpu devices')
@pytest.mark.parametrize('name', available_optimizers)
def test_trainer_muon_resume_correctness_zero1_param_config(tmp_path, name):
    config_file = 'trainer_args_muon_hybrid.yaml'
    launch_torchrun(2, trainer_muon_worker, tmp_path, config_file, '1', [
        '--optimizer.args.config.optimizers.1.type', get_original_optimizer_type(name),
    ])
    launch_torchrun(2, trainer_muon_worker, tmp_path, config_file, '2', [
        '--compute_config.zero_param_level_sharding', True,
        '--compute_config.use_zero', 1,
        '--optimizer.args.config.optimizers.1.type', get_optimizer_type(name),
    ])
    launch_torchrun(2, trainer_muon_worker, tmp_path, config_file, '3', [
        '--compute_config.use_zero', 1,
        '--optimizer.args.config.optimizers.1.type', get_optimizer_type(name),
        '--optimizer.param_clss_fn', 'tests.cli.test_trainer_muon.param_clss_fn2',
    ])

    zero0_ckpt = torch.load(tmp_path / '1' / 'result.pt', weights_only=False)
    zero1_ckpt = torch.load(tmp_path / '2' / 'result.pt', weights_only=False)
    zero2_ckpt = torch.load(tmp_path / '3' / 'result.pt', weights_only=False)

    assert_equal(zero0_ckpt['model'], zero1_ckpt['model'])
    assert_equal(zero0_ckpt['model'], zero2_ckpt['model'])
    assert_equal(zero0_ckpt['optimizer']['state'], zero1_ckpt['optimizer']['state'])
    assert_equal(zero0_ckpt['optimizer']['state'], zero2_ckpt['optimizer']['state'])


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4, reason='lack of gpu devices')
@pytest.mark.parametrize('name', available_optimizers)
def test_trainer_muon_resume_correctness_zero_ngroups(tmp_path, name):
    config_file = 'trainer_args_muon.yaml'
    launch_torchrun(4, trainer_muon_worker, tmp_path, config_file, '1',  [
        '--compute_config.zero_param_level_sharding', True,
        '--compute_config.use_zero', 1,
        '--compute_config.runtime_ngpus', 4,
        '--optimizer.type', get_optimizer_type(name),
    ])
    launch_torchrun(4, trainer_muon_worker, tmp_path, config_file, '2',  [
        '--compute_config.runtime_ngpus', 4,
        '--optimizer.type', get_original_optimizer_type(name),
    ])
    launch_torchrun(4, trainer_muon_worker, tmp_path, config_file, '3', [
        '--compute_config.zero_param_level_sharding', True,
        '--compute_config.use_zero', 1,
        '--compute_config.zero_ngroups', 2,
        '--compute_config.runtime_ngpus', 4,
        '--optimizer.type', get_optimizer_type(name),
    ])

    zero0_ckpt = torch.load(tmp_path / '1' / 'result.pt', weights_only=False)
    zero1_ckpt = torch.load(tmp_path / '2' / 'result.pt', weights_only=False)
    zero2_ckpt = torch.load(tmp_path / '3' / 'result.pt', weights_only=False)

    assert_equal(zero0_ckpt['model'], zero1_ckpt['model'])
    assert_equal(zero0_ckpt['model'], zero2_ckpt['model'])
    assert_equal(zero0_ckpt['optimizer']['state'], zero1_ckpt['optimizer']['state'])
    assert_equal(zero0_ckpt['optimizer']['state'], zero2_ckpt['optimizer']['state'])


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4, reason='lack of gpu devices')
@pytest.mark.parametrize('name', available_optimizers)
def test_trainer_muon_resume_correctness_zero_ngroups_hybrid_param_config(tmp_path, name):
    config_file = 'trainer_args_muon_hybrid.yaml'
    optimizer_options = []
    if name == 'dion':
        # Dion's default Polar Express amplifies tiny rounding differences from
        # different ZeRO parameter layouts. Use Newton-Schulz to test layout equivalence.
        optimizer_options = [
            '--optimizer.args.config.optimizers.1.options.use_polar_express', False,
        ]

    launch_torchrun(4, trainer_muon_worker, tmp_path, config_file, '1', [
        '--compute_config.use_zero', 1,
        '--compute_config.runtime_ngpus', 4,
        '--optimizer.args.config.optimizers.1.type', get_optimizer_type(name),
        '--optimizer.param_clss_fn', 'tests.cli.test_trainer_muon.param_clss_fn2',
        *optimizer_options,
    ])

    launch_torchrun(4, trainer_muon_worker, tmp_path, config_file, '2', [
        '--compute_config.use_zero', 1,
        '--compute_config.zero_ngroups', 2,
        '--compute_config.runtime_ngpus', 4,
        '--optimizer.args.config.optimizers.1.type', get_optimizer_type(name),
        '--optimizer.param_clss_fn', 'tests.cli.test_trainer_muon.param_clss_fn2',
        *optimizer_options,
    ])

    zero0_ckpt = torch.load(tmp_path / '1' / 'result.pt', weights_only=False)
    zero1_ckpt = torch.load(tmp_path / '2' / 'result.pt', weights_only=False)

    atol = 1e-6 # if name == 'dion' else 5e-3
    assert_close(zero0_ckpt['model'], zero1_ckpt['model'], atol=atol)
    assert_close(zero0_ckpt['optimizer']['state'], zero1_ckpt['optimizer']['state'], atol=atol)


def trainer_hybrid_reshard_worker(save_dir, config_file, additional_options, targets):
    """
    Save a checkpoint with the config file's layout, then resume it into each target layout
    twice: resharding the rank files directly (`with_merged=False` falls back to
    `load_from_rank_state_dicts` because the compute config changed) and merging them first.
    Both resumed runs must end with identical model and optimizer states.
    """
    save_dir = Path(save_dir)
    config_path = Path(__file__).with_name(config_file).resolve()
    source_dir = save_dir / 'source'
    trainer = Trainer([
        '-f', config_path,
        '--max_epochs', '1',
        '--gen_savedir', str(source_dir / 'gen'),
        '--checkpoint.save_dir', str(source_dir / 'ckpt'),
        *additional_options,
    ])
    trainer.run()
    torch.distributed.barrier()

    for target, options in targets.items():
        results = {}
        for mode, with_merged in (('reshard', 'false'), ('merged', 'true')):
            work_dir = save_dir / target / mode
            with catch_log(logger) as log:
                trainer = Trainer([
                    '-f', config_path,
                    '--max_epochs', '2',
                    # a new instance name gives the generated module a new Python module name,
                    # otherwise the already imported source module would be reused in this process
                    '--instance_name', target,
                    '--gen_savedir', str(save_dir / target / 'gen'),
                    '--checkpoint.save_dir', str(work_dir),
                    '--checkpoint.resume_from.checkpoint', str(source_dir / 'ckpt' / 'last'),
                    '--checkpoint.resume_from.with_merged', with_merged,
                    *additional_options,
                    *options,
                ])
                trainer.run()
            assert ('each rank assembles its own shard' in log.getvalue()) == (mode == 'reshard')
            torch.distributed.barrier()
            results[mode] = torch.load(work_dir / 'last' / f'{trainer.rank}.ckpt', weights_only=False)
        for key in ('model', 'optimizer'):
            assert_equal(results['reshard'][key], results['merged'][key])


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason='lack of gpu devices')
@pytest.mark.parametrize('name', available_optimizers)
def test_trainer_hybrid_reshard(tmp_path, name):
    # HybridOptimizer with a MixedPrecisionAdamW child and a Muon child, resumed from a
    # data-parallel checkpoint into a batch-partitioned plan and into ZeRO-1 buckets.
    # Muon forbids partitioning its weights on the last two dims, so no weight TP here.
    config_file = 'trainer_args_muon_hybrid.yaml'
    launch_torchrun(2, trainer_hybrid_reshard_worker, tmp_path, config_file, [
        '--optimizer.args.config.optimizers.1.type', get_optimizer_type(name),
    ], {
        'data': ['--compute_config.plan_ngpus', '2', '--pas_policy', 'data'],
        'zero1': ['--compute_config.use_zero', '1', '--compute_config.zero_param_level_sharding', 'True'],
        'adam_tp': ['--compute_config.plan_ngpus', '2', '--pas_policy', 'tests.cli.test_trainer_muon.adam_tp_policy'],
        'adam_tp_zero1': [
            '--compute_config.plan_ngpus', '2', '--pas_policy', 'tests.cli.test_trainer_muon.adam_tp_policy',
            '--compute_config.use_zero', '1', '--compute_config.zero_param_level_sharding', 'True',
        ],
    })


def adam_tp_policy(graph, cfg):
    """
    Partition only the linear weights that `param_clss_fn` assigns to MixedPrecisionAdamW along
    their output dimension; the weights handled by Muon stay replicated, as Muon forbids
    partitioning them on the last two dimensions.
    """
    from nnscaler.policies import get_pas_ops, OpPlan, OpPartition
    for node in get_pas_ops(graph):
        if node.fn == torch.nn.functional.linear and param_clss_fn(node.input(1).name)[0] == 0:
            yield OpPlan(node, partition=OpPartition(input=1, dim=0))
        else:
            yield OpPlan(node)


def param_clss_fn(param_name: str) -> tuple[int, int]:
    """
    Classify a parameter name into an optimizer index and a parameter group index.
    """
    if 'layers.1.' in param_name or 'layers.10.' in param_name or 'layers.2.' in param_name:
        return 0, 0
    else:
        return 1, 0


def param_clss_fn2(param_name: str) -> tuple[int, int]:
    """
    Classify a parameter name into an optimizer index and a parameter group index.
    """
    if 'layers.1.' in param_name or 'layers.10.' in param_name or 'layers.2.' in param_name:
        return 0, 0, {'zero_param_level_sharding': None}
    else:
        return 1, 0, {'zero_param_level_sharding': True}


def param_clss_fn3(param_name: str) -> tuple[int, int]:
    """
    Classify a parameter name into an optimizer index and a parameter group index.
    """
    if 'layers.1.' in param_name:
        # 1 param only, so we need to pad in zero param level sharding
        return 1, 0, {'zero_param_level_sharding': True}
    else:
        return 0, 0, {'zero_param_level_sharding': None}



@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2 or not MuonAvailable, reason='lack of gpu devices')
def test_trainer_muon_resume_correctness_zero1_param_config_less_params(tmp_path):
    config_file = 'trainer_args_muon_hybrid.yaml'

    launch_torchrun(2, trainer_muon_worker, tmp_path, config_file, '2', [
        '--compute_config.use_zero', 1,
        '--optimizer.args.config.optimizers.1.type', 'nnscaler.runtime.muon_optimizer.Muon',
        '--optimizer.param_clss_fn', 'tests.cli.test_trainer_muon.param_clss_fn3',
    ])

    launch_torchrun(2, trainer_muon_worker, tmp_path, config_file, '1', [
        '--optimizer.args.config.optimizers.1.type', 'torch.optim.Muon',
        '--optimizer.param_clss_fn', 'tests.cli.test_trainer_muon.param_clss_fn3',
    ])

    zero0_ckpt = torch.load(tmp_path / '1' / 'result.pt', weights_only=False)
    zero1_ckpt = torch.load(tmp_path / '2' / 'result.pt', weights_only=False)

    assert_equal(zero0_ckpt['model'], zero1_ckpt['model'])
    assert_equal(zero0_ckpt['optimizer']['state'], zero1_ckpt['optimizer']['state'])
