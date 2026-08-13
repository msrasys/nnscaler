from pathlib import Path

import pytest
import torch

from nnscaler.cli.trainer import Trainer
from tests.cli.common import BranchedMixedModule
from tests.launch_torchrun import launch_torchrun
from tests.parallel_module.common import assert_equal


class FirstBranchDataset(torch.utils.data.Dataset):
    def __init__(self, dim: int, size: int = 8):
        generator = torch.Generator().manual_seed(0)
        self.data = torch.randn(size, dim, generator=generator)
        self.target = torch.rand(size, dim, generator=generator)

    def __getitem__(self, idx: int):
        return {
            'data1': self.data[idx],
            'target': self.target[idx],
        }

    def __len__(self):
        return len(self.data)


class SecondBranchDataset(FirstBranchDataset):
    def __getitem__(self, idx: int):
        return {
            'data2': self.data[idx],
            'target': self.target[idx],
        }


class SequentialBranchDataset(torch.utils.data.Dataset):
    def __init__(self, dim: int, size: int = 16):
        if size % 2 != 0:
            raise ValueError(f'size must be even, got {size}')
        generator = torch.Generator().manual_seed(0)
        self.data = torch.randn(size, dim, generator=generator)
        self.target = torch.rand(size, dim, generator=generator)

    def __getitem__(self, idx: int):
        data_key = 'data1' if idx < len(self) // 2 else 'data2'
        return {
            data_key: self.data[idx],
            'target': self.target[idx],
        }

    def __len__(self):
        return len(self.data)


_reducer_none_grad_weight_history = []


def record_reducer_none_grad_weights(trainer: Trainer):
    _reducer_none_grad_weight_history.append({
        'mlp0': trainer.model.mlp0.layers[0].weight.detach().cpu().clone(),
        'mlp1': trainer.model.mlp1.layers[0].weight.detach().cpu().clone(),
    })


def branch_param_clss_fn(param_name: str) -> int:
    if param_name.startswith('mlp0.'):
        return 0
    if param_name.startswith('mlp1.'):
        return 1
    return 2


def shared_branch_param_clss_fn(param_name: str) -> int:
    if param_name.startswith(('mlp0.', 'mlp1.')):
        return 0
    return 1


def mixed1_worker(save_dir, config_file):
    save_dir = Path(save_dir)
    stem = Path(config_file).stem
    config_path = str(Path(__file__).with_name(config_file).resolve())
    gen_savedir = save_dir /  f'gen_{stem}'
    ckpt_savedir = save_dir / f'ckpt_{stem}'

    # ground truth: train 6 epoches in one time with zero 0
    trainer = Trainer([
        '-f', config_path,
        '--max_epochs', '6',
        '--compute_config.use_zero', '0',
        '--enable_progress_bar', False,
        '--gen_savedir', str(gen_savedir),
        '--checkpoint.save_type', 'deduped',
        '--checkpoint.save_dir', str(ckpt_savedir),
        '--checkpoint.resume_from', 'last',
    ])
    trainer.run()
    torch.distributed.barrier()

    # train 6 epoches in 6 times, each time resume from last checkpoint
    # 1
    ckpt0_savedir = save_dir / f'ckpt0_{stem}'
    gen0_savedir = save_dir / f'gen0_{stem}'  # use a different gen_savedir for resumable dataloader
    trainer = Trainer([
        '-f', config_path,
        '--max_epochs', '1',
        '--enable_progress_bar', 'false',
        '--gen_savedir', str(gen0_savedir),
        '--checkpoint.save_type', 'deduped',
        '--checkpoint.save_dir', str(ckpt0_savedir),
        '--checkpoint.resume_from', 'last',
    ])
    trainer.run()

    torch.distributed.barrier()
    # 2
    trainer = Trainer([
        '-f', config_path,
        '--max_epochs', '2',
        '--enable_progress_bar', 'false',
        '--gen_savedir', str(gen0_savedir),
        '--checkpoint.save_type', 'sharded',
        '--checkpoint.save_dir', str(ckpt0_savedir),
        '--checkpoint.resume_from', 'last',
    ])
    trainer.run()

    torch.distributed.barrier()
    if trainer.rank == 0:
        Trainer.merge_checkpoint(list((ckpt0_savedir / 'last').glob('*.ckpt')), ckpt0_savedir / 'merged2.pt')

    torch.distributed.barrier()
    # 3
    trainer = Trainer([
        '-f', config_path,
        '--max_epochs', '3',
        '--enable_progress_bar', 'false',
        '--gen_savedir', str(gen0_savedir),
        '--checkpoint.save_type', 'deduped',
        '--checkpoint.save_dir', str(ckpt0_savedir),
        '--checkpoint.resume_from.checkpoint', str(ckpt0_savedir / 'merged2.pt'),
        '--checkpoint.resume_from.save_memory', False,
    ])
    trainer.run()

    torch.distributed.barrier()
    if trainer.rank == 0:
        Trainer.merge_checkpoint(list((ckpt0_savedir / 'last').glob('*.ckpt')), ckpt0_savedir / 'merged3.pt')

    torch.distributed.barrier()
    # 4
    trainer = Trainer([
        '-f', config_path,
        '--max_epochs', '4',
        '--enable_progress_bar', 'false',
        '--gen_savedir', str(gen0_savedir),
        '--checkpoint.save_type', 'deduped',
        '--checkpoint.save_dir', str(ckpt0_savedir),
        '--checkpoint.resume_from.checkpoint', str(ckpt0_savedir / 'merged3.pt'),
        '--checkpoint.resume_from.save_memory', True,
    ])
    trainer.run()

    torch.distributed.barrier()
    # 5
    trainer = Trainer([
        '-f', config_path,
        '--max_epochs', '5',
        '--enable_progress_bar', 'false',
        '--gen_savedir', str(gen0_savedir),
        '--checkpoint.save_type', 'deduped',
        '--checkpoint.save_dir', str(ckpt0_savedir),
        '--checkpoint.resume_from.checkpoint', 'last',
        '--checkpoint.resume_from.save_memory', True,
        '--checkpoint.resume_from.slow_fs', True,
    ])
    trainer.run()

    torch.distributed.barrier()
    # 6
    trainer = Trainer([
        '-f', config_path,
        '--max_epochs', '6',
        '--enable_progress_bar', 'false',
        '--gen_savedir', str(gen0_savedir),
        '--checkpoint.save_type', 'deduped',
        '--checkpoint.save_dir', str(ckpt0_savedir),
        '--checkpoint.resume_from.checkpoint', 'last',
        '--checkpoint.resume_from.save_memory', False,
    ])
    trainer.run()

    torch.distributed.barrier()

    if torch.distributed.get_rank() == 0:
        Trainer.merge_checkpoint(list((ckpt0_savedir / 'last').glob('*.ckpt')), ckpt0_savedir / 'merged.pt')
        Trainer.merge_checkpoint(list((ckpt_savedir / 'last').glob('*.ckpt')), ckpt_savedir / 'merged.pt')

        merged1 = torch.load(ckpt0_savedir / 'merged.pt', weights_only=False)
        merged2 = torch.load(ckpt_savedir / 'merged.pt', weights_only=False)
        assert_equal(merged1['model'], merged2['model'])
        assert_equal(merged1['optimizer'], merged2['optimizer'])


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4, reason='lack of gpu devices')
@pytest.mark.parametrize('config_file', [
    'trainer_args_mixed1.yaml',
    'trainer_args_mixed2.yaml',
    'trainer_args_mixed3.yaml',
    'trainer_args_mixed3_async.yaml',
])
def test_mixed1(tmp_path, config_file):
    launch_torchrun(4, mixed1_worker, tmp_path, config_file)


def conditional_parallel_module_worker(save_dir, include_inactive_parallel_module):
    save_dir = Path(save_dir)
    run_name = 'with_inactive' if include_inactive_parallel_module else 'active_only'
    run_dir = save_dir / run_name
    config_path = str(Path(__file__).with_name('trainer_args_mixed3.yaml').resolve())
    module_types = ['tests.cli.common.MixModuleMLP3']
    if include_inactive_parallel_module:
        module_types.insert(0, 'tests.cli.common.MixModuleMLP2')
    parallel_module_args = []
    for index, module_type in enumerate(module_types):
        prefix = f'--model.parallel_modules.{index}'
        parallel_module_args.extend([
            f'{prefix}.type', module_type,
            f'{prefix}.args.dim', '16',
            f'{prefix}.args.nlayers', '16',
            f'{prefix}.forward_args_gen_fn', 'tests.cli.common.forward_args_gen_fn',
        ])

    trainer = Trainer([
        '-f', config_path,
        '--compute_config.plan_ngpus', '1',
        '--compute_config.runtime_ngpus', '2',
        '--max_epochs', '1',
        '--max_train_steps', '1',
        '--dataset.type', 'tests.cli.test_mixed_module.FirstBranchDataset',
        '--dataset.train_args.size', '8',
        '--dataset.val_args.size', '4',
        '--enable_progress_bar', 'false',
        '--gen_savedir', str(run_dir / 'gen'),
        '--checkpoint.save_type', 'sharded',
        '--checkpoint.save_dir', str(run_dir / 'ckpt'),
        *parallel_module_args,
    ])
    trainer.run()
    torch.distributed.barrier()
    if trainer.rank == 0:
        Trainer.merge_checkpoint(
            list((run_dir / 'ckpt' / 'last').glob('*.ckpt')),
            run_dir / 'merged.pt',
        )
    torch.distributed.barrier()

    trainer = Trainer([
        '-f', config_path,
        '--compute_config.plan_ngpus', '1',
        '--compute_config.runtime_ngpus', '2',
        '--max_epochs', '2',
        '--max_train_steps', '2',
        '--dataset.type', 'tests.cli.test_mixed_module.FirstBranchDataset',
        '--dataset.train_args.size', '8',
        '--dataset.val_args.size', '4',
        '--enable_progress_bar', 'false',
        '--gen_savedir', str(run_dir / 'gen'),
        '--checkpoint.save_type', 'deduped',
        '--checkpoint.save_dir', str(run_dir / 'resumed_ckpt'),
        '--checkpoint.resume_from.checkpoint', str(run_dir / 'merged.pt'),
        *parallel_module_args,
    ])
    trainer.run()
    torch.distributed.barrier()

    trainer = Trainer([
        '-f', config_path,
        '--compute_config.plan_ngpus', '1',
        '--compute_config.runtime_ngpus', '2',
        '--max_epochs', '3',
        '--max_train_steps', '3',
        '--dataset.type', 'tests.cli.test_mixed_module.FirstBranchDataset',
        '--dataset.train_args.size', '8',
        '--dataset.val_args.size', '4',
        '--enable_progress_bar', 'false',
        '--gen_savedir', str(run_dir / 'gen'),
        '--checkpoint.save_type', 'sharded',
        '--checkpoint.save_dir', str(run_dir / 'resumed_ckpt'),
        '--checkpoint.resume_from.checkpoint', 'last',
        *parallel_module_args,
    ])
    trainer.run()
    torch.distributed.barrier()
    if trainer.rank == 0:
        Trainer.merge_checkpoint(
            list((run_dir / 'resumed_ckpt' / 'last').glob('*.ckpt')),
            run_dir / 'result.pt',
        )
    torch.distributed.barrier()


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason='lack of gpu devices')
def test_conditional_parallel_module_sync(tmp_path):
    launch_torchrun(2, conditional_parallel_module_worker, tmp_path, False)
    launch_torchrun(2, conditional_parallel_module_worker, tmp_path, True)

    active_only = torch.load(tmp_path / 'active_only' / 'result.pt', weights_only=False)
    with_inactive = torch.load(tmp_path / 'with_inactive' / 'result.pt', weights_only=False)
    assert_equal(active_only['model'], with_inactive['model'])
    assert_equal(
        active_only['optimizer']['param_groups'],
        with_inactive['optimizer']['param_groups'],
    )

    # inactive states are saved in `active_only` checkpoints,
    # because they are part of non-parallel module continuous buffer.
    # but they are not present in `with_inactive` checkpoints,
    # because they are part of inactive parallel module continuous buffer.
    original_module = BranchedMixedModule(dim=16, nlayers=16)
    inactive_state_indices = {
        index for index, (name, _) in enumerate(original_module.named_parameters())
        if name.startswith('mlp1.')
    }
    active_only_states = active_only['optimizer']['state']
    with_inactive_states = with_inactive['optimizer']['state']
    assert set(with_inactive_states) == set(active_only_states) - inactive_state_indices
    for index, state in with_inactive_states.items():
        assert_equal(state, active_only_states[index])


def conditional_parallel_module_switch_worker(save_dir, resume_from_merged):
    save_dir = Path(save_dir)
    run_name = 'merged_resume' if resume_from_merged else 'sharded_resume'
    run_dir = save_dir / run_name
    config_path = str(Path(__file__).with_name('trainer_args_mixed3.yaml').resolve())
    parallel_module_args = []
    for index, module_type in enumerate([
        'tests.cli.common.MixModuleMLP2',
        'tests.cli.common.MixModuleMLP3',
    ]):
        prefix = f'--model.parallel_modules.{index}'
        parallel_module_args.extend([
            f'{prefix}.type', module_type,
            f'{prefix}.args.dim', '16',
            f'{prefix}.args.nlayers', '16',
            f'{prefix}.forward_args_gen_fn', 'tests.cli.common.forward_args_gen_fn',
        ])

    common_args = [
        '-f', config_path,
        '--compute_config.plan_ngpus', '1',
        '--compute_config.runtime_ngpus', '2',
        '--dataset.train_args.size', '8',
        '--dataset.val_args.size', '4',
        '--enable_progress_bar', 'false',
        '--gen_savedir', str(run_dir / 'gen'),
        '--checkpoint.save_type', 'sharded',
        '--checkpoint.save_dir', str(run_dir / 'ckpt'),
        *parallel_module_args,
    ]

    trainer = Trainer([
        *common_args,
        '--max_epochs', '1',
        '--max_train_steps', '1',
        '--dataset.type', 'tests.cli.test_mixed_module.FirstBranchDataset',
    ])
    trainer.run()
    torch.distributed.barrier()
    if trainer.rank == 0:
        Trainer.merge_checkpoint(
            list((run_dir / 'ckpt' / 'last').glob('*.ckpt')),
            run_dir / 'branch1.pt',
        )
    torch.distributed.barrier()

    trainer = Trainer([
        *common_args,
        '--max_epochs', '2',
        '--max_train_steps', '2',
        '--dataset.type', 'tests.cli.test_mixed_module.SecondBranchDataset',
        '--checkpoint.resume_from.checkpoint',
        str(run_dir / 'branch1.pt') if resume_from_merged else 'last',
    ])
    trainer.run()
    torch.distributed.barrier()
    if trainer.rank == 0:
        Trainer.merge_checkpoint(
            list((run_dir / 'ckpt' / 'last').glob('*.ckpt')),
            run_dir / 'result.pt',
        )
    torch.distributed.barrier()


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason='lack of gpu devices')
def test_conditional_parallel_module_switch_branches(tmp_path):
    launch_torchrun(2, conditional_parallel_module_switch_worker, tmp_path, False)
    launch_torchrun(2, conditional_parallel_module_switch_worker, tmp_path, True)

    sharded_resume = torch.load(tmp_path / 'sharded_resume' / 'result.pt', weights_only=False)
    merged_resume = torch.load(tmp_path / 'merged_resume' / 'result.pt', weights_only=False)
    assert_equal(sharded_resume['model'], merged_resume['model'])
    assert_equal(sharded_resume['optimizer'], merged_resume['optimizer'])

    branch1 = torch.load(tmp_path / 'merged_resume' / 'branch1.pt', weights_only=False)
    original_module = BranchedMixedModule(dim=16, nlayers=16)
    branch2_state_indices = {
        index for index, (name, _) in enumerate(original_module.named_parameters())
        if name.startswith('mlp1.')
    }
    assert branch2_state_indices.isdisjoint(branch1['optimizer']['state'])
    assert branch2_state_indices.issubset(merged_resume['optimizer']['state'])


def reducer_none_grad_worker(save_dir, use_none_grad):
    _reducer_none_grad_weight_history.clear()
    save_dir = Path(save_dir) / str(use_none_grad)
    config_path = str(Path(__file__).with_name('trainer_args_mixed3.yaml').resolve())
    trainer = Trainer([
        '-f', config_path,
        '--compute_config.plan_ngpus', '1',
        '--compute_config.runtime_ngpus', '2',
        '--max_epochs', '1',
        '--max_train_steps', '2',
        '--max_val_steps', '1',
        '--dataset.type', 'tests.cli.test_mixed_module.SequentialBranchDataset',
        '--dataset.train_args.size', '16',
        '--dataset.val_args.size', '8',
        '--enable_progress_bar', 'false',
        '--gen_savedir', str(save_dir / 'gen'),
        '--checkpoint.no_save', 'true',
        '--model.non_parallel_params_reducer_config.reducer_none_grad', str(use_none_grad),
        '--optimizer.param_clss_fn', 'tests.cli.test_mixed_module.branch_param_clss_fn',
        '--hook.after_optimizer_step',
        'tests.cli.test_mixed_module.record_reducer_none_grad_weights',
    ])
    trainer.run()

    reducer = trainer.optimizer._non_parallel_module_reducer
    assert reducer._use_none_grad is use_none_grad
    assert {bucket.param_cls[0] for bucket in reducer.buckets} == {0, 1, 2}
    assert len(_reducer_none_grad_weight_history) == 2
    return _reducer_none_grad_weight_history


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason='lack of gpu devices')
def test_reducer_none_grad_end2end(tmp_path):
    zero_grad_history = launch_torchrun(2, reducer_none_grad_worker, tmp_path, False)
    none_grad_history = launch_torchrun(2, reducer_none_grad_worker, tmp_path, True)

    for rank in range(2):
        assert_equal(zero_grad_history[rank][0], none_grad_history[rank][0])

        # mlp0 is active in the first step and inactive in the second step.
        assert not torch.equal(
            zero_grad_history[rank][0]['mlp0'],
            zero_grad_history[rank][1]['mlp0'],
        )
        assert torch.equal(
            none_grad_history[rank][0]['mlp0'],
            none_grad_history[rank][1]['mlp0'],
        )

        # mlp1 takes the opposite branch and must be updated in the second step.
        assert not torch.equal(
            none_grad_history[rank][0]['mlp1'],
            none_grad_history[rank][1]['mlp1'],
        )

    assert_equal(zero_grad_history[0], zero_grad_history[1])
    assert_equal(none_grad_history[0], none_grad_history[1])


def reducer_none_grad_partial_bucket_worker(save_dir):
    config_path = str(Path(__file__).with_name('trainer_args_mixed3.yaml').resolve())
    trainer = Trainer([
        '-f', config_path,
        '--compute_config.plan_ngpus', '1',
        '--compute_config.runtime_ngpus', '2',
        '--max_epochs', '1',
        '--max_train_steps', '1',
        '--dataset.type', 'tests.cli.test_mixed_module.FirstBranchDataset',
        '--dataset.train_args.size', '8',
        '--dataset.val_args.size', '4',
        '--enable_progress_bar', 'false',
        '--gen_savedir', str(Path(save_dir) / 'gen'),
        '--checkpoint.no_save', 'true',
        '--model.non_parallel_params_reducer_config.reducer_none_grad', 'true',
        '--optimizer.param_clss_fn',
        'tests.cli.test_mixed_module.shared_branch_param_clss_fn',
    ])

    with pytest.raises(RuntimeError, match='all parameters in the same bucket'):
        trainer.run()


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason='lack of gpu devices')
def test_reducer_none_grad_partial_bucket_raises(tmp_path):
    launch_torchrun(2, reducer_none_grad_partial_bucket_worker, tmp_path)


def reducer_none_grad_checkpoint_worker(save_dir, resume_from_merged):
    run_name = 'merged_resume' if resume_from_merged else 'sharded_resume'
    run_dir = Path(save_dir) / run_name
    config_path = str(Path(__file__).with_name('trainer_args_mixed3.yaml').resolve())
    common_args = [
        '-f', config_path,
        '--compute_config.plan_ngpus', '1',
        '--compute_config.runtime_ngpus', '2',
        '--dataset.train_args.size', '8',
        '--dataset.val_args.size', '4',
        '--enable_progress_bar', 'false',
        '--gen_savedir', str(run_dir / 'gen'),
        '--checkpoint.save_type', 'sharded',
        '--checkpoint.save_dir', str(run_dir / 'ckpt'),
        '--model.non_parallel_params_reducer_config.reducer_none_grad', 'true',
        '--optimizer.param_clss_fn', 'tests.cli.test_mixed_module.branch_param_clss_fn',
    ]

    trainer = Trainer([
        *common_args,
        '--max_epochs', '1',
        '--max_train_steps', '1',
        '--dataset.type', 'tests.cli.test_mixed_module.FirstBranchDataset',
    ])
    trainer.run()
    torch.distributed.barrier()
    if trainer.rank == 0:
        Trainer.merge_checkpoint(
            list((run_dir / 'ckpt' / 'last').glob('*.ckpt')),
            run_dir / 'branch1.pt',
        )
    torch.distributed.barrier()

    trainer = Trainer([
        *common_args,
        '--max_epochs', '2',
        '--max_train_steps', '2',
        '--dataset.type', 'tests.cli.test_mixed_module.SecondBranchDataset',
        '--checkpoint.resume_from.checkpoint',
        str(run_dir / 'branch1.pt') if resume_from_merged else 'last',
    ])
    trainer.run()
    torch.distributed.barrier()
    if trainer.rank == 0:
        Trainer.merge_checkpoint(
            list((run_dir / 'ckpt' / 'last').glob('*.ckpt')),
            run_dir / 'result.pt',
        )
    torch.distributed.barrier()


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason='lack of gpu devices')
def test_reducer_none_grad_checkpoint_roundtrip(tmp_path):
    launch_torchrun(2, reducer_none_grad_checkpoint_worker, tmp_path, False)
    launch_torchrun(2, reducer_none_grad_checkpoint_worker, tmp_path, True)

    sharded_resume = torch.load(tmp_path / 'sharded_resume' / 'result.pt', weights_only=False)
    merged_resume = torch.load(tmp_path / 'merged_resume' / 'result.pt', weights_only=False)
    assert_equal(sharded_resume['model'], merged_resume['model'])
    assert_equal(sharded_resume['optimizer'], merged_resume['optimizer'])

    branch1 = torch.load(tmp_path / 'merged_resume' / 'branch1.pt', weights_only=False)
    original_module = BranchedMixedModule(dim=16, nlayers=16)
    branch2_state_indices = {
        index for index, (name, _) in enumerate(original_module.named_parameters())
        if name.startswith('mlp1.')
    }
    assert branch2_state_indices.isdisjoint(branch1['optimizer']['state'])
    assert branch2_state_indices.issubset(merged_resume['optimizer']['state'])
