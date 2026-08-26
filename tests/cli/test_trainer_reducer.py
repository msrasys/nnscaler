import torch
import pytest
from pathlib import Path

from nnscaler.cli.trainer import Trainer

from tests.launch_torchrun import launch_torchrun
from tests.parallel_module.common import assert_close


def trainer_worker_reducer(save_dir, config_file, run_name=None, additional_args=None):
    save_dir = Path(save_dir)
    config_path = Path(__file__).with_name(config_file).resolve()
    run_name = run_name or config_path.stem
    gen_savedir = save_dir / run_name / 'gen'
    ckpt_savedir = save_dir / run_name / 'ckpt'
    instance_name = f'instance_{run_name}'

    additional_args = additional_args or []

    trainer = Trainer([
        '-f', config_path,
        '--instance_name', instance_name,
        '--max_epochs', '2',
        '--gen_savedir', str(gen_savedir),
        '--checkpoint.save_dir', str(ckpt_savedir),
        *additional_args
    ])
    trainer.run()
    if trainer.train_args.compute_config.reducer_replicated_params:
        assert len(trainer.model.reducers[0].ranks) == trainer.world_size
        assert trainer.model.reducers[0]._nreplicas == trainer.train_args.compute_config.plan_ngpus
    else:
        assert len(trainer.model.reducers[0].ranks) == trainer.world_size // trainer.train_args.compute_config.plan_ngpus
        assert trainer.model.reducers[0]._nreplicas == 1

    if trainer.rank == 0:
        Trainer.merge_checkpoint(list((ckpt_savedir / 'last').glob('*.ckpt')), save_dir / f'merged_{run_name}.pt')

    torch.distributed.barrier()



def replicas_reducer_no_partition_pas(graph, cfg):
    from nnscaler.policies import OpPlan, OpPartition, get_pas_ops
    return []


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4, reason='lack of gpu devices')
def test_trainer_reducer_replicas_no_partition(tmp_path):
    launch_torchrun(4, trainer_worker_reducer, tmp_path,
        'trainer_args_reducer.yaml',
        'replicas',
        ['--compute_config.reducer_replicated_params', True]
    )
    launch_torchrun(4, trainer_worker_reducer, tmp_path,
        'trainer_args_reducer.yaml',
        'no_replicas',
        ['--compute_config.reducer_replicated_params', False]
    )
    merged_files = list((tmp_path).glob('merged_*.pt'))
    assert len(merged_files) == 2
    state_dict0 = torch.load(merged_files[0], weights_only=False)
    state_dict1 = torch.load(merged_files[1], weights_only=False)

    assert_close(state_dict0['model'], state_dict1['model'])
    assert_close(state_dict0['optimizer'], state_dict1['optimizer'])
