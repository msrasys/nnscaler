from pathlib import Path

import pytest
import torch

from nnscaler.cli.trainer import Trainer

from tests.launch_torchrun import launch_torchrun
from tests.parallel_module.common import assert_equal


def trainer_worker_pipeline(save_dir, config_file):
    save_dir = Path(save_dir)
    config_path = Path(__file__).with_name(config_file).resolve()
    run_name = config_path.stem
    gen_savedir = save_dir / run_name / 'gen'
    ckpt_savedir = save_dir / run_name / 'ckpt'
    instance_name = f'instance_{config_path.stem}'

    trainer = Trainer([
        '-f', config_path,
        '--instance_name', instance_name,
        '--max_epochs', '2',
        '--gen_savedir', str(gen_savedir),
        '--checkpoint.save_dir', str(ckpt_savedir),
    ])
    trainer.run()
    assert trainer.model.use_scheduler
    assert trainer.model.nmicros_per_scheduler_step == 4

    if trainer.rank == 0:
        Trainer.merge_checkpoint(list((ckpt_savedir / 'last').glob('*.ckpt')), save_dir / f'merged_{run_name}.pt')

    torch.distributed.barrier()


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4, reason='lack of gpu devices')
def test_trainer_pipeline(tmp_path):
    launch_torchrun(4, trainer_worker_pipeline, tmp_path, 'trainer_args_pipeline_autodist.yaml')
    launch_torchrun(4, trainer_worker_pipeline, tmp_path, 'trainer_args_pipeline.yaml')

    merged_files = list((tmp_path).glob('merged_*.pt'))
    assert len(merged_files) == 2
    merged_state_dicts = [torch.load(merged_file, weights_only=False) for merged_file in merged_files]

    assert_equal(merged_state_dicts[0]['model'], merged_state_dicts[1]['model'])
    assert_equal(merged_state_dicts[0]['optimizer'], merged_state_dicts[1]['optimizer'])
