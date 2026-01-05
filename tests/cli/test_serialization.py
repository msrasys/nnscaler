import pytest
import torch
from pathlib import Path

from nnscaler.cli.serialization import (
    convert_format, SerializationRunner, register_serialization_runner,
    Checkpointer
)
from nnscaler.cli.trainer import Trainer
from tests.launch_torchrun import launch_torchrun
from tests.parallel_module.common import assert_equal


def test_runner(tmp_path):

    class SplitSerializationRunner(SerializationRunner):
        name: str = 'split'

        def run_load(self, load_func, f, *, device='cpu'):
            model_state_dict = load_func(f, device=device)
            opt_state_dict = load_func(str(f) + '.opt', device=device)
            return {
                'model': model_state_dict,
                'optimizer': opt_state_dict
            }

        def run_save(self, save_func, state_dict, f):
            save_func(state_dict['model'], f)
            save_func(state_dict['optimizer'], str(f) + '.opt')

    register_serialization_runner(SplitSerializationRunner)

    a = torch.randn((2, 2), device='cpu')
    b = torch.randn((2, 3), device='cpu')
    c = torch.randn((4, 4), device='cpu')
    d = torch.randn((3, 3), device='cpu')
    tensors = {
        "model": {
            "embedding": a,
            "attention": b,
        },
        "optimizer": {
            "state": {
                0: {
                    "exp_avg": c,
                    "exp_avg_sq": d,
                }
            }
        }
    }
    checkpointer = Checkpointer()
    checkpointer.save(tensors, tmp_path / "model.ckpt")
    checkpointer.flush()

    convert_format(
        src=str(tmp_path / "model.ckpt"),
        dst=str(tmp_path / "model_split.ckpt"),
        dst_serializer='split',
    )

    assert Path(tmp_path / "model_split.ckpt").exists()
    assert Path(tmp_path / "model_split.ckpt.opt").exists()
    tensor3 = Checkpointer(serializer='split').load(tmp_path / "model_split.ckpt")
    assert_equal(tensors, tensor3)

    checkpointer2 = Checkpointer(serializer=':split')
    tensor2 = checkpointer2.load(tmp_path / "model.ckpt")
    assert_equal(tensors, tensor2)

    checkpointer2.save(tensor2, tmp_path / "model_split2.ckpt")
    checkpointer2.flush()
    assert Path(tmp_path / "model_split2.ckpt").exists()
    assert Path(tmp_path / "model_split2.ckpt.opt").exists()

    tensor4 = Checkpointer(serializer='split').load(tmp_path / "model_split2.ckpt")
    assert_equal(tensors, tensor4)


def trainer_split_serializer_worker(tmp_path, symblink):
    save_dir = Path(tmp_path)
    config_path = str(Path(__file__).with_name('trainer_args.yaml').resolve())
    gen_savedir = save_dir / 'gen'
    ckpt_savedir = save_dir / 'ckpt'

    optimizer_type = 'nnscaler.runtime.f16_optimizer.MixedPrecisionAdam'
    use_zero = 1
    format = 'safetensors'
    rev_format = 'pt' if format == 'safetensors' else 'safetensors'

    def list_ckpt_files(dir):
        return set(dir.glob('**/*.ckpt')) | set(dir.glob('**/*.safetensors'))


    class SplitSerializationRunner(SerializationRunner):
        name: str = 'split'

        def __init__(self, mark=''):
            self.mark = mark

        def run_load(self, load_func, f, *, device='cpu'):
            other_state_dict = load_func(f, device=device)
            opt_state_dict = load_func(str(f) + '.opt', device=device)
            model_state_dict = load_func(str(f) + '.model', device=device)
            return {
                'model': model_state_dict,
                'optimizer': opt_state_dict,
                **other_state_dict
            }

        def run_save(self, save_func, state_dict, f):
            save_func(state_dict['model'], str(f) + '.model')
            save_func(state_dict['optimizer'], str(f) + '.opt')
            other_state_dict = {k: v for k, v in state_dict.items() if k not in ['model', 'optimizer']}
            other_state_dict['mark'] = self.mark
            save_func(other_state_dict, f)

    register_serialization_runner(SplitSerializationRunner)

    # train 4 epcho in one time
    trainer = Trainer([
        '-f', config_path,
        '--precision', 'bf16',
        '--optimizer.type', optimizer_type,
        '--max_epochs', '4',
        '--enable_progress_bar', 'false',
        '--gen_savedir', str(gen_savedir),
        '--compute_config.plan_ngpus', '2',
        '--compute_config.runtime_ngpus', '4',
        '--compute_config.use_zero', str(use_zero),
        '--checkpoint.save_type', 'deduped',
        '--checkpoint.save_dir', str(ckpt_savedir),
        '--checkpoint.resume_from', 'last',
        '--checkpoint.keep_last_n_checkpoints', '10',
        '--checkpoint.format', format,
        '--checkpoint.serializer.name', 'split',
        '--checkpoint.serializer.args.mark', 'hello',
        '--checkpoint.symlink_best_and_last', str(symblink),
    ])
    trainer.run()
    torch.distributed.barrier()

    ckpt_files = list_ckpt_files(ckpt_savedir)
    assert len(ckpt_files)/4 == min(10, trainer.total_train_steps_per_epoch * 4) + 2 # 2 for best/last

    for f in ckpt_files:
        assert trainer.checkpointer.load(f)['mark'] == 'hello'
        assert Path(str(f) + '.opt').exists()
        assert Path(str(f) + '.model').exists()

    torch.distributed.barrier()
    # train 4 epcho two times (resume from last)
    ckpt0_savedir = save_dir / 'ckpt0'
    # first two epochs
    trainer = Trainer([
        '-f', config_path,
        '--precision', 'bf16',
        '--optimizer.type', optimizer_type,
        '--max_epochs', '2',
        '--enable_progress_bar', 'false',
        '--gen_savedir', str(gen_savedir),
        '--compute_config.plan_ngpus', '2',
        '--compute_config.runtime_ngpus', '4',
        '--compute_config.use_zero', str(use_zero),
        '--checkpoint.save_type', 'deduped',
        '--checkpoint.save_dir', str(ckpt0_savedir),
        '--checkpoint.resume_from', 'last',
        '--checkpoint.keep_last_n_checkpoints', '10',
        '--checkpoint.format', format,
        '--checkpoint.serializer', 'split',
        '--checkpoint.symlink_best_and_last', str(symblink),
    ])
    trainer.run()

    torch.distributed.barrier()
    # create merged checkpoint
    ckpt1_savedir = save_dir / 'ckpt1'
    ckpt1_savedir.mkdir(parents=True, exist_ok=True)
    merged_file_name = f'merged{Checkpointer.NAME_MAP[format]}'
    if trainer.rank == 0:
        Trainer.merge_checkpoint(trainer.checkpointer.list_checkpoints(ckpt0_savedir / 'last'), ckpt1_savedir / merged_file_name, serializer='split')
        assert Path(str(ckpt1_savedir / merged_file_name) + '.opt').exists()
        assert Path(str(ckpt1_savedir / merged_file_name) + '.model').exists()

    torch.distributed.barrier()
    # continue with the last two epochs (resume for sharded/deduped checkpoint)
    trainer = Trainer([
        '-f', config_path,
        '--precision', 'bf16',
        '--optimizer.type', optimizer_type,
        '--max_epochs', '4',
        '--enable_progress_bar', 'false',
        '--gen_savedir', str(gen_savedir),
        '--compute_config.plan_ngpus', '2',
        '--compute_config.runtime_ngpus', '4',
        '--compute_config.use_zero', str(use_zero),
        '--checkpoint.save_type', 'deduped',
        '--checkpoint.save_dir', str(ckpt0_savedir),
        '--checkpoint.resume_from', 'last',
        '--checkpoint.format', rev_format,
        '--checkpoint.keep_last_n_checkpoints', '10',
        '--checkpoint.serializer', 'split',
        '--checkpoint.symlink_best_and_last', str(symblink),
    ])
    trainer.run()

    torch.distributed.barrier()

    # continue with the last two epochs (resume for merged)
    trainer = Trainer([
        '-f', config_path,
        '--precision', 'bf16',
        '--optimizer.type', optimizer_type,
        '--max_epochs', '4',
        '--gen_savedir', str(gen_savedir),
        '--compute_config.plan_ngpus', '2',
        '--compute_config.runtime_ngpus', '4',
        '--compute_config.use_zero', str(use_zero),
        '--checkpoint.save_type', 'deduped',
        '--checkpoint.save_dir', str(ckpt1_savedir),
        '--checkpoint.format', rev_format,
        '--checkpoint.resume_from', str(ckpt1_savedir / merged_file_name),
        '--checkpoint.keep_last_n_checkpoints', '10',
        '--checkpoint.serializer', 'split',
        '--checkpoint.symlink_best_and_last', str(symblink),
    ])
    trainer.run()

    torch.distributed.barrier()

    ckpt0_files1 = list_ckpt_files(ckpt0_savedir)

    torch.distributed.barrier()

    if torch.distributed.get_rank() == 0:
        assert {f.parent.name for f in ckpt_files} == {f.parent.name for f in ckpt0_files1}
        for i in range(4):
            x = trainer.checkpointer.load_for_rank(ckpt_savedir / 'last', i)
            y = trainer.checkpointer.load_for_rank(ckpt0_savedir / 'last', i)
            z = trainer.checkpointer.load_for_rank(ckpt1_savedir / 'last', i)
            assert_equal(x['model'], y['model'])
            assert_equal(x['optimizer'], y['optimizer'])
            assert_equal(x['lr_scheduler'], y['lr_scheduler'])
            assert_equal(x['model'], z['model'])
            assert_equal(x['optimizer'], z['optimizer'])
            assert_equal(x['lr_scheduler'], z['lr_scheduler'])


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4, reason='lack of gpu devices')
@pytest.mark.parametrize('symblink', [True, False])
def test_trainer_split_serializer(tmp_path, symblink):
    launch_torchrun(4, trainer_split_serializer_worker, tmp_path, symblink)
