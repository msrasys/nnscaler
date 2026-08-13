from pathlib import Path
import pytest
import torch
from torch.utils.data import  Dataset

from nnscaler.cli import TrainerArgs, Trainer
from tests.launch_torchrun import launch_torchrun
from tests.parallel_module.test_gencode import _gencode_contains, print_gencode
from tests.utils import catch_log


class NanoGptDataset(Dataset):
    def __init__(self, *args, **kwargs):
        pass

    def __getitems__(self, indices):
        return [torch.randint(0, 151936, (1, 4096), dtype=torch.int64) for _ in indices]

    def __len__(self):
        return 10000


def gen_args(trainer_args: 'TrainerArgs'):
    src_token = torch.randint(0, 151936, (1, 4096), dtype=torch.int64)
    ret = dict(
        input_ids=src_token, # torch.Size([1, 4096]) torch.int64
    )
    return ret


class WrappedSubModel(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.embedding = torch.nn.Embedding(151936, 1536)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        return x


class WrapperModel(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.model = WrappedSubModel()

    def forward(self, src_tokens):
        # the logic is from task.train_step
        logits = self.model(
            src_tokens
        )
        return torch.sum(logits)


def trainer_mixed_worker(save_dir):
    save_dir = Path(save_dir)
    config_path = str(Path(__file__).with_name('trainer_args_mixed_bf16.yaml').resolve())
    gen_savedir = save_dir / 'gen'
    ckpt_savedir = save_dir / 'ckpt'

    args = TrainerArgs.from_cli([
        '-f', config_path,
        '--gen_savedir', str(gen_savedir),
        '--checkpoint.save_dir', str(ckpt_savedir),
    ])
    trainer = Trainer(train_args=args)
    trainer.run()
    # should reach here without error
    assert True


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason='lack of gpu devices')
def test_mixed_bf16_model(tmp_path):
    launch_torchrun(2, trainer_mixed_worker, tmp_path)


class SharedWeightsDataset(Dataset):
    def __init__(self, *args, **kwargs):
        pass

    def __getitems__(self, indices):
        return [torch.randn(4, 4) for _ in indices]

    def __len__(self):
        return 10000


class SharedWeightsModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4, bias=False)
        self.linear2 = torch.nn.Linear(4, 4, bias=False)
        self.linear2.weight = self.linear.weight  # share weight

    def forward(self, x):
        y =  x * 2
        z =  x + 2
        r = self.linear2(y)
        r = r + self.linear(z)
        return torch.sum(r)


def trainer_zero3_shared_weights_worker(save_dir):
    save_dir = Path(save_dir)
    config_path = str(Path(__file__).with_name('trainer_args_shared_weights.yaml').resolve())
    gen_savedir = save_dir / 'gen'
    ckpt_savedir = save_dir / 'ckpt'

    args = TrainerArgs.from_cli([
        '-f', config_path,
        '--gen_savedir', str(gen_savedir),
        '--checkpoint.save_dir', str(ckpt_savedir),
    ])
    trainer = Trainer(train_args=args)
    trainer.run()
    # weight sharing multiref should have clone_level=1 in gencode
    assert _gencode_contains(
        gen_savedir,
        SharedWeightsModule,
        torch.distributed.get_rank(),
        r'linear_weight_\d+, linear_weight_\d+ = nnscaler.runtime.function.multiref\(self.linear_weight_\d+, times=2, clone_level=1\)'
    )
    # non-weight tensor multiref should not have clone_level
    assert _gencode_contains(
        gen_savedir,
        SharedWeightsModule,
        torch.distributed.get_rank(),
        r'x_\d+, x_\d+ = nnscaler.runtime.function.multiref\(x_\d+, times=2\)'
    )


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4, reason='lack of gpu devices')
def test_zero3_shared_weights(tmp_path):
    launch_torchrun(4, trainer_zero3_shared_weights_worker, tmp_path)


def trainer_npbuffer_worker(save_dir):
    """Test that first training loads full weights, resume loads only npbuffer.pt."""
    save_dir = Path(save_dir)
    config_path = str(Path(__file__).with_name('trainer_args.yaml').resolve())
    gen_savedir = save_dir / 'gen'
    ckpt_savedir = save_dir / 'ckpt'

    from nnscaler.runtime.module import _logger as module_logger

    common_args = [
        '-f', config_path,
        '--model.type', 'tests.cli.common.MLPWithNPBuffer',
        '--enable_progress_bar', 'false',
        '--gen_savedir', str(gen_savedir),
        '--compute_config.plan_ngpus', '1',
        '--compute_config.runtime_ngpus', '1',
        '--compute_config.use_zero', '0',
        '--checkpoint.save_type', 'deduped',
        '--checkpoint.save_dir', str(ckpt_savedir),
        '--checkpoint.keep_last_n_checkpoints', '30',
        '--checkpoint.every_n_train_steps', '1',
        '--micro_batch_size', '2',
        '--global_batch_size', '2',
    ]

    # Step 1: First training - should load full weights (load_attr_content)
    with catch_log(module_logger, 'INFO') as log_stream:
        trainer1 = Trainer([
            *common_args,
            '--max_train_steps', '2',
        ])
        trainer1.run()
        logs = log_stream.getvalue()
        assert 'loading partitioned model from' in logs, \
            "First training should load full weights from fullmodel.pt"
        assert 'loading non-persistent buffers from' not in logs, \
            "First training should NOT use npbuffer.pt path"

    # Step 2: Resume - should load only npbuffer.pt (not full weights)
    with catch_log(module_logger, 'INFO') as log_stream:
        trainer2 = Trainer([
            *common_args,
            '--max_train_steps', '4',
            '--checkpoint.resume_from', 'last',
        ])
        trainer2.run()
        logs = log_stream.getvalue()
        assert 'loading non-persistent buffers from' in logs, \
            "Resume should load non-persistent buffers from npbuffer.pt"
        assert 'loading partitioned model from' not in logs, \
            "Resume should NOT load full weights from fullmodel.pt"

    # Step 3: Verify training actually progressed
    assert trainer2.train_status.finished_train_steps == 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason='lack of gpu devices')
def test_trainer_npbuffer(tmp_path):
    """Test npbuffer.pt loading optimization for non-persistent buffers during resume."""
    launch_torchrun(1, trainer_npbuffer_worker, tmp_path)
