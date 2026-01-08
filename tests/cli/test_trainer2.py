from pathlib import Path
import pytest
import torch
from torch.utils.data import  Dataset

from nnscaler.cli import TrainerArgs, Trainer
from tests.launch_torchrun import launch_torchrun
from tests.parallel_module.test_gencode import _gencode_contains, print_gencode


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
