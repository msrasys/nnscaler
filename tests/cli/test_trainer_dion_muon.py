#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from pathlib import Path

import pytest
import torch
import torch.distributed
from torch import nn
from torch.utils.data import Dataset

from nnscaler.cli.trainer import Trainer
from tests.launch_torchrun import launch_torchrun

try:
    import dion  # noqa: F401
except ImportError:
    pytest.skip('Dion Muon not available', allow_module_level=True)


class DionMuonTestModel(nn.Module):

    def __init__(self, dim, nlayers):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(dim, dim, bias=False) for _ in range(nlayers)])
        self.loss_fn = nn.BCELoss()

    def forward(self, data):
        output = data['data']
        for layer in self.layers:
            output = layer(output)
        return self.loss_fn(torch.sigmoid(output), data['target'])


class DionMuonTestDataset(Dataset):

    def __init__(self, dim, size):
        generator = torch.Generator().manual_seed(0)
        self.data = torch.randn(size, dim, generator=generator)
        self.target = torch.rand(size, dim, generator=generator)

    def __getitem__(self, index):
        return {
            'data': self.data[index],
            'target': self.target[index],
        }

    def __len__(self):
        return len(self.data)


def _assert_checkpoint_dtypes(checkpoint):
    model_tensors = [
        value for value in checkpoint['model'].values()
        if torch.is_tensor(value) and value.is_floating_point()
    ]
    assert model_tensors
    assert all(tensor.dtype == torch.bfloat16 for tensor in model_tensors)

    optimizer_state = checkpoint['optimizer']['state']
    assert optimizer_state
    optimizer_tensors = [(key, value)
                         for param_state in optimizer_state.values()
                         for key, value in param_state.items()
                         if key in {'momentum', 'fp32_params'}]
    assert {key
            for key, _ in optimizer_tensors} == {
                'momentum',
                'fp32_params',
            }
    assert all(tensor.dtype == torch.float32
               for _, tensor in optimizer_tensors)


def _trainer_checkpoint_worker(save_dir, runtime_ngpus, nlayers):
    save_dir = Path(save_dir)
    config_path = Path(__file__).with_name(
        'trainer_args_dion_muon.yaml').resolve()
    checkpoint_dir = save_dir / 'checkpoints'
    gen_savedir = save_dir / 'generated'

    trainer = Trainer([
        '-f',
        str(config_path),
        '--gen_savedir',
        str(gen_savedir),
        '--checkpoint.save_dir',
        str(checkpoint_dir),
        '--compute_config.runtime_ngpus',
        str(runtime_ngpus),
        '--global_batch_size',
        str(2 * runtime_ngpus),
        '--model.args.nlayers',
        str(nlayers),
    ])
    trainer.run()
    torch.distributed.barrier()

    if trainer.rank == 0:
        dumped = torch.load(
            checkpoint_dir / 'last' / '0.ckpt',
            map_location='cpu',
            weights_only=False,
        )
        _assert_checkpoint_dtypes(dumped)
        Trainer.merge_checkpoint(
            list((checkpoint_dir / 'last').glob('*.ckpt')),
            checkpoint_dir / 'merged.pt',
        )
        merged = torch.load(
            checkpoint_dir / 'merged.pt',
            map_location='cpu',
            weights_only=False,
        )
        _assert_checkpoint_dtypes(merged)

    torch.distributed.barrier()
    resumed = Trainer([
        '-f',
        str(config_path),
        '--max_train_steps',
        '4',
        '--gen_savedir',
        str(gen_savedir),
        '--checkpoint.save_dir',
        str(checkpoint_dir),
        '--checkpoint.resume_from',
        str(checkpoint_dir / 'merged.pt'),
        '--compute_config.runtime_ngpus',
        str(runtime_ngpus),
        '--global_batch_size',
        str(2 * runtime_ngpus),
        '--model.args.nlayers',
        str(nlayers),
    ])
    resumed.run()
    torch.distributed.barrier()

    if resumed.rank == 0:
        resumed_checkpoint = torch.load(
            checkpoint_dir / 'last' / '0.ckpt',
            map_location='cpu',
            weights_only=False,
        )
        _assert_checkpoint_dtypes(resumed_checkpoint)

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


@pytest.mark.parametrize('runtime_ngpus, nlayers', [(1, 4), (2, 1)])
def test_trainer_dumps_and_resumes_fp32_dion_state(
    tmp_path,
    runtime_ngpus,
    nlayers,
):
    if torch.cuda.device_count() < runtime_ngpus:
        pytest.skip(f'{runtime_ngpus} CUDA devices required')
    launch_torchrun(
        runtime_ngpus,
        _trainer_checkpoint_worker,
        tmp_path,
        runtime_ngpus,
        nlayers,
    )
