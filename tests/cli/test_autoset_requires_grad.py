#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from pathlib import Path
from unittest.mock import patch

import pytest

from nnscaler.cli.trainer import Trainer
from tests.utils import replace_all_device_with


@replace_all_device_with('cpu')
def test_end2end(tmp_path):
    save_dir = Path(tmp_path)
    config_path = str(Path(__file__).with_name('trainer_args.yaml').resolve())
    gen_savedir = save_dir / 'gen'
    # compile only
    trainer = Trainer([
        '-f', config_path,
        '--max_epochs', '2',
        '--gen_savedir', str(gen_savedir),
        '--compute_config.plan_ngpus', '2',
        '--compute_config.runtime_ngpus', '4',
        '--checkpoint.no_save', 'true',
        '--run_mode', 'compile',
        '--broadcast_strategy', 'none',
    ])
    with patch('nnscaler.cli.mixed_module.nnscaler.parallelize') as mocked_parallelize:
        trainer.run()
        assert mocked_parallelize.call_args.kwargs['autoset_requires_grad'] is True


@replace_all_device_with('cpu')
@pytest.mark.parametrize('autoset_requires_grad', [True, False])
def test_not_end2end(tmp_path, autoset_requires_grad):
    save_dir = Path(tmp_path)
    config_path = str(Path(__file__).with_name('trainer_args_mixed1.yaml').resolve())
    gen_savedir = save_dir / 'gen'
    # compile only
    trainer = Trainer([
        '-f', config_path,
        '--max_epochs', '2',
        '--gen_savedir', str(gen_savedir),
        '--compute_config.plan_ngpus', '2',
        '--compute_config.runtime_ngpus', '4',
        '--checkpoint.no_save', 'true',
        '--run_mode', 'compile',
        '--broadcast_strategy', 'none',
        '--model.parallel_modules.0.forward_args_autoset_requires_grad', str(autoset_requires_grad).lower(),
    ])
    with patch('nnscaler.cli.mixed_module.nnscaler.parallelize') as mocked_parallelize:
        trainer.run()
        assert mocked_parallelize.call_args.kwargs['autoset_requires_grad'] is autoset_requires_grad
