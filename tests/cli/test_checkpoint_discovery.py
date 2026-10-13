# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from nnscaler.cli.serialization import Checkpointer


def test_discovers_paths_generated_by_checkpointer(tmp_path):
    checkpointer = Checkpointer()
    for epoch, step in [(0, 9999), (0, 10000), (1, 2)]:
        path = tmp_path / checkpointer.get_checkpoint_file_path(epoch, step, rank=0)
        path.parent.mkdir()
        checkpointer.save({'step': step}, path)
        assert Checkpointer.find_latest_checkpoint(tmp_path) == str(path.parent)
    selected = Checkpointer.find_latest_checkpoint(tmp_path)
    assert Path(selected).name == '0001-0002'
    assert checkpointer.load_for_rank(selected, rank=0) == {'step': 2}


def test_latest_orders_numeric_epoch_step_and_ignores_exports(tmp_path):
    for name in ['0000-0099', '0000-0100', '0001-0001', 'epoch2-step3',
                 '0009-9999-hf', '0009-9999-merged', 'last', 'best', 'unrelated']:
        (tmp_path / name).mkdir()
    (tmp_path / '9999-9999').write_text('not a directory')
    assert Checkpointer.find_latest_checkpoint(tmp_path) == str(tmp_path / 'epoch2-step3')
    # Do not fall back to an older checkpoint just because this one is incomplete.
    (tmp_path / 'epoch2-step4').mkdir()
    assert Checkpointer.find_latest_checkpoint(tmp_path) == str(tmp_path / 'epoch2-step4')


def test_missing_or_no_numbered_directory(tmp_path):
    assert Checkpointer.find_latest_checkpoint(tmp_path / 'absent') is None
    assert Checkpointer.find_latest_checkpoint(tmp_path) is None
    for name in ['last', 'best', '0000-0100-hf']:
        (tmp_path / name).mkdir()
    assert Checkpointer.find_latest_checkpoint(tmp_path) is None


def test_scan_errors_are_not_treated_as_missing_checkpoint(tmp_path):
    with patch.object(Path, 'iterdir', side_effect=PermissionError('scan denied')):
        with pytest.raises(PermissionError, match='scan denied'):
            Checkpointer.find_latest_checkpoint(tmp_path)


def _discovery_worker(rank, rendezvous, root):
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank,
                            world_size=2, timeout=timedelta(seconds=30))
    try:
        directory = Path(root) / 'checkpoints'
        scans = 0
        fail_scan = False
        iterdir = Path.iterdir

        def scan(path):
            nonlocal scans
            if path == directory:
                assert rank == 0, 'non-source rank accessed checkpoint directory'
                scans += 1
                if fail_scan:
                    raise PermissionError('checkpoint scan denied')
            return iterdir(path)

        with patch.object(Path, 'iterdir', scan):
            assert Checkpointer.find_latest_checkpoint(directory) == str(directory / '0000-0020')
            if rank == 0:
                (directory / '0000-0021').mkdir()
            dist.barrier()
            assert Checkpointer.find_latest_checkpoint(directory) == str(directory / '0000-0021')
            fail_scan = True
            with pytest.raises(RuntimeError, match='checkpoint scan denied'):
                Checkpointer.find_latest_checkpoint(directory)
            fail_scan = False
            assert Checkpointer.find_latest_checkpoint(directory) == str(directory / '0000-0021')
        assert scans == (4 if rank == 0 else 0)
    finally:
        dist.destroy_process_group()


def test_distributed_discovery_scans_once_and_shares_results_and_errors(tmp_path):
    (tmp_path / 'checkpoints' / '0000-0020').mkdir(parents=True)
    mp.spawn(_discovery_worker, args=(f'file://{tmp_path}/gloo-init', str(tmp_path)),
             nprocs=2, join=True)
