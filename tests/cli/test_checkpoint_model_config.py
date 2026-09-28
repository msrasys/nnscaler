import copy

import pytest
import torch

import nnscaler
from nnscaler.cli.serialization import Checkpointer
from nnscaler.cli.trainer import Trainer


def _model_config(path):
    return {
        'type': 'vl.VLPipelineWrapperModel',
        'args': {
            'model_args': {
                'vision_encoder_path': path,
                'vision_patch_size': 14,
                'vision_freeze': True,
                'd_model': 3072,
            },
            'precision': 'bf16',
            'pipeline_size': 3,
        },
        'parallel_modules': [
            {'type': 'vl.VLLanguagePipelineWrapperModel',
             'compute_config': {'use_end2end': True}},
            {'type': 'vl.VisionBackbone'},
        ],
    }


def _write_checkpoints(tmp_path, configs, *, lr_steps=(10, 10)):
    writer = Checkpointer()
    paths = []
    for rank, config in enumerate(configs):
        path = tmp_path / f'{rank}.ckpt'
        writer.save({
            'model': {'weight': torch.tensor([rank])},
            'optimizer': {'state': torch.tensor([rank + 10])},
            'train_args': {'model': config, 'checkpoint': {'save_type': 'deduped'}},
            'lr_scheduler': {'step': lr_steps[rank]},
            'train_status': {'finished_train_steps': 10},
            'rank': rank,
            'nnscaler': 'test',
            'dataloader': {'rank': rank},
        }, path)
        paths.append(path)
    return paths


def _stub_tensor_merge(monkeypatch):
    calls = []

    def merge(models, optimizers):
        calls.append((models, optimizers))
        return ({'weights': [m['weight'] for m in models]},
                {'states': [o['state'] for o in optimizers]} if optimizers is not None else None)

    monkeypatch.setattr(nnscaler, 'merge_state_dicts', merge)
    return calls


@pytest.mark.parametrize('model_only', [False, True])
def test_merge_accepts_rank_local_vision_paths_without_rewriting_metadata(tmp_path, monkeypatch, model_only):
    configs = [_model_config('/data/tmp.node0/vision'),
               _model_config('/data/tmp.node1/vision')]
    original = copy.deepcopy(configs)
    paths = _write_checkpoints(tmp_path, configs)
    file_contents = [p.read_bytes() for p in paths]
    calls = _stub_tensor_merge(monkeypatch)

    # Directory resume and the offline public API share this merge method.
    output = tmp_path / 'merged.ckpt'
    Trainer.merge_checkpoint(paths, output, model_only=model_only)
    merged = Checkpointer().load(output)
    assert len(calls) == 1
    assert [v.item() for v in merged['model']['weights']] == [0, 1]
    if not model_only:
        assert merged['train_args']['model'] == original[0]
        assert merged['train_args']['checkpoint']['save_type'] == 'merged'
        assert [v.item() for v in merged['optimizer']['states']] == [10, 11]
        assert merged['dataloader'] == [{'rank': 0}, {'rank': 1}]
    assert configs == original
    assert [p.read_bytes() for p in paths] == file_contents


@pytest.mark.parametrize('key,value', [
    ('d_model', 4096),
    ('vision_patch_size', 16),
    ('vision_freeze', False),
])
def test_merge_still_rejects_model_changes_with_relocated_vision(tmp_path, monkeypatch, key, value):
    configs = [_model_config('/node0/vision'), _model_config('/node1/vision')]
    configs[1]['args']['model_args'][key] = value
    paths = _write_checkpoints(tmp_path, configs)
    calls = _stub_tensor_merge(monkeypatch)
    with pytest.raises(ValueError, match='model config in'):
        Trainer._merge_checkpoint(paths)
    assert not calls


@pytest.mark.parametrize('change', ['type', 'precision', 'pipeline_size', 'parallel_modules',
                                  'top_level_path', 'other_nested_path'])
def test_merge_only_ignores_the_documented_path(tmp_path, monkeypatch, change):
    configs = [_model_config('/node0/vision'), _model_config('/node1/vision')]
    if change == 'type':
        configs[1]['type'] = 'another.Model'
    elif change in ('precision', 'pipeline_size'):
        configs[1]['args'][change] = 'fp32' if change == 'precision' else 4
    elif change == 'parallel_modules':
        configs[1]['parallel_modules'][0]['compute_config']['use_end2end'] = False
    elif change == 'top_level_path':
        for rank, config in enumerate(configs):
            config['vision_encoder_path'] = f'/other{rank}'
    else:
        for rank, config in enumerate(configs):
            config['args']['other'] = {'vision_encoder_path': f'/other{rank}'}
    paths = _write_checkpoints(tmp_path, configs)
    calls = _stub_tensor_merge(monkeypatch)
    with pytest.raises(ValueError, match='model config in'):
        Trainer._merge_checkpoint(paths)
    assert not calls


def test_merge_still_checks_lr_scheduler(tmp_path, monkeypatch):
    paths = _write_checkpoints(tmp_path, [_model_config('/node0/vision'),
                                        _model_config('/node1/vision')], lr_steps=(10, 11))
    calls = _stub_tensor_merge(monkeypatch)
    with pytest.raises(ValueError, match='lr_scheduler state'):
        Trainer._merge_checkpoint(paths)
    assert not calls


@pytest.mark.parametrize('args', [{}, {'model_args': None}, {'model_args': 'unchanged'}])
def test_merge_models_without_vl_path(tmp_path, monkeypatch, args):
    config = {'type': 'text.Model', 'args': args}
    paths = _write_checkpoints(tmp_path, [config, copy.deepcopy(config)])
    _stub_tensor_merge(monkeypatch)
    merged = Trainer._merge_checkpoint(paths)
    assert merged['train_args']['model'] == config
