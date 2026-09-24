#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from pathlib import Path

import pytest
import torch

from nnscaler.graph.parser import FxModuleParser
from nnscaler.graph.parser.frame import Frame
from nnscaler.ir.tensor import IRFullTensor
from nnscaler.runtime.module import AttrMeta, CubeModule


def test_save_attr_content_index(tmp_path: Path):
    frame = Frame()
    tensors = [IRFullTensor((4,), name=f'w{idx}') for idx in range(3)]
    values = [torch.arange(4) + idx * 4 for idx in range(3)]
    for idx, (tensor, value) in enumerate(zip(tensors, values)):
        frame.add_attr(tensor, value, f'w{idx}')

    file_stem = tmp_path / FxModuleParser.ATTR_CONTENT_FILE_STEM
    frame.save_attr_content(file_stem, params_per_file=5)

    tid_to_chunk = torch.load(
        tmp_path / FxModuleParser.ATTR_CONTENT_INDEX_FILE,
        weights_only=True,
    )
    assert tid_to_chunk == {tensor.tid: idx for idx, tensor in enumerate(tensors)}
    for idx, (tensor, value) in enumerate(zip(tensors, values)):
        assert torch.equal(torch.load(f'{file_stem}.{idx}', weights_only=True)[tensor.tid], value)


def test_load_attr_content_only_reads_required_chunks(tmp_path: Path, monkeypatch):
    file_stem = tmp_path / FxModuleParser.ATTR_CONTENT_FILE_STEM
    torch.save({10: torch.full((4,), 10.0)}, f'{file_stem}.0')
    # This chunk must never be opened by the loader.
    Path(f'{file_stem}.1').write_bytes(b'not a torch checkpoint')
    torch.save({30: torch.arange(8, dtype=torch.float32)}, f'{file_stem}.2')
    torch.save({10: 0, 20: 1, 30: 2}, tmp_path / FxModuleParser.ATTR_CONTENT_INDEX_FILE)

    module = CubeModule()
    module.register_parameter('local_weight', torch.nn.Parameter(torch.empty(3)))
    module._fullmap['local_weight'] = AttrMeta(
        tid=30,
        is_param=True,
        orig_name='weight',
        shape=(8,),
        slicers=(slice(2, 5),),
        val_chunks=1,
        dtype=torch.float32,
        sub_shape=(3,),
    )

    load_calls = []
    torch_load = torch.load

    def record_load(filename, *args, **kwargs):
        load_calls.append((Path(filename).name, kwargs.copy()))
        return torch_load(filename, *args, **kwargs)

    monkeypatch.setattr(torch, 'load', record_load)
    module.load_attr_content(str(file_stem))

    assert torch.equal(module.local_weight, torch.arange(2, 5, dtype=torch.float32))
    assert load_calls == [
        (FxModuleParser.ATTR_CONTENT_INDEX_FILE, {'weights_only': True}),
        ('fullmodel.pt.2', {'mmap': True, 'weights_only': True}),
    ]


@pytest.mark.parametrize('workers', [1, 4])
def test_save_attr_content_balances_elements_and_preserves_values(tmp_path, workers):
    frame = Frame()
    base = torch.arange(48, dtype=torch.float32).reshape(6, 8)
    values = [base[:, ::2], torch.arange(3, dtype=torch.bfloat16),
              torch.arange(3, dtype=torch.int64), base.t()]
    tensors = [IRFullTensor(tuple(value.shape), name=f'w{i}', dtype=value.dtype)
               for i, value in enumerate(values)]
    for i, (tensor, value) in enumerate(zip(tensors, values)):
        frame.add_attr(tensor, value, f'w{i}')
    stem = tmp_path/'fullmodel.pt'
    frame.save_attr_content(stem, params_per_file=8, max_workers=workers)
    index = torch.load(f'{stem}.index', weights_only=True)
    assert index == {tensors[0].tid: 0, tensors[1].tid: 1,
                     tensors[2].tid: 1, tensors[3].tid: 2}
    for tensor, value in zip(tensors, values):
        actual = torch.load(f'{stem}.{index[tensor.tid]}', mmap=True, weights_only=True)[tensor.tid]
        assert torch.equal(actual, value)
        assert actual.dtype == value.dtype
        assert actual.stride() == value.stride()


def test_save_attr_content_failure_does_not_publish_index(tmp_path, monkeypatch):
    frame = Frame()
    for i in range(4):
        frame.add_attr(IRFullTensor((4,), name=f'w{i}'), torch.ones(4), f'w{i}')
    save = torch.save
    def fail_one(obj, filename, **kwargs):
        if str(filename).endswith('.1'):
            raise OSError('injected write failure')
        save(obj, filename, **kwargs)
    monkeypatch.setattr(torch, 'save', fail_one)
    stem = tmp_path/'fullmodel.pt'
    with pytest.raises(OSError, match='injected write failure'):
        frame.save_attr_content(stem, params_per_file=4)
    assert not Path(f'{stem}.index').exists()


def test_save_empty_attr_content(tmp_path):
    stem = tmp_path/'fullmodel.pt'
    Frame().save_attr_content(stem)
    assert torch.load(f'{stem}.0', weights_only=True) == {}
    assert torch.load(f'{stem}.index', weights_only=True) == {}
