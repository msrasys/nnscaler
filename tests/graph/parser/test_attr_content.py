#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from pathlib import Path

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

    loaded_files = []
    torch_load = torch.load

    def record_load(filename, *args, **kwargs):
        loaded_files.append((Path(filename).name, kwargs.get('weights_only')))
        return torch_load(filename, *args, **kwargs)

    monkeypatch.setattr(torch, 'load', record_load)
    module.load_attr_content(str(file_stem))

    assert torch.equal(module.local_weight, torch.arange(2, 5, dtype=torch.float32))
    assert loaded_files == [
        (FxModuleParser.ATTR_CONTENT_INDEX_FILE, True),
        ('fullmodel.pt.2', True),
    ]
