#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from pathlib import Path
import tempfile

import torch
import pytest

from nnscaler.graph import IRGraph
from nnscaler.parallel import ComputeConfig, _gen_graph, parallelize

from ..utils import replace_all_device_with


class QuickModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 5)

    def forward(self, x):
        x = self.linear(x)
        return torch.sum(x)


@replace_all_device_with('cpu')
@pytest.mark.parametrize('autoset_requires_grad', [True, False])
@pytest.mark.parametrize('end2end_mode', [True, False])
@pytest.mark.parametrize('requires_grad', [True, False])
def test_gen_graph_autoset_requires_grad_non_end2end(
    tmp_path: Path,
    autoset_requires_grad: bool,
    end2end_mode: bool,
    requires_grad: bool
):
    parallelize(
            QuickModule(),
            {'x': torch.randn(2, 3, requires_grad=requires_grad)},
            'dp',
            ComputeConfig(1, 1, use_end2end=end2end_mode),
            gen_savedir=tmp_path,
            reuse='override',
            autoset_requires_grad=autoset_requires_grad,
            load_module=False,
    )
    graph_ckp = list(Path(tmp_path).glob('**/graph.ckp'))[0]
    graph = IRGraph.load(graph_ckp)
    if end2end_mode:
        input_tensor = graph.node(0).output(0)
    else:
        input_tensor = graph.input(0)

    assert input_tensor.name == 'x'
    assert input_tensor.requires_grad == (
        (autoset_requires_grad and not end2end_mode)
        or (not autoset_requires_grad and requires_grad)
    )
