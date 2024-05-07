import pytest

import tempfile
import torch
import os
from pathlib import Path
from nnscaler.graph.parser.converter import to_fx_graph, to_ir_graph
from nnscaler.autodist.model_graph import ModelGraph
from nnscaler.autodist.autodist_config import AutoDistConfig
from nnscaler.autodist.spmd_solver import SPMDSolver

import nnscaler


@nnscaler.register_op(
    '(1 h) l^ d^, (1 h) l^ d^, (1 h) l^ d^ -> (1 h) l^ d^', 'mock_attention')
def mock_attention(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
    return x + y + z


class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, y, z):
        t = mock_attention(x, y, z)
        return t.sum()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cube_operator():
    bsz, head_num, seq_len, head_dim = 1, 8, 128, 64
    data = torch.randn((bsz * head_num, seq_len, head_dim))

    dummy_input = {'x': data, 'y': data, 'z': data}
    model = Model()
    model.train()
    fx_graph = to_fx_graph(model, dummy_input)

    with tempfile.TemporaryDirectory() as tempdir:
        ir_graph = to_ir_graph(fx_graph,
                               dummy_input,
                               attr_savedir=tempdir,
                               dynamic_shape=False)
        cfg = AutoDistConfig(mesh_col=2)
        model_graph = ModelGraph(ir_graph, cfg)
        mock_attention_op = model_graph.operator_list[0]
        assert mock_attention_op.pos2dim_id((0, 0)) == 'h'
        assert mock_attention_op.dim_id2pos('h') == (0, 0)
