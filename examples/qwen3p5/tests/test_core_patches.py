"""Unit tests for Phase 0 surgical `nnscaler/` patches supporting Qwen3.5."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from nnscaler.codegen.emit import _safe_repr_value
from nnscaler.graph.function.dimops import IRDimops
from nnscaler.ir.operator import IRFwOperation
from nnscaler.parallel import ComputeConfig, ParamInitStrategy, _compile_flags, _gen_graph, parallelize


class _PatchProbeModule(nn.Module):
    def __init__(self, hidden: int = 32) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden, hidden, bias=False)
        self.act = nn.Softplus()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Tests: F.softplus, nn.Softplus, trailing-dim broadcast masked_fill, and contiguous(memory_format=...)
        h = self.proj(x)
        h = F.softplus(h) + self.act(h)
        # x is [B, S, H], mask[:, :, None] is [B, S, 1] -> trailing-dim broadcast in masked_fill!
        h = h.masked_fill(mask.unsqueeze(-1), 0.0)
        h = h.contiguous(memory_format=torch.contiguous_format)
        return h.sum()


def test_safe_repr_memory_format() -> None:
    assert repr(_safe_repr_value(torch.contiguous_format)) == "torch.contiguous_format"
    assert repr(_safe_repr_value(torch.preserve_format)) == "torch.preserve_format"


def test_softplus_and_broadcast_masked_fill_parse_as_dimops(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    mod = _PatchProbeModule(hidden=16)
    x = torch.randn(2, 8, 16)
    mask = torch.zeros(2, 8, dtype=torch.bool)
    mask[0, -1] = True

    cfg = ComputeConfig(1, 1, trace_strategy="cpu")
    with _compile_flags(cfg):
        graph, _ = _gen_graph(mod, {"x": x, "mask": mask}, str(tmp_path), cfg)
    fw_ops = graph.select(ntype=IRFwOperation)
    assert all(isinstance(op, IRDimops) for op in fw_ops)

    softplus_ops = [op for op in fw_ops if "softplus" in str(op.signature)]
    assert len(softplus_ops) == 2
    for sp in softplus_ops:
        assert (0, 2) in sp.transform_space()

    mfill_ops = [op for op in fw_ops if "masked_fill" in str(op.signature)]
    assert len(mfill_ops) == 1
    # Dimension 2 (`H=16`) is broadcast against `1` in mask `[2, 8, 1]`, so `(0, 2)` is partitionable!
    assert (0, 2) in mfill_ops[0].transform_space()


def test_weightless_meta_module_parallelize(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    with torch.device("meta"):
        meta_mod = _PatchProbeModule(hidden=32)

    x = torch.randn(2, 8, 32)
    mask = torch.zeros(2, 8, dtype=torch.bool)

    parallelize(
        meta_mod,
        {"x": x, "mask": mask},
        pas_policy="tp",
        compute_config=ComputeConfig(
            2,
            2,
            use_end2end=True,
            trace_strategy="meta",
            param_init_strategy=ParamInitStrategy.SHARD,
        ),
        gen_savedir=str(tmp_path),
        reuse="override",
        load_module=False,
        broadcast_strategy="none",
    )
    gencode_files = list(tmp_path.rglob("gencode*.py"))
    assert len(gencode_files) == 2
