# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Declarative JSON-driven Parallelization Policy (PAS) for Kimi-Linear & Kimi-K2.5."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any

import more_itertools as mitr

from nnscaler.graph import IRGraph
from nnscaler.graph.function.dimops import IRDimops
from nnscaler.graph.segment import IRSegment
from nnscaler.ir.operator import IRDataOperation, IRFwOperation
from nnscaler.policies import _replica, _tp, auto_multiref

from ..rewrites import RewriteConfig


class StrategyValidationError(ValueError):
    """Raised when a declarative JSON strategy violates tensor/mesh invariants."""


@dataclass
class MeshConfig:
    plan_ngpus: int = 4
    runtime_ngpus: int | None = None
    pipeline_nstages: int = 1
    pipeline_size: int | None = None
    pipeline_nmicros: int = 1
    pipeline_scheduler: str = "1f1b"
    zero_stage: int = 0

    def resolved_runtime_ngpus(self) -> int:
        return self.runtime_ngpus if self.runtime_ngpus is not None else self.plan_ngpus

    def resolved_pipeline_size(self) -> int:
        return self.pipeline_size if self.pipeline_size is not None else self.pipeline_nstages


@dataclass
class StageSplitRule:
    layers: str
    stage_id: int

    def layer_indices(self) -> list[int]:
        parts = [p.strip() for p in self.layers.split(",") if p.strip()]
        indices: list[int] = []
        for part in parts:
            if ".." in part:
                lo, hi = part.split("..", 1)
                indices.extend(range(int(lo), int(hi) + 1))
            elif "-" in part:
                lo, hi = part.split("-", 1)
                indices.extend(range(int(lo), int(hi) + 1))
            else:
                indices.append(int(part))
        return indices


@dataclass
class PartitionRule:
    name: str
    partition: str | dict[str, int]
    fqn_regex: str | None = None
    op_regex: str | None = None
    param_regex: str | None = None
    on_indivisible: str = "error"

    def matches(self, node: IRFwOperation, fqn: str) -> bool:
        if self.fqn_regex is not None and not re.search(self.fqn_regex, fqn):
            return False
        if self.op_regex is not None and not re.search(self.op_regex, node.signature):
            return False
        if self.param_regex is not None:
            names = [getattr(t, "name", "") for t in node.inputs()]
            if not any(re.search(self.param_regex, str(n)) for n in names):
                return False
        return True


@dataclass
class StrategyConfig:
    name: str
    description: str = ""
    mesh: MeshConfig = field(default_factory=MeshConfig)
    rewrites: RewriteConfig = field(default_factory=RewriteConfig)
    default_partition: str = "replicate"
    recompute_modules: list[str] = field(default_factory=list)
    stage_splits: list[StageSplitRule] = field(default_factory=list)
    rules: list[PartitionRule] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "mesh": {
                "plan_ngpus": self.mesh.plan_ngpus,
                "runtime_ngpus": self.mesh.resolved_runtime_ngpus(),
                "pipeline_nstages": self.mesh.pipeline_nstages,
                "pipeline_size": self.mesh.resolved_pipeline_size(),
                "pipeline_nmicros": self.mesh.pipeline_nmicros,
                "pipeline_scheduler": self.mesh.pipeline_scheduler,
                "zero_stage": self.mesh.zero_stage,
            },
            "rewrites": self.rewrites.to_dict(),
            "default_partition": self.default_partition,
            "recompute_modules": list(self.recompute_modules),
            "stage_splits": [{"layers": s.layers, "stage_id": s.stage_id} for s in self.stage_splits],
            "rules": [
                {
                    "name": r.name,
                    "fqn_regex": r.fqn_regex,
                    "op_regex": r.op_regex,
                    "param_regex": r.param_regex,
                    "partition": r.partition,
                    "on_indivisible": r.on_indivisible,
                }
                for r in self.rules
            ],
        }


def load_strategy_config(path: str | Path) -> StrategyConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    mesh_raw = raw.get("mesh", {})
    mesh = MeshConfig(
        plan_ngpus=int(mesh_raw.get("plan_ngpus", 4)),
        runtime_ngpus=mesh_raw.get("runtime_ngpus"),
        pipeline_nstages=int(mesh_raw.get("pipeline_nstages", 1)),
        pipeline_size=mesh_raw.get("pipeline_size"),
        pipeline_nmicros=int(mesh_raw.get("pipeline_nmicros", 1)),
        pipeline_scheduler=str(mesh_raw.get("pipeline_scheduler", "1f1b")),
        zero_stage=int(mesh_raw.get("zero_stage", 0)),
    )
    rewrites = RewriteConfig.from_dict(raw.get("rewrites"))
    splits = [StageSplitRule(layers=str(s["layers"]), stage_id=int(s["stage_id"])) for s in raw.get("stage_splits", [])]
    rules = [
        PartitionRule(
            name=str(r.get("name", f"rule_{idx}")),
            partition=r["partition"],
            fqn_regex=r.get("fqn_regex"),
            op_regex=r.get("op_regex"),
            param_regex=r.get("param_regex"),
            on_indivisible=str(r.get("on_indivisible", "error")),
        )
        for idx, r in enumerate(raw.get("rules", []))
    ]
    return StrategyConfig(
        name=str(raw.get("name", Path(path).stem)),
        description=str(raw.get("description", "")),
        mesh=mesh,
        rewrites=rewrites,
        default_partition=str(raw.get("default_partition", "replicate")),
        recompute_modules=[str(x) for x in raw.get("recompute_modules", [])],
        stage_splits=splits,
        rules=rules,
    )


_LAST_POLICY_REPORT: dict[str, Any] = {}


def get_last_policy_report() -> dict[str, Any]:
    return dict(_LAST_POLICY_REPORT)


def _node_fqn(node: IRFwOperation) -> str:
    if not getattr(node, "module_stack", None):
        return ""
    return str(next(reversed(node.module_stack)))


def _extract_layer_id(fqn: str) -> int | None:
    m = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", fqn)
    return int(m.group(1)) if m else None


def _resolve_partition_spec(node: IRFwOperation, spec: str | dict[str, int]) -> tuple[int, int] | str:
    if isinstance(spec, dict):
        return (int(spec["input"]), int(spec["dim"]))
    s = spec.lower().strip()
    if s in ("replicate", "auto"):
        return s
    if s == "tp_col":
        return (1, 0) if "linear" in node.signature else (0, -1)
    if s == "tp_row":
        return (1, 1) if "linear" in node.signature else (0, -1)
    if s == "vocab_parallel":
        return (1, 0)
    if s == "head_parallel":
        sig = node.signature
        if "kimi_chunk_kda" in sig:
            return (0, 2)  # [B, S, H, D] -> H
        if "kimi_causal_conv1d" in sig:
            return (0, 1)  # [B, (h d^), S] -> h
        if "kimi_mla_concat_kv" in sig:
            return (0, 1)  # [B, H, S, Dn] -> H
        if "kimi_k25_vision_varlen_sdpa" in sig:
            return (0, 1)  # [1, H, S, D] -> H
        if "scaled_dot_product_attention" in sig:
            return (0, 1)  # [B, H, S, D] -> H
        return (0, 1)
    if s == "expert_parallel":
        if "kimi_moe_grouped_gemm" in node.signature:
            return (3, 0)
        if "kimi_grouped_linear" in node.signature:
            return (2, 0)
        return (0, 0)
    if s == "expert_tp_col":
        return (2, 1)
    if s == "expert_tp_row":
        return (2, 2)
    raise StrategyValidationError(f"Unknown partition spec {spec!r} on node {node.signature}")


def _validate_and_tp(graph: IRGraph, node: IRFwOperation, devs: list[int], idx: int, dim: int, rule_name: str, fqn: str, on_indivisible: str) -> bool:
    tp_size = len(devs)
    if not isinstance(node, IRDimops):
        if on_indivisible == "fallback_replicate":
            _replica(graph, node, devs)
            return False
        raise StrategyValidationError(f"Rule '{rule_name}': node {node.signature} ({fqn}) is not an IRDimops")

    t = node.input(idx)
    if not hasattr(t, "shape"):
        raise StrategyValidationError(f"Rule '{rule_name}': input {idx} of {node.signature} ({fqn}) is not a tensor")
    ndims = len(t.shape)
    norm_dim = dim if dim >= 0 else ndims + dim
    if norm_dim < 0 or norm_dim >= ndims:
        raise StrategyValidationError(f"Rule '{rule_name}': dim={dim} out of range for shape {tuple(t.shape)} ({fqn})")
    axis_len = int(t.shape[norm_dim])
    if axis_len % tp_size != 0:
        msg = (
            f"Rule '{rule_name}': tensor shape {tuple(t.shape)} dim {norm_dim} (size={axis_len}) "
            f"is not divisible by tp_size={tp_size} on {node.signature} ({fqn})"
        )
        if on_indivisible == "fallback_replicate":
            _replica(graph, node, devs)
            return False
        raise StrategyValidationError(msg)
    try:
        _tp(graph, node, devs, idx=idx, dim=norm_dim)
        return True
    except Exception as exc:
        if on_indivisible == "fallback_replicate":
            _replica(graph, node, devs)
            return False
        raise StrategyValidationError(f"Rule '{rule_name}' failed on {node.signature} ({fqn}): {exc}") from exc


def build_json_pas(
    strategy: StrategyConfig,
    *,
    pre_pas_hook: Any = None,
    post_pas_hook: Any = None,
):
    def pas_policy(graph: IRGraph, cfg: Any) -> IRGraph:
        global _LAST_POLICY_REPORT
        if pre_pas_hook is not None:
            pre_pas_hook(graph)
        ngpus = cfg.plan_ngpus
        nstages = strategy.mesh.pipeline_nstages
        nmicros = cfg.pas_config.get("pipeline_nmicros", strategy.mesh.pipeline_nmicros)
        scheduler = cfg.pas_config.get("pipeline_scheduler", strategy.mesh.pipeline_scheduler)
        pp_size = cfg.pas_config.get("pipeline_size", strategy.mesh.resolved_pipeline_size())

        if ngpus % pp_size != 0:
            raise StrategyValidationError(f"plan_ngpus={ngpus} not divisible by pipeline_size={pp_size}")
        if nstages % pp_size != 0:
            raise StrategyValidationError(f"pipeline_nstages={nstages} not divisible by pipeline_size={pp_size}")
        tp_size = ngpus // pp_size

        auto_multiref(graph)
        fnodes = graph.select(ntype=IRFwOperation)

        if strategy.recompute_modules:
            recompute_regs = [re.compile(p) for p in strategy.recompute_modules]
            groups: dict[int, list[IRFwOperation]] = {}
            for node in fnodes:
                fqn = _node_fqn(node)
                lid = _extract_layer_id(fqn)
                if lid is not None and any(r.search(fqn) for r in recompute_regs):
                    groups.setdefault(lid, []).append(node)
            for grp in groups.values():
                graph.recompute(grp)

        if nstages > 1:
            if strategy.stage_splits:
                layer_to_stage: dict[int, int] = {}
                for sp in strategy.stage_splits:
                    for lid in sp.layer_indices():
                        layer_to_stage[lid] = sp.stage_id
                stage_buckets: list[list[IRFwOperation]] = [[] for _ in range(nstages)]
                cur_stage = 0
                for node in fnodes:
                    lid = _extract_layer_id(_node_fqn(node))
                    if lid is not None and lid in layer_to_stage:
                        cur_stage = layer_to_stage[lid]
                    stage_buckets[cur_stage].append(node)
                pivots = [b[0] for b in stage_buckets if b]
                if len(pivots) != nstages:
                    raise StrategyValidationError(f"Expected {nstages} non-empty pipeline stages, got {len(pivots)}")
                graph.staging(pivots)
            else:
                stages = [list(s) for s in mitr.divide(nstages, fnodes)]
                graph.staging([s[0] for s in stages])
        else:
            graph.staging([fnodes[0]])

        segments = [s for s in graph.select(ntype=IRSegment, flatten=False) if s.isfw()]
        rule_hits: dict[str, int] = {r.name: 0 for r in strategy.rules}
        warnings: list[str] = []
        replicated_default = 0
        all_devs = list(range(ngpus))

        for seg_idx, seg in enumerate(segments):
            pp_idx = seg_idx % pp_size
            devs = all_devs[pp_idx * tp_size : (pp_idx + 1) * tp_size]
            for node in seg.nodes():
                if not isinstance(node, IRFwOperation):
                    continue
                fqn = _node_fqn(node)
                matched_rule: PartitionRule | None = None
                for rule in strategy.rules:
                    if rule.matches(node, fqn):
                        matched_rule = rule
                        break

                if matched_rule is None:
                    if strategy.default_partition == "auto":
                        try:
                            _tp(graph, node, devs, idx=0, dim=0)
                        except Exception:
                            _replica(graph, node, devs)
                    else:
                        _replica(graph, node, devs)
                    replicated_default += 1
                    continue

                spec = _resolve_partition_spec(node, matched_rule.partition)
                if spec == "replicate":
                    _replica(graph, node, devs)
                    rule_hits[matched_rule.name] += 1
                elif spec == "auto":
                    try:
                        _tp(graph, node, devs, idx=0, dim=0)
                    except Exception:
                        _replica(graph, node, devs)
                    rule_hits[matched_rule.name] += 1
                else:
                    idx, dim = spec
                    ok = _validate_and_tp(
                        graph, node, devs, idx, dim, matched_rule.name, fqn, matched_rule.on_indivisible
                    )
                    if ok:
                        rule_hits[matched_rule.name] += 1
                    else:
                        warnings.append(f"Rule '{matched_rule.name}' fell back to replicate on {node.signature} ({fqn})")

        for dl in graph.select(ntype=IRDataOperation):
            _replica(graph, dl, devs=all_devs)

        if nstages > 1:
            cfg.apply_pipeline_scheduler(graph, nstages, nmicros, scheduler)

        _LAST_POLICY_REPORT = {
            "strategy_name": strategy.name,
            "plan_ngpus": ngpus,
            "pp_size": pp_size,
            "tp_size": tp_size,
            "nstages": nstages,
            "rule_hit_counts": rule_hits,
            "default_replicated_ops": replicated_default,
            "warnings": warnings,
        }
        if post_pas_hook is not None:
            post_pas_hook(graph)
        return graph

    return pas_policy
