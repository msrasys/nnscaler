# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""JSON-configurable declarative parallelization compiler targeting `nnscaler.policies.fn` for DeepSeek."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Callable

from nnscaler.graph import IRGraph
from nnscaler.graph.function.dimops import IRDimops
from nnscaler.ir.operator import IRFwOperation
from nnscaler.ir.tensor import IRSubTensor
from nnscaler.parallel import ComputeConfig, ParamInitStrategy
from nnscaler.policies import OpPartition, OpPlan
from nnscaler.policies import fn as policies_fn

from examples.Deepseek.rewrites.model_rewrites import RewriteConfig


class StrategyValidationError(ValueError):
    """Raised when a declarative JSON strategy violates mesh, stage, or divisibility invariants."""


@dataclass
class MeshSpec:
    plan_ngpus: int = 2
    runtime_ngpus: int | None = None
    pipeline_nstages: int = 1
    pipeline_size: int | None = None
    pipeline_nmicros: int = 1
    pipeline_scheduler: str = "1f1b"
    zero_stage: int = 0
    zero_ngroups: int = 1

    @property
    def effective_pp_size(self) -> int:
        return self.pipeline_size if self.pipeline_size is not None else self.pipeline_nstages

    @property
    def tp_size(self) -> int:
        return self.plan_ngpus // self.effective_pp_size


@dataclass
class StageSplitSpec:
    layers: str  # e.g. "0..15" or "0-15" or "5"
    stage_id: int

    def expand_indices(self) -> list[int]:
        sep = ".." if ".." in self.layers else ("-" if "-" in self.layers else None)
        if sep is None:
            return [int(self.layers.strip())]
        lo_s, hi_s = self.layers.split(sep, 1)
        lo, hi = int(lo_s.strip()), int(hi_s.strip())
        if hi < lo:
            raise StrategyValidationError(f"Invalid layer range {self.layers!r}")
        return list(range(lo, hi + 1))


@dataclass
class PartitionRuleSpec:
    name: str = ""
    fqn_regex: str | None = None
    op_regex: str | None = None
    param_regex: str | None = None
    partition: str | dict[str, int] | None = "auto"
    on_indivisible: str = "error"  # "error" | "fallback_replicate" | "fallback_auto"


@dataclass
class StrategyConfig:
    name: str = "default_tp2"
    description: str = ""
    mesh: MeshSpec = field(default_factory=MeshSpec)
    rewrites: RewriteConfig = field(default_factory=RewriteConfig)
    default_partition: str = "auto"  # "auto" | "replicate"
    recompute_modules: list[str] = field(default_factory=list)
    stage_splits: list[StageSplitSpec] = field(default_factory=list)
    rules: list[PartitionRuleSpec] = field(default_factory=list)

    def validate_mesh(self) -> None:
        m = self.mesh
        if m.plan_ngpus < 1:
            raise StrategyValidationError(f"plan_ngpus must be >= 1, got {m.plan_ngpus}")
        pp = m.effective_pp_size
        if m.pipeline_nstages < 1 or pp < 1:
            raise StrategyValidationError(f"pipeline_nstages ({m.pipeline_nstages}) and pp_size ({pp}) must be >= 1")
        if m.pipeline_nstages % pp != 0:
            raise StrategyValidationError(f"pipeline_nstages={m.pipeline_nstages} must be divisible by pp_size={pp}")
        if m.plan_ngpus % pp != 0:
            raise StrategyValidationError(f"plan_ngpus={m.plan_ngpus} must be divisible by pp_size={pp}")
        if m.pipeline_nstages > 1 and m.pipeline_nmicros < 1:
            raise StrategyValidationError("pipeline_nmicros must be >= 1 when pipeline_nstages > 1")
        if m.runtime_ngpus is not None and m.runtime_ngpus % m.plan_ngpus != 0:
            raise StrategyValidationError(
                f"runtime_ngpus={m.runtime_ngpus} must be a multiple of plan_ngpus={m.plan_ngpus}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "mesh": {
                "plan_ngpus": self.mesh.plan_ngpus,
                "runtime_ngpus": self.mesh.runtime_ngpus or self.mesh.plan_ngpus,
                "pipeline_nstages": self.mesh.pipeline_nstages,
                "pipeline_size": self.mesh.effective_pp_size,
                "tp_size": self.mesh.tp_size,
                "pipeline_nmicros": self.mesh.pipeline_nmicros,
                "pipeline_scheduler": self.mesh.pipeline_scheduler,
                "zero_stage": self.mesh.zero_stage,
                "zero_ngroups": self.mesh.zero_ngroups,
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

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "StrategyConfig":
        mesh_raw = dict(raw.get("mesh", {}))
        mesh = MeshSpec(
            plan_ngpus=int(mesh_raw.get("plan_ngpus", 2)),
            runtime_ngpus=int(mesh_raw["runtime_ngpus"]) if mesh_raw.get("runtime_ngpus") is not None else None,
            pipeline_nstages=int(mesh_raw.get("pipeline_nstages", 1)),
            pipeline_size=int(mesh_raw["pipeline_size"]) if mesh_raw.get("pipeline_size") is not None else None,
            pipeline_nmicros=int(mesh_raw.get("pipeline_nmicros", 1)),
            pipeline_scheduler=str(mesh_raw.get("pipeline_scheduler", "1f1b")),
            zero_stage=int(mesh_raw.get("zero_stage", 0)),
            zero_ngroups=int(mesh_raw.get("zero_ngroups", 1)),
        )
        rewrites = RewriteConfig.from_dict(raw.get("rewrites"))
        stage_splits = [
            StageSplitSpec(layers=str(item["layers"]), stage_id=int(item["stage_id"]))
            for item in raw.get("stage_splits", [])
        ]
        rules = [
            PartitionRuleSpec(
                name=str(item.get("name", f"rule_{idx}")),
                fqn_regex=item.get("fqn_regex"),
                op_regex=item.get("op_regex"),
                param_regex=item.get("param_regex"),
                partition=item.get("partition", "auto"),
                on_indivisible=str(item.get("on_indivisible", "error")),
            )
            for idx, item in enumerate(raw.get("rules", []))
        ]
        cfg = cls(
            name=str(raw.get("name", "custom_strategy")),
            description=str(raw.get("description", "")),
            mesh=mesh,
            rewrites=rewrites,
            default_partition=str(raw.get("default_partition", "auto")),
            recompute_modules=[str(m) for m in raw.get("recompute_modules", [])],
            stage_splits=stage_splits,
            rules=rules,
        )
        cfg.validate_mesh()
        return cfg


def load_strategy_config(source: str | Path | dict[str, Any]) -> StrategyConfig:
    if isinstance(source, dict):
        return StrategyConfig.from_dict(source)
    path = Path(source)
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return StrategyConfig.from_dict(data)


# ============================================================================
# Lowering JSON Rules -> `list[OpPlan]` for `nnscaler.policies.fn`
# ============================================================================

_LANG_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def _node_fqn(node: IRFwOperation) -> str:
    return str(getattr(node, "fqn", "") or "")


def _node_sig(node: IRFwOperation) -> str:
    sig = str(getattr(node, "signature", "") or "")
    name = str(getattr(node, "name", "") or "")
    return f"{name}|{sig}"


def _node_param_names(node: IRFwOperation) -> list[str]:
    names: list[str] = []
    for inp in node.inputs():
        if isinstance(inp, IRSubTensor) and inp.is_param():
            names.append(str(getattr(inp.parent, "origin_name", None) or inp.name))
    return names


def _resolve_semantic_partition(
    node: IRFwOperation,
    spec: str | dict[str, int] | None,
) -> OpPartition | None | str:
    """Translate semantic partition shorthands into exact `OpPartition(input, dim)`."""
    if spec is None or spec == "replicate":
        return None
    if spec == "auto":
        return "auto"
    if isinstance(spec, dict):
        return OpPartition(input=int(spec["input"]), dim=int(spec["dim"]))

    sig = str(getattr(node, "signature", "") or "")
    if spec in ("tp_col", "vocab_parallel"):
        return OpPartition(input=1, dim=0)
    if spec == "tp_row":
        return OpPartition(input=1, dim=1)
    if spec == "data_parallel":
        return OpPartition(input=0, dim=0)
    if spec == "expert_parallel":
        if "deepseek_moe_grouped_gemm" in sig:
            return OpPartition(input=3, dim=0)
        if "deepseek_grouped_linear" in sig:
            return OpPartition(input=1, dim=0)
        return OpPartition(input=1, dim=0)
    if spec == "expert_tp_col":
        if "deepseek_grouped_linear" in sig:
            return OpPartition(input=1, dim=1)
        return OpPartition(input=1, dim=0)
    if spec == "expert_tp_row":
        if "deepseek_grouped_linear" in sig:
            return OpPartition(input=1, dim=2)
        return OpPartition(input=1, dim=1)
    if spec == "head_parallel":
        if "deepseek_mla_concat_kv" in sig:
            return OpPartition(input=0, dim=1)
        if "scaled_dot_product_attention" in sig or "matmul" in sig:
            return OpPartition(input=0, dim=1)
        return OpPartition(input=0, dim=1)

    raise StrategyValidationError(f"Unknown partition shorthand {spec!r}")


def _validate_partition_on_node(
    node: IRFwOperation,
    part: OpPartition | None | str,
    tp_size: int,
    rule_name: str,
    on_indivisible: str,
    warnings: list[dict[str, Any]],
) -> OpPartition | None | str:
    if part is None or part == "auto" or tp_size == 1:
        return None if tp_size == 1 and isinstance(part, OpPartition) else part

    if not isinstance(node, IRDimops):
        if on_indivisible == "fallback_replicate":
            warnings.append({"rule": rule_name, "fqn": _node_fqn(node), "reason": "non_dimops_fallback_replicate"})
            return None
        raise StrategyValidationError(
            f"Rule {rule_name!r} requested {part} on non-IRDimops op {node} (fqn={_node_fqn(node)!r})"
        )

    assert isinstance(part, OpPartition)
    inputs = node.inputs()
    if part.input >= len(inputs) or not isinstance(inputs[part.input], IRSubTensor):
        raise StrategyValidationError(
            f"Rule {rule_name!r}: input index {part.input} invalid for op {node.signature} (fqn={_node_fqn(node)!r})"
        )

    in_tensor: IRSubTensor = inputs[part.input]
    if part.dim < 0 or part.dim >= len(in_tensor.shape):
        raise StrategyValidationError(
            f"Rule {rule_name!r}: dim {part.dim} out of range for shape {in_tensor.shape} on {node.signature}"
        )

    dim_len = in_tensor.shape[part.dim]
    algo = node.algorithm("dim")
    axis_legal = bool(algo.satisfy(idx=part.input, dim=part.dim, num=1))
    divisible = bool(axis_legal and algo.satisfy(idx=part.input, dim=part.dim, num=tp_size))

    if axis_legal and divisible:
        return part

    tspace = node.transform_space()
    reason = (
        f"axis ({part.input}, {part.dim}) is locked/non-partitionable (anno={node.anno}, canonical_space={tspace})"
        if not axis_legal
        else f"tensor shape {tuple(in_tensor.shape)} dim {part.dim} (={dim_len}) not divisible by tp_size={tp_size}"
    )
    if on_indivisible in ("fallback_replicate", "fallback_auto"):
        fallback_val: None | str = None if on_indivisible == "fallback_replicate" else "auto"
        warnings.append(
            {
                "rule": rule_name,
                "fqn": _node_fqn(node),
                "signature": str(node.signature),
                "requested": {"input": part.input, "dim": part.dim},
                "fallback": on_indivisible,
                "reason": reason,
            }
        )
        return fallback_val

    raise StrategyValidationError(
        f"Rule {rule_name!r} failed on op {node.signature} (fqn={_node_fqn(node)!r}): {reason}. "
        f"(Set 'on_indivisible': 'fallback_replicate' if replication is intended.)"
    )


def _build_stage_lookup(strategy: StrategyConfig) -> dict[int, int]:
    lang_map: dict[int, int] = {}
    for split in strategy.stage_splits:
        for idx in split.expand_indices():
            lang_map[idx] = split.stage_id
    return lang_map


def _extract_recompute_unit_key(fqn: str, patterns: list[re.Pattern[str]]) -> str | None:
    for pat in patterns:
        m = pat.search(fqn)
        if m:
            return fqn[: m.end()].rstrip(".")
    return None


def compile_op_plans(
    graph: IRGraph,
    strategy: StrategyConfig,
) -> tuple[list[OpPlan], dict[str, Any]]:
    """Compile a `StrategyConfig` onto `graph` forward ops, returning `(op_plans, policy_report)`."""
    strategy.validate_mesh()
    tp_size = strategy.mesh.tp_size
    nstages = strategy.mesh.pipeline_nstages

    lang_stage_map = _build_stage_lookup(strategy)
    compiled_rules = [
        (
            rule,
            re.compile(rule.fqn_regex) if rule.fqn_regex else None,
            re.compile(rule.op_regex) if rule.op_regex else None,
            re.compile(rule.param_regex) if rule.param_regex else None,
        )
        for rule in strategy.rules
    ]
    recompute_pats = [re.compile(pat) for pat in strategy.recompute_modules]

    fw_nodes = graph.select(ntype=IRFwOperation)
    plans: list[OpPlan] = []
    warnings: list[dict[str, Any]] = []
    rule_hit_counts: dict[str, int] = {r.name: 0 for r in strategy.rules}
    partition_counts = {"sharded": 0, "auto": 0, "replicated": 0}

    auto_stage_pivots: set[IRFwOperation] = set()
    if nstages > 1 and not strategy.stage_splits and fw_nodes:
        step = max(1, len(fw_nodes) // nstages)
        for s_idx in range(1, nstages):
            pivot_pos = min(len(fw_nodes) - (nstages - s_idx), s_idx * step)
            auto_stage_pivots.add(fw_nodes[pivot_pos])

    cur_stage = 0
    cur_recompute_key: tuple[int, str] | None = None
    cur_recompute_id = -1
    next_recompute_id = 0

    for node in fw_nodes:
        fqn = _node_fqn(node)
        sig_str = _node_sig(node)
        pnames = _node_param_names(node)

        if node in auto_stage_pivots and cur_stage + 1 < nstages:
            cur_stage += 1
        else:
            m_lang = _LANG_LAYER_RE.search(fqn)
            if m_lang is not None and int(m_lang.group(1)) in lang_stage_map:
                target_stage = lang_stage_map[int(m_lang.group(1))]
                if target_stage > cur_stage:
                    if target_stage > cur_stage + 1:
                        raise StrategyValidationError(
                            f"Stage jump {cur_stage} -> {target_stage} at fqn={fqn!r}; "
                            "pipeline stage splits must cover contiguous stage IDs 0..N-1 in execution order."
                        )
                    cur_stage = target_stage

        unit_prefix = _extract_recompute_unit_key(fqn, recompute_pats) if recompute_pats else None
        if unit_prefix is not None:
            rk = (cur_stage, unit_prefix)
            if rk != cur_recompute_key:
                cur_recompute_key = rk
                cur_recompute_id = next_recompute_id
                next_recompute_id += 1
            recompute_id = cur_recompute_id
        else:
            cur_recompute_key = None
            recompute_id = -1

        chosen_part: OpPartition | None | str = (
            "auto" if (strategy.default_partition == "auto" and tp_size > 1) else None
        )
        for rule, fqn_re, op_re, param_re in compiled_rules:
            if fqn_re is not None and not fqn_re.search(fqn):
                continue
            if op_re is not None and not op_re.search(sig_str):
                continue
            if param_re is not None and not any(param_re.search(pn) for pn in pnames):
                continue

            raw_part = _resolve_semantic_partition(node, rule.partition)
            chosen_part = _validate_partition_on_node(
                node, raw_part, tp_size, rule.name, rule.on_indivisible, warnings
            )
            rule_hit_counts[rule.name] += 1
            break

        if not isinstance(node, IRDimops):
            chosen_part = None

        if isinstance(chosen_part, OpPartition):
            partition_counts["sharded"] += 1
        elif chosen_part == "auto":
            partition_counts["auto"] += 1
        else:
            partition_counts["replicated"] += 1

        plans.append(
            OpPlan(
                op=node,
                stage_id=cur_stage,
                recompute_id=recompute_id,
                partition=chosen_part,
            )
        )

    observed_stages = sorted({p.stage_id for p in plans}) if plans else [0]
    if len(observed_stages) != nstages:
        raise StrategyValidationError(
            f"Strategy declared pipeline_nstages={nstages}, but stage_splits produced stages {observed_stages}. "
            "Adjust 'stage_splits' layer ranges to match the model's num_hidden_layers."
        )

    report = {
        "total_fw_ops": len(fw_nodes),
        "tp_size": tp_size,
        "nstages": nstages,
        "recompute_groups": next_recompute_id,
        "partition_counts": partition_counts,
        "rule_hit_counts": rule_hit_counts,
        "warnings": warnings,
    }
    return plans, report


def build_compute_config(
    strategy: StrategyConfig,
    *,
    inference_only: bool = False,
    trace_strategy: str = "meta",
    weightless: bool = True,
) -> ComputeConfig:
    """Create a validated `nnscaler.parallel.ComputeConfig` from `StrategyConfig`."""
    strategy.validate_mesh()
    m = strategy.mesh
    pas_cfg: dict[str, Any] = {
        "pipeline_nstages": m.pipeline_nstages,
        "pipeline_size": m.effective_pp_size,
        "pipeline_nmicros": m.pipeline_nmicros,
        "pipeline_scheduler": m.pipeline_scheduler,
    }
    use_zero = 1 if m.zero_stage > 0 else 0
    return ComputeConfig(
        plan_ngpus=m.plan_ngpus,
        runtime_ngpus=m.runtime_ngpus or m.plan_ngpus,
        use_zero=use_zero,
        zero_ngroups=m.zero_ngroups,
        use_end2end=not inference_only,
        inference_only=inference_only,
        trace_strategy=trace_strategy,
        pas_config=pas_cfg,
        param_init_strategy=ParamInitStrategy.SHARD if weightless else ParamInitStrategy.FULL,
    )


def build_pas_policy(
    strategy: StrategyConfig,
    hook_sink: dict[str, Any] | None = None,
) -> Callable[[IRGraph, ComputeConfig], IRGraph]:
    """Build a `pas_policy` callable for `nnscaler.parallelize(...)` that records pre/post-PAS IR snapshots."""

    def _pas(graph: IRGraph, cfg: ComputeConfig) -> IRGraph:
        if hook_sink is not None and "pre_pas_hook" in hook_sink:
            hook_sink["pre_pas_hook"](graph)

        plans, policy_report = compile_op_plans(graph, strategy)
        if hook_sink is not None:
            hook_sink["policy_report"] = policy_report

        dist_graph = policies_fn(graph, cfg, lambda _g, _c: plans)

        if hook_sink is not None and "post_pas_hook" in hook_sink:
            hook_sink["post_pas_hook"](dist_graph)
        return dist_graph

    return _pas
