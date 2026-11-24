# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Pre-PAS / Post-PAS Static IR Auditor & Structured Verification Bundle Exporter for DeepSeek.

Produces machine-readable inspection bundles (`00_model_and_strategy.json`, `02_ir_pre_pas.txt`,
`02_ir_audit.json`, `03_ir_post_pas.txt`, `03_adapters.json`, `04_generated/gencode*.py`) on a
CPU-only machine with zero GPUs.
"""

from __future__ import annotations

import ast
from collections import Counter
import json
from pathlib import Path
import shutil
from typing import Any

from nnscaler.graph import IRGraph
from nnscaler.graph.function.dimops import IRDimops
from nnscaler.graph.segment import IRSegment
from nnscaler.ir.adapter import IRAdapter
from nnscaler.ir.operator import IRFwOperation
from nnscaler.ir.tensor import IRSubTensor


def _tensor_desc(t: Any) -> dict[str, Any]:
    if isinstance(t, IRSubTensor):
        return {
            "tid": t.parent.tid,
            "name": str(getattr(t.parent, "origin_name", None) or t.name),
            "full_shape": [int(d) if isinstance(d, int) else str(d) for d in t.parent.shape],
            "sub_shape": [int(d) if isinstance(d, int) else str(d) for d in t.shape],
            "indmap": [[int(lo), int(hi)] for lo, hi in t.indmap],
            "valmap": [int(t.valmap[0]), int(t.valmap[1])] if t.valmap is not None else None,
            "dtype": str(t.dtype),
            "is_param": bool(t.is_param()),
            "device": list(t.device) if t.device else [],
        }
    return {"repr": repr(t)}


def audit_pre_pas_graph(graph: IRGraph) -> dict[str, Any]:
    """Inspect a pre-PAS `IRGraph` for opaque ops, dynamic dimensions, frozen CE reductions, and op census."""
    fw_ops = graph.select(ntype=IRFwOperation)
    sig_counter: Counter[str] = Counter()
    opaque_ops: list[dict[str, Any]] = []
    dynamic_tensors: list[dict[str, Any]] = []
    mean_ce_ops: list[dict[str, Any]] = []
    node_records: list[dict[str, Any]] = []

    seen_tids: set[int] = set()
    for idx, op in enumerate(fw_ops):
        sig = str(getattr(op, "signature", op.name))
        fqn = str(getattr(op, "fqn", "") or "")
        sig_counter[sig] += 1

        is_dimops = isinstance(op, IRDimops)
        anno_str = str(op.anno) if is_dimops else None
        tspace = [list(pair) for pair in op.transform_space()] if is_dimops else []

        has_tensor_in = any(isinstance(x, IRSubTensor) for x in op.inputs())
        has_tensor_out = any(isinstance(x, IRSubTensor) for x in op.outputs())
        if not is_dimops and op.name != "multiref" and has_tensor_in and has_tensor_out:
            opaque_ops.append({"idx": idx, "name": op.name, "signature": sig, "fqn": fqn})

        if "cross_entropy" in sig and op.kwargs.get("reduction", "mean") == "mean":
            mean_ce_ops.append({"idx": idx, "signature": sig, "fqn": fqn, "anno": anno_str})

        for t in list(op.inputs()) + list(op.outputs()):
            if isinstance(t, IRSubTensor) and t.parent.tid not in seen_tids:
                seen_tids.add(t.parent.tid)
                if any(not isinstance(d, int) or d < 0 for d in t.parent.shape):
                    dynamic_tensors.append(_tensor_desc(t))

        node_records.append(
            {
                "idx": idx,
                "cid": op.cid,
                "name": op.name,
                "signature": sig,
                "fqn": fqn,
                "is_dimops": is_dimops,
                "anno": anno_str,
                "transform_space": tspace,
                "kwargs": {k: repr(v) for k, v in op.kwargs.items()},
                "inputs": [_tensor_desc(x) for x in op.inputs()],
                "outputs": [_tensor_desc(x) for x in op.outputs()],
            }
        )

    full_params = [t for t in graph.full_tensors() if t.is_param()]
    total_param_elems = sum(
        int(t.nelement()) for t in full_params if all(isinstance(d, int) and d >= 0 for d in t.shape)
    )

    return {
        "summary": {
            "num_fw_ops": len(fw_ops),
            "num_opaque_fw_ops": len(opaque_ops),
            "num_dynamic_tensors": len(dynamic_tensors),
            "num_mean_ce_bottlenecks": len(mean_ce_ops),
            "num_parameters": len(full_params),
            "total_parameter_elements": total_param_elems,
            "audit_passed": len(opaque_ops) == 0 and len(dynamic_tensors) == 0,
        },
        "op_signature_counts": dict(sig_counter.most_common()),
        "opaque_ops": opaque_ops,
        "dynamic_tensors": dynamic_tensors,
        "mean_ce_ops": mean_ce_ops,
        "nodes": node_records,
    }


def audit_post_pas_graph(graph: IRGraph) -> dict[str, Any]:
    """Inspect a post-PAS distributed `IRGraph` for segments, collectives (`IRAdapter`), and sharded ops."""
    adapters = graph.select(ntype=IRAdapter)
    prim_counter: Counter[str] = Counter()
    adapter_records: list[dict[str, Any]] = []

    for idx, adp in enumerate(adapters):
        prims = []
        for prim in adp.prims:
            pname = type(prim).__name__
            prim_counter[pname] += 1
            prims.append(
                {
                    "prim_type": pname,
                    "repr": repr(prim),
                    "device": list(prim.device) if getattr(prim, "device", None) else [],
                }
            )
        adapter_records.append(
            {
                "idx": idx,
                "cid": adp.cid,
                "isfw": bool(adp.isfw()),
                "custom": bool(adp.custom),
                "device": list(adp.device) if adp.device else [],
                "inputs": [_tensor_desc(x) for x in adp.inputs()],
                "outputs": [_tensor_desc(x) for x in adp.outputs()],
                "prims": prims,
            }
        )

    segments = [seg for seg in graph.select(ntype=IRSegment, flatten=False) if seg.isfw()]
    fw_ops = graph.select(ntype=IRFwOperation)
    post_nodes: list[dict[str, Any]] = []
    for idx, op in enumerate(fw_ops):
        post_nodes.append(
            {
                "idx": idx,
                "cid": op.cid,
                "name": op.name,
                "signature": str(getattr(op, "signature", op.name)),
                "fqn": str(getattr(op, "fqn", "") or ""),
                "device": list(op.device) if op.device else [],
                "kwargs": {k: repr(v) for k, v in op.kwargs.items()},
                "inputs": [_tensor_desc(x) for x in op.inputs()],
                "outputs": [_tensor_desc(x) for x in op.outputs()],
            }
        )

    return {
        "summary": {
            "num_fw_segments": len(segments) if segments else 1,
            "num_post_pas_fw_ops": len(fw_ops),
            "num_adapters_total": len(adapters),
            "num_fw_adapters": sum(1 for a in adapters if a.isfw()),
            "num_bw_adapters": sum(1 for a in adapters if not a.isfw()),
            "primitive_counts": dict(prim_counter.most_common()),
        },
        "adapters": adapter_records,
        "post_pas_fw_nodes": post_nodes,
    }


def _format_ir_text(audit_nodes: list[dict[str, Any]], header: str) -> str:
    lines = [f"# {header}", f"# total_nodes: {len(audit_nodes)}", ""]
    for n in audit_nodes:
        devs = f" devs={n['device']}" if "device" in n else ""
        anno = f" anno=[{n['anno']}]" if n.get("anno") else ""
        in_shapes = ", ".join(str(x.get("sub_shape", x.get("repr"))) for x in n["inputs"])
        out_shapes = ", ".join(str(x.get("sub_shape", x.get("repr"))) for x in n["outputs"])
        lines.append(
            f"[{n['idx']:04d}] {n['signature']}{anno}{devs} | ({in_shapes}) -> ({out_shapes}) | fqn={n['fqn']}"
        )
    return "\n".join(lines) + "\n"


def _count_adapter_calls(tree: ast.AST) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            cur: ast.AST = node.func.value
            parts: list[str] = [node.func.attr]
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                parts.append(cur.id)
            dotted = ".".join(reversed(parts))
            if dotted.startswith("nnscaler.runtime.adapter."):
                counts[dotted.removeprefix("nnscaler.runtime.adapter.")] += 1
    return dict(counts.most_common())


def validate_generated_code(gencode_dir: Path, expected_ranks: int) -> dict[str, Any]:
    """Parse every `gencode{rank}.py` with `ast.parse` and extract per-rank collective telemetry."""
    files = sorted(gencode_dir.glob("gencode*.py"))
    parsed_files: list[dict[str, Any]] = []
    total_collectives: Counter[str] = Counter()
    for fpath in files:
        src = fpath.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(fpath))
        coll = _count_adapter_calls(tree)
        total_collectives.update(coll)
        parsed_files.append(
            {
                "file": fpath.name,
                "lines": src.count("\n") + 1,
                "ast_body_statements": len(tree.body),
                "collectives": coll,
            }
        )
    return {
        "expected_ranks": expected_ranks,
        "found_gencode_files": len(files),
        "all_ranks_present": len(files) >= expected_ranks,
        "total_collective_calls_across_ranks": dict(total_collectives.most_common()),
        "files": parsed_files,
    }


def export_verification_bundle(
    out_dir: Path,
    *,
    model_meta: dict[str, Any],
    model_config_dict: dict[str, Any],
    strategy_dict: dict[str, Any],
    policy_report: dict[str, Any],
    pre_pas_audit: dict[str, Any],
    post_pas_audit: dict[str, Any],
    nnscaler_gen_dir: Path,
    plan_ngpus: int,
) -> dict[str, Any]:
    """Write the standardized verification artifact bundle (`00_*` .. `04_generated/`)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    gen_out = out_dir / "04_generated"
    gen_out.mkdir(parents=True, exist_ok=True)

    copied_files: list[str] = []
    for src_file in sorted(nnscaler_gen_dir.rglob("*")):
        if src_file.is_file() and src_file.suffix in (".py", ".pt", ".json"):
            dst = gen_out / src_file.name
            if src_file.resolve() != dst.resolve():
                shutil.copy2(src_file, dst)
            copied_files.append(dst.name)

    codegen_check = validate_generated_code(gen_out, expected_ranks=plan_ngpus)

    bundle_manifest = {
        "model_meta": model_meta,
        "model_config": model_config_dict,
        "strategy": strategy_dict,
        "policy_report": policy_report,
        "pre_pas_summary": pre_pas_audit["summary"],
        "post_pas_summary": post_pas_audit["summary"],
        "codegen_validation": codegen_check,
        "copied_artifacts": copied_files,
    }

    (out_dir / "00_model_and_strategy.json").write_text(
        json.dumps(bundle_manifest, indent=2), encoding="utf-8"
    )
    (out_dir / "02_ir_pre_pas.txt").write_text(
        _format_ir_text(pre_pas_audit["nodes"], "Pre-PAS Logical IRGraph"), encoding="utf-8"
    )
    (out_dir / "02_ir_audit.json").write_text(
        json.dumps(pre_pas_audit, indent=2), encoding="utf-8"
    )
    (out_dir / "03_ir_post_pas.txt").write_text(
        _format_ir_text(post_pas_audit["post_pas_fw_nodes"], "Post-PAS Distributed IRGraph"), encoding="utf-8"
    )
    (out_dir / "03_adapters.json").write_text(
        json.dumps(post_pas_audit, indent=2), encoding="utf-8"
    )
    return bundle_manifest
