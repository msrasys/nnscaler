# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Static pre-PAS, post-PAS, and AST codegen auditors for GLM adaptation."""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path
from typing import Any

from nnscaler.ir.adapter import IRAdapter
from nnscaler.ir.cten import IRCell
from nnscaler.ir.operator import IRFwOperation
from nnscaler.ir.tensor import IRFullTensor


def _is_dynamic_tensor(tensor: Any) -> bool:
    return isinstance(tensor, IRFullTensor) and any(not isinstance(d, int) for d in tensor.shape)


def _format_cell_line(cell: IRCell, idx: int) -> str:
    fqn = ""
    if getattr(cell, "module_stack", None):
        fqn = str(next(reversed(cell.module_stack)))
    anno = getattr(cell, "anno", "")
    in_shapes = [tuple(t.shape) if hasattr(t, "shape") else type(t).__name__ for t in cell.inputs()]
    out_shapes = [tuple(t.shape) if hasattr(t, "shape") else type(t).__name__ for t in cell.outputs()]
    return (
        f"[{idx:04d}] {type(cell).__name__:<14} sig={cell.signature:<46} "
        f"fqn={fqn or '-':<48} anno={str(anno)!r} in={in_shapes} -> out={out_shapes}"
    )


def dump_ir_text(graph: Any, out_path: str | Path) -> str:
    lines = [
        f"IRGraph: {getattr(graph, 'name', 'glm')}",
        f"Total nodes: {len(graph.nodes())}",
        "=" * 100,
    ]
    for idx, node in enumerate(graph.nodes()):
        lines.append(_format_cell_line(node, idx))
    text = "\n".join(lines) + "\n"
    Path(out_path).write_text(text, encoding="utf-8")
    return text


def audit_pre_pas(graph: Any) -> dict[str, Any]:
    fw_ops = graph.select(ntype=IRFwOperation)
    op_counts: Counter[str] = Counter()
    opaque_fw_ops: list[dict[str, Any]] = []
    dynamic_tensors: list[dict[str, Any]] = []
    mean_ce_bottlenecks: list[dict[str, Any]] = []

    for node in fw_ops:
        op_counts[node.signature] += 1
        is_dimop = hasattr(node, "anno")
        has_tensor_out = any(isinstance(o, IRFullTensor) for o in node.outputs())
        if not is_dimop and has_tensor_out:
            opaque_fw_ops.append({"cid": node.cid, "signature": node.signature, "comment": str(node.comment)})
        for t in list(node.inputs()) + list(node.outputs()):
            if _is_dynamic_tensor(t):
                dynamic_tensors.append({"cid": node.cid, "signature": node.signature, "shape": list(t.shape)})
        if "cross_entropy" in node.signature:
            anno_str = str(getattr(node, "anno", ""))
            if "N^ C^" in anno_str or node.kwargs.get("reduction", "mean") == "mean":
                mean_ce_bottlenecks.append({"cid": node.cid, "signature": node.signature, "anno": anno_str})

    params = graph.attributes()
    total_param_elems = 0
    for p in params:
        if isinstance(p, IRFullTensor):
            prod = 1
            for d in p.shape:
                prod *= int(d)
            total_param_elems += prod

    summary = {
        "num_fw_ops": len(fw_ops),
        "num_opaque_fw_ops": len(opaque_fw_ops),
        "num_dynamic_tensors": len(dynamic_tensors),
        "num_mean_ce_bottlenecks": len(mean_ce_bottlenecks),
        "num_parameters": len(params),
        "total_parameter_elements": total_param_elems,
        "audit_passed": (
            len(opaque_fw_ops) == 0
            and len(dynamic_tensors) == 0
            and len(mean_ce_bottlenecks) == 0
        ),
    }
    return {
        "summary": summary,
        "op_histogram": dict(op_counts.most_common()),
        "opaque_fw_ops": opaque_fw_ops,
        "dynamic_tensors": dynamic_tensors,
        "mean_ce_bottlenecks": mean_ce_bottlenecks,
    }


def audit_post_pas(graph: Any) -> dict[str, Any]:
    adapters = graph.select(ntype=IRAdapter)
    prim_counts: Counter[str] = Counter()
    adapter_entries: list[dict[str, Any]] = []
    fw_adapters = 0
    bw_adapters = 0
    for ad in adapters:
        if ad.isfw():
            fw_adapters += 1
        else:
            bw_adapters += 1
        prims = [type(p).__name__ for p in ad.prims]
        for p in prims:
            prim_counts[p] += 1
        adapter_entries.append({
            "cid": ad.cid,
            "isfw": ad.isfw(),
            "device": list(ad.device),
            "prims": prims,
        })
    segments = [s for s in graph.nodes() if type(s).__name__ == "IRSegment"]
    fw_segments = [s for s in segments if s.isfw()]
    return {
        "summary": {
            "num_fw_segments": len(fw_segments),
            "num_post_pas_fw_ops": len(graph.select(ntype=IRFwOperation)),
            "num_adapters_total": len(adapters),
            "num_fw_adapters": fw_adapters,
            "num_bw_adapters": bw_adapters,
            "primitive_counts": dict(prim_counts),
        },
        "adapters": adapter_entries,
    }


def audit_codegen_dir(gen_dir: str | Path, expected_ranks: int) -> dict[str, Any]:
    gen_path = Path(gen_dir)
    files = sorted(gen_path.rglob("gencode*.py"), key=lambda p: int(p.stem.replace("gencode", "")))
    per_file: list[dict[str, Any]] = []
    total_collectives: Counter[str] = Counter()
    for fp in files:
        src = fp.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(fp))
        col_counts: Counter[str] = Counter()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func
                name = ""
                if isinstance(fn, ast.Attribute):
                    parts = [fn.attr]
                    cur = fn.value
                    while isinstance(cur, ast.Attribute):
                        parts.append(cur.attr)
                        cur = cur.value
                    if isinstance(cur, ast.Name):
                        parts.append(cur.id)
                    name = ".".join(reversed(parts))
                elif isinstance(fn, ast.Name):
                    name = fn.id
                if any(k in name for k in ("all_reduce", "all_gather", "reduce_scatter", "allreduce", "allgather", "reducescatter", "all_to_all", "send", "recv", "chunk")):
                    short = name.replace("nnscaler.runtime.adapter.", "")
                    col_counts[short] += 1
                    total_collectives[short] += 1
        per_file.append({
            "file": fp.name,
            "lines": len(src.splitlines()),
            "ast_body_statements": len(tree.body),
            "collectives": dict(col_counts),
        })
    return {
        "expected_ranks": expected_ranks,
        "found_gencode_files": len(files),
        "all_ranks_present": len(files) == expected_ranks,
        "total_collective_calls_across_ranks": dict(total_collectives),
        "files": per_file,
    }
