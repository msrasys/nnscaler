# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import ast
from collections import Counter
import json
from pathlib import Path
import re


COLLECTIVE_PATTERN = re.compile(
    r"nnscaler\.runtime\.adapter(?:\.nn)?\.([A-Za-z0-9_]+)"
)
EXPERT_RANGE_PATTERN = re.compile(
    r"local_expert_start=(\d+), local_expert_end=(\d+)"
)
CUSTOM_OP_PATTERN = re.compile(r"examples\.([A-Za-z0-9_\.]+)\(")


def _slice_value(node: ast.AST):
    if isinstance(node, ast.Constant) and node.value is Ellipsis:
        return Ellipsis
    if isinstance(node, ast.Tuple):
        return tuple(_slice_value(element) for element in node.elts)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "slice":
        values = [ast.literal_eval(argument) for argument in node.args]
        values.extend([None] * (3 - len(values)))
        return slice(*values)
    raise ValueError(f"unsupported slice expression: {ast.dump(node)}")


def _full_maps(source: str):
    maps = {}
    for line in source.splitlines():
        if "self.add_full_map(" not in line:
            continue
        call = ast.parse(line.strip()).body[0].value
        local_name = ast.literal_eval(call.args[0])
        is_parameter = ast.literal_eval(call.args[2])
        fqn = ast.literal_eval(call.args[3])
        full_shape = ast.literal_eval(call.args[4])
        slices = _slice_value(call.args[5])
        maps[local_name] = {
            "fqn": fqn,
            "is_parameter": is_parameter,
            "full_shape": full_shape,
            "slices": slices,
        }
    return maps


def _is_full_slice(shape, slices) -> bool:
    if slices is Ellipsis:
        return True
    return all(
        part.start in (None, 0)
        and part.stop == size
        and part.step is None
        for size, part in zip(shape, slices)
    )


def inspect_plan(output_dir: Path):
    scripts = sorted(output_dir.rglob("gencode*.py"))
    if not scripts:
        raise FileNotFoundError(f"no gencode files found below {output_dir}")

    ranks = []
    parameter_sets = set()
    state_sets = set()
    has_parameters = False
    collectives = Counter()
    custom_ops = Counter()
    expert_ranges = set()
    for script in scripts:
        source = script.read_text(encoding="utf-8")
        maps = _full_maps(source)
        parameters = frozenset(
            item["fqn"]
            for item in maps.values()
            if item["is_parameter"]
        )
        has_parameters = has_parameters or bool(parameters)
        parameter_sets.add(parameters)
        state_sets.add(frozenset(item["fqn"] for item in maps.values()))
        sharded_parameters = sorted(
            item["fqn"]
            for item in maps.values()
            if item["is_parameter"]
            and not _is_full_slice(item["full_shape"], item["slices"])
        )
        sharded_buffers = sorted(
            item["fqn"]
            for item in maps.values()
            if not item["is_parameter"]
            if not _is_full_slice(item["full_shape"], item["slices"])
        )
        rank_collectives = Counter(
            name
            for name in COLLECTIVE_PATTERN.findall(source)
            if name != "Reducer"
        )
        collectives.update(rank_collectives)
        rank_custom_ops = Counter(CUSTOM_OP_PATTERN.findall(source))
        custom_ops.update(rank_custom_ops)
        rank_expert_ranges = sorted(
            (int(start), int(end))
            for start, end in EXPERT_RANGE_PATTERN.findall(source)
        )
        expert_ranges.update(rank_expert_ranges)
        ranks.append(
            {
                "file": str(script),
                "line_count": source.count("\n") + 1,
                "sharded_parameter_count": len(sharded_parameters),
                "sharded_parameters": sharded_parameters,
                "sharded_buffer_count": len(sharded_buffers),
                "sharded_buffers": sharded_buffers,
                "collectives": dict(sorted(rank_collectives.items())),
                "custom_ops": dict(sorted(rank_custom_ops.items())),
                "expert_ranges": rank_expert_ranges,
            }
        )

    return {
        "output_dir": str(output_dir.resolve()),
        "rank_count": len(ranks),
        "distinct_parameter_sets": len(parameter_sets),
        "distinct_state_sets": len(state_sets),
        "pipeline_detected": (
            len(parameter_sets) > 1
            if has_parameters
            else len(state_sets) > 1
        ),
        "collectives": dict(sorted(collectives.items())),
        "custom_ops": dict(sorted(custom_ops.items())),
        "expert_ranges": sorted(expert_ranges),
        "ranks": ranks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize nnScaler parameter slices and collectives from generated code."
    )
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = inspect_plan(args.output_dir)
    serialized = json.dumps(report, indent=2)
    if args.output is None:
        print(serialized)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
