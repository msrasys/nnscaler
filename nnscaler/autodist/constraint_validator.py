#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""
Pre-solve constraint validation engine.

Run :func:`validate_constraints` before the SPMD/pipeline solver to get
actionable diagnostics: coverage reports, search-space estimates, and
early error detection for constraints that eliminate all valid partitions.
"""

from __future__ import annotations

import re
import math
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, TYPE_CHECKING

from .constraints import (
    ConstraintSet,
    PartitionConstraintV2,
    ConstraintValidationError,
)

if TYPE_CHECKING:
    from .model_graph import ModelGraph

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Outcome of :func:`validate_constraints`."""

    is_valid: bool
    errors: List[str]
    warnings: List[str]
    coverage: Dict[str, Any]
    search_space_estimate: Optional[Dict[str, Any]] = None

    def summary(self) -> str:
        lines = [
            f'Constraint Validation: {"PASSED" if self.is_valid else "FAILED"}',
            f'  Total constraints: {self.coverage.get("total_constraints", 0)}',
            f'  Matched: {self.coverage.get("matched_constraints", 0)}',
            f'  Unmatched: {self.coverage.get("unmatched_constraints", 0)}',
            f'  Total operators: {self.coverage.get("total_operators", 0)}',
            f'  Constrained operators: {self.coverage.get("constrained_operators", 0)}',
            f'  Unconstrained operators: {self.coverage.get("unconstrained_operators", 0)}',
        ]
        if self.search_space_estimate:
            rf = self.search_space_estimate.get('reduction_factor')
            if rf is not None:
                lines.append(f'  Search-space reduction factor: {rf:.1f}x')
        if self.errors:
            lines.append('  Errors:')
            for e in self.errors:
                lines.append(f'    - {e}')
        if self.warnings:
            lines.append('  Warnings:')
            for w in self.warnings:
                lines.append(f'    - {w}')
        return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LAYER_INDEX_RE = re.compile(r'(?:layers?|blocks?)\.(\d+)')


def _extract_layer_index(module_stack: Dict[str, Any]) -> Optional[int]:
    """Try to infer a layer index from keys like ``model.layers.12``."""
    for key in module_stack:
        m = _LAYER_INDEX_RE.search(key)
        if m:
            return int(m.group(1))
    return None


def _safe_log(x: float) -> float:
    return math.log(max(x, 1))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_constraints(
    constraint_set: ConstraintSet,
    model_graph: 'ModelGraph',
    device_num: int,
) -> ValidationResult:
    """Validate *constraint_set* against *model_graph* before solving.

    Checks performed:

    1. All constraints match at least one operator (warn on unmatched).
    2. Stage-assignment constraints are within valid range.
    3. Search-space estimation with / without constraints.
    """
    from nnscaler.ir import IRTensor
    from nnscaler.graph.function.dimops import IRDimops

    errors: List[str] = []
    warnings: List[str] = []

    # ---- match constraints against operators -----------------------------
    op_constraint_map: Dict[int, List[PartitionConstraintV2]] = {}

    for i, operator in enumerate(model_graph.operator_list):
        op_name: str = operator.op_name
        ir_cell = operator.ir_cell

        module_stack: Dict[str, Any] = (
            ir_cell.module_stack if isinstance(ir_cell, IRDimops) else {}
        )

        param_fqns: List[str] = []
        for item in ir_cell.inputs():
            if isinstance(item, IRTensor) and item.is_param():
                if hasattr(item, 'name'):
                    param_fqns.append(item.name)

        layer_index = _extract_layer_index(module_stack)

        matches = constraint_set.get_matching_constraints(
            op_name, module_stack, layer_index, param_fqns,
        )
        if matches:
            op_constraint_map[i] = matches

    # ---- unmatched -------------------------------------------------------
    for pc in constraint_set.get_unmatched_constraints():
        warnings.append(f'Constraint did not match any operator: {pc}')

    # ---- stage_id sanity -------------------------------------------------
    for pc in constraint_set.partition_constraints:
        if pc.stage_id is not None and pc.stage_id < 0:
            errors.append(f'Invalid stage_id {pc.stage_id} in constraint: {pc}')

    # ---- search-space estimate -------------------------------------------
    search_est = _estimate_search_space(
        model_graph, op_constraint_map, device_num,
    )

    # ---- coverage report -------------------------------------------------
    coverage = constraint_set.get_coverage_report(len(model_graph.operator_list))
    coverage['constrained_operators'] = len(op_constraint_map)
    coverage['unconstrained_operators'] = (
        len(model_graph.operator_list) - len(op_constraint_map)
    )

    result = ValidationResult(
        is_valid=(len(errors) == 0),
        errors=errors,
        warnings=warnings,
        coverage=coverage,
        search_space_estimate=search_est,
    )
    _logger.info(result.summary())
    return result


def _estimate_search_space(
    model_graph: 'ModelGraph',
    op_constraint_map: Dict[int, List[PartitionConstraintV2]],
    device_num: int,
) -> Dict[str, Any]:
    from nnscaler.graph.function.dimops import IRDimops

    unconstrained_log = 0.0
    constrained_log = 0.0

    for i, operator in enumerate(model_graph.operator_list):
        if isinstance(operator.ir_cell, IRDimops):
            n_dims = len(operator.parallelable_dims) + 1  # +1 for replicate
        else:
            n_dims = 1

        unconstrained_log += _safe_log(n_dims)

        if i in op_constraint_map:
            effective = n_dims
            for c in op_constraint_map[i]:
                if c.force_replicate:
                    effective = 1
                    break
                if c.forbid_replicate:
                    # Remove replicate option (the +1 above)
                    effective = max(1, effective - 1)
                if c.allowed_dims is not None:
                    effective = min(effective, len(c.allowed_dims) + (0 if c.forbid_replicate else 1))
                if c.forbidden_dims is not None:
                    effective = max(1, effective - len(c.forbidden_dims))
                if c.max_partition_degree is not None:
                    # Caps the number of valid partition strategies
                    # Each dim can be split into factors of device_num up to max_partition_degree
                    # Approximation: reduces effective choices
                    effective = max(1, min(effective, c.max_partition_degree))
            constrained_log += _safe_log(effective)
        else:
            constrained_log += _safe_log(n_dims)

    diff = unconstrained_log - constrained_log
    reduction = math.exp(diff) if diff < 700 else float('inf')

    return {
        'unconstrained_log': unconstrained_log,
        'constrained_log': constrained_log,
        'reduction_factor': reduction,
    }
