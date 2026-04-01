#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""
Unified constraint specification for the autodist parallelization solver.

This module provides an expressive constraint API (V2) that feeds into the
autodist solver, replacing both the YAML-only path and the fully-manual OpPlan
path with a middle ground: users can pin some operators and let the solver
optimize the rest.

The legacy ``PartitionConstraint`` in ``descs.py`` is preserved for backward
compatibility.  New code should prefer :class:`PartitionConstraintV2` and
:class:`ConstraintSet`.
"""

from __future__ import annotations

import re
import math
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any, Set

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ConstraintValidationError(Exception):
    """Raised when constraint validation fails."""
    pass


# ---------------------------------------------------------------------------
# Partition constraint (V2)
# ---------------------------------------------------------------------------

@dataclass
class PartitionConstraintV2:
    """Unified partition constraint that targets operators by multiple criteria.

    **Matching criteria** (AND logic — all specified criteria must match):

    - ``op_name``:  exact operator signature
    - ``module_class``:  module class-name suffix match
    - ``layer_range``:  half-open ``[start, end)`` layer-index range
    - ``param_name_pattern``:  regex on parameter fully-qualified names

    **Partition constraints:**

    - ``allowed_dims``:  ``(input_idx, dim_idx)`` pairs allowed for partitioning
    - ``forbidden_dims``:  ``(input_idx, dim_idx)`` pairs forbidden
    - ``force_replicate``:  operator must be replicated
    - ``max_partition_degree``:  maximum TP degree (caps the split factor)

    **Assignment constraints:**

    - ``stage_id``:  pin to a specific pipeline stage

    **Recompute constraints:**

    - ``recompute``:  ``True`` / ``False`` / ``None`` (solver decides)
    """

    # -- Matching criteria --------------------------------------------------
    op_name: Optional[str] = None
    module_class: Optional[str] = None
    layer_range: Optional[Tuple[int, int]] = None
    param_name_pattern: Optional[str] = None

    # -- Partition constraints ----------------------------------------------
    allowed_dims: Optional[List[Tuple[int, int]]] = None
    forbidden_dims: Optional[List[Tuple[int, int]]] = None
    force_replicate: bool = False
    max_partition_degree: Optional[int] = None

    # -- Assignment constraints ---------------------------------------------
    stage_id: Optional[int] = None

    # -- Recompute constraints ----------------------------------------------
    recompute: Optional[bool] = None

    # -- Internal bookkeeping (not serialized) ------------------------------
    _matched: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self):
        self.validate()

    # ---- validation -------------------------------------------------------

    def validate(self) -> None:
        """Raise :class:`ConstraintValidationError` on internal conflicts."""
        errors: List[str] = []

        if not any([
            self.op_name is not None,
            self.module_class is not None,
            self.layer_range is not None,
            self.param_name_pattern is not None,
        ]):
            errors.append(
                'Constraint must specify at least one matching criterion '
                '(op_name, module_class, layer_range, or param_name_pattern)'
            )

        if self.force_replicate and self.allowed_dims:
            errors.append(
                'force_replicate=True conflicts with allowed_dims '
                '(replicated ops cannot be partitioned)'
            )

        if self.allowed_dims and self.forbidden_dims:
            overlap = set(self.allowed_dims) & set(self.forbidden_dims)
            if overlap:
                errors.append(
                    f'allowed_dims and forbidden_dims overlap: {overlap}'
                )

        if self.max_partition_degree is not None and self.max_partition_degree < 1:
            errors.append(
                f'max_partition_degree must be >= 1, got {self.max_partition_degree}'
            )

        if self.layer_range is not None:
            if len(self.layer_range) != 2:
                errors.append(
                    f'layer_range must be a 2-tuple, got {self.layer_range}'
                )
            elif self.layer_range[0] >= self.layer_range[1]:
                errors.append(
                    f'layer_range start must be < end, got {self.layer_range}'
                )

        if self.param_name_pattern is not None:
            try:
                re.compile(self.param_name_pattern)
            except re.error as exc:
                errors.append(f'Invalid param_name_pattern regex: {exc}')

        if errors:
            raise ConstraintValidationError(
                f'Invalid constraint: {"; ".join(errors)}'
            )

    # ---- matching ---------------------------------------------------------

    def mark_matched(self) -> None:
        """Mark this constraint as having matched at least one operator."""
        self._matched = True

    @property
    def is_matched(self) -> bool:
        return self._matched

    def matches_operator(
        self,
        op_name: str,
        module_stack: Dict[str, Any],
        layer_index: Optional[int] = None,
        param_fqns: Optional[List[str]] = None,
    ) -> bool:
        """Return *True* if **all** specified criteria match the operator."""

        if self.op_name is not None and self.op_name != op_name:
            return False

        if self.module_class is not None:
            nested = '.'.join(
                mt.__name__ for _, mt in module_stack.items()
            ) if module_stack else ''
            if self.module_class not in nested:
                return False

        if self.layer_range is not None:
            if layer_index is None:
                return False
            if not (self.layer_range[0] <= layer_index < self.layer_range[1]):
                return False

        if self.param_name_pattern is not None:
            if not param_fqns:
                return False
            pat = re.compile(self.param_name_pattern)
            if not any(pat.search(fqn) for fqn in param_fqns):
                return False

        return True

    # ---- YAML serialization -----------------------------------------------

    @staticmethod
    def from_yaml(content: Dict[str, Any]) -> 'PartitionConstraintV2':
        """Deserialize from a YAML-compatible dict."""

        def _parse_dim(s: str) -> Tuple[int, int]:
            a, b = s.split(',')
            return (int(a.strip()), int(b.strip()))

        kw: Dict[str, Any] = {}

        if 'op_name' in content:
            kw['op_name'] = content['op_name']
        if 'module_class' in content:
            kw['module_class'] = content['module_class']
        if 'layer_range' in content:
            lr = content['layer_range']
            kw['layer_range'] = (int(lr[0]), int(lr[1]))
        if 'param_name_pattern' in content:
            kw['param_name_pattern'] = content['param_name_pattern']

        if 'allowed_dims' in content:
            kw['allowed_dims'] = [_parse_dim(d) for d in content['allowed_dims']]
        if 'forbidden_dims' in content:
            kw['forbidden_dims'] = [_parse_dim(d) for d in content['forbidden_dims']]
        if 'force_replicate' in content:
            kw['force_replicate'] = bool(content['force_replicate'])
        if 'max_partition_degree' in content:
            kw['max_partition_degree'] = int(content['max_partition_degree'])

        if 'stage_id' in content:
            kw['stage_id'] = int(content['stage_id'])
        if 'recompute' in content:
            kw['recompute'] = bool(content['recompute'])

        return PartitionConstraintV2(**kw)

    def to_yaml(self) -> Dict[str, Any]:
        """Serialize to a YAML-compatible dict."""
        out: Dict[str, Any] = {}

        if self.op_name is not None:
            out['op_name'] = self.op_name
        if self.module_class is not None:
            out['module_class'] = self.module_class
        if self.layer_range is not None:
            out['layer_range'] = list(self.layer_range)
        if self.param_name_pattern is not None:
            out['param_name_pattern'] = self.param_name_pattern

        if self.allowed_dims is not None:
            out['allowed_dims'] = [f'{d[0]},{d[1]}' for d in self.allowed_dims]
        if self.forbidden_dims is not None:
            out['forbidden_dims'] = [f'{d[0]},{d[1]}' for d in self.forbidden_dims]
        if self.force_replicate:
            out['force_replicate'] = True
        if self.max_partition_degree is not None:
            out['max_partition_degree'] = self.max_partition_degree

        if self.stage_id is not None:
            out['stage_id'] = self.stage_id
        if self.recompute is not None:
            out['recompute'] = self.recompute

        return out

    def __hash__(self):
        return hash((
            self.op_name, self.module_class, self.layer_range,
            self.param_name_pattern,
            tuple(self.allowed_dims) if self.allowed_dims else None,
            tuple(self.forbidden_dims) if self.forbidden_dims else None,
            self.force_replicate, self.max_partition_degree,
            self.stage_id, self.recompute,
        ))


# ---------------------------------------------------------------------------
# Scheduler constraint
# ---------------------------------------------------------------------------

VALID_SCHEDULERS = frozenset({
    '1f1b', '1f1b_plus', '1f1b_interleaved',
    'gpipe', 'chimera_direct', 'infer_pipe',
})


@dataclass
class SchedulerConstraint:
    """Constraints on pipeline scheduler selection."""

    allowed_schedulers: Optional[List[str]] = None
    max_bubble_ratio: float = 0.2
    max_stages: Optional[int] = None
    min_microbatches: Optional[int] = None
    prefer_interleaved: bool = False

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        errors: List[str] = []

        if self.allowed_schedulers is not None:
            invalid = set(self.allowed_schedulers) - VALID_SCHEDULERS
            if invalid:
                errors.append(
                    f'Unknown schedulers: {invalid}. '
                    f'Valid: {sorted(VALID_SCHEDULERS)}'
                )

        if not (0 < self.max_bubble_ratio < 1):
            errors.append(
                f'max_bubble_ratio must be in (0, 1), got {self.max_bubble_ratio}'
            )

        if self.max_stages is not None and self.max_stages < 1:
            errors.append(f'max_stages must be >= 1, got {self.max_stages}')

        if self.min_microbatches is not None and self.min_microbatches < 1:
            errors.append(
                f'min_microbatches must be >= 1, got {self.min_microbatches}'
            )

        if errors:
            raise ConstraintValidationError(
                f'Invalid scheduler constraint: {"; ".join(errors)}'
            )

    @staticmethod
    def from_yaml(content: Dict[str, Any]) -> 'SchedulerConstraint':
        kw: Dict[str, Any] = {}
        if 'allowed_schedulers' in content:
            kw['allowed_schedulers'] = content['allowed_schedulers']
        if 'max_bubble_ratio' in content:
            kw['max_bubble_ratio'] = float(content['max_bubble_ratio'])
        if 'max_stages' in content:
            kw['max_stages'] = int(content['max_stages'])
        if 'min_microbatches' in content:
            kw['min_microbatches'] = int(content['min_microbatches'])
        if 'prefer_interleaved' in content:
            kw['prefer_interleaved'] = bool(content['prefer_interleaved'])
        return SchedulerConstraint(**kw)

    def to_yaml(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if self.allowed_schedulers is not None:
            out['allowed_schedulers'] = self.allowed_schedulers
        out['max_bubble_ratio'] = self.max_bubble_ratio
        if self.max_stages is not None:
            out['max_stages'] = self.max_stages
        if self.min_microbatches is not None:
            out['min_microbatches'] = self.min_microbatches
        if self.prefer_interleaved:
            out['prefer_interleaved'] = True
        return out


# ---------------------------------------------------------------------------
# Constraint set
# ---------------------------------------------------------------------------

@dataclass
class ConstraintSet:
    """A validated collection of partition and scheduler constraints."""

    partition_constraints: List[PartitionConstraintV2] = field(default_factory=list)
    scheduler_constraint: Optional[SchedulerConstraint] = None

    def validate_all(self) -> None:
        """Validate every contained constraint and check cross-constraint conflicts."""
        errors: List[str] = []
        for i, pc in enumerate(self.partition_constraints):
            try:
                pc.validate()
            except ConstraintValidationError as exc:
                errors.append(f'Partition constraint [{i}]: {exc}')
        if self.scheduler_constraint:
            try:
                self.scheduler_constraint.validate()
            except ConstraintValidationError as exc:
                errors.append(f'Scheduler constraint: {exc}')
        if errors:
            raise ConstraintValidationError(
                'Constraint set validation failed:\n' + '\n'.join(errors)
            )

    def get_matching_constraints(
        self,
        op_name: str,
        module_stack: Dict[str, Any],
        layer_index: Optional[int] = None,
        param_fqns: Optional[List[str]] = None,
    ) -> List[PartitionConstraintV2]:
        """Return all partition constraints that match the given operator."""
        matches: List[PartitionConstraintV2] = []
        for pc in self.partition_constraints:
            if pc.matches_operator(op_name, module_stack, layer_index, param_fqns):
                pc.mark_matched()
                matches.append(pc)
        return matches

    def get_unmatched_constraints(self) -> List[PartitionConstraintV2]:
        return [pc for pc in self.partition_constraints if not pc.is_matched]

    def get_coverage_report(self, total_ops: int) -> Dict[str, Any]:
        matched = sum(1 for pc in self.partition_constraints if pc.is_matched)
        return {
            'total_constraints': len(self.partition_constraints),
            'matched_constraints': matched,
            'unmatched_constraints': len(self.partition_constraints) - matched,
            'total_operators': total_ops,
            'has_scheduler_constraint': self.scheduler_constraint is not None,
        }

    # ---- YAML I/O ---------------------------------------------------------

    @staticmethod
    def from_yaml(content: Dict[str, Any]) -> 'ConstraintSet':
        pcs: List[PartitionConstraintV2] = []
        if 'partition_constraints' in content:
            for item in content['partition_constraints']:
                pcs.append(PartitionConstraintV2.from_yaml(item))

        sc = None
        if 'scheduler_constraint' in content:
            sc = SchedulerConstraint.from_yaml(content['scheduler_constraint'])

        cs = ConstraintSet(partition_constraints=pcs, scheduler_constraint=sc)
        cs.validate_all()
        return cs

    @staticmethod
    def from_yaml_file(path: str) -> 'ConstraintSet':
        import yaml as _yaml
        from pathlib import Path as _Path

        p = _Path(path)
        if not p.exists():
            raise FileNotFoundError(f'Constraint file not found: {path}')
        with open(p, 'r') as fh:
            content = _yaml.safe_load(fh)
        return ConstraintSet.from_yaml(content)

    def to_yaml(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if self.partition_constraints:
            out['partition_constraints'] = [
                pc.to_yaml() for pc in self.partition_constraints
            ]
        if self.scheduler_constraint:
            out['scheduler_constraint'] = self.scheduler_constraint.to_yaml()
        return out


# ---------------------------------------------------------------------------
# Legacy conversion helper
# ---------------------------------------------------------------------------

def convert_legacy_constraints(
    legacy_list: List[Dict[str, Any]],
) -> ConstraintSet:
    """Convert legacy YAML format (list of dicts with *name*, *parent_module*,
    *allowed_partition_dims*, *replica_allowed*) into a :class:`ConstraintSet`.

    Existing YAML files continue to work without modification.
    """

    def _parse_legacy_dim(s: str) -> Tuple[int, int]:
        a, b = s.split(',')
        return (int(a.strip()), int(b.strip()))

    pcs: List[PartitionConstraintV2] = []
    for lc in legacy_list:
        kw: Dict[str, Any] = {'op_name': lc['name']}

        parent = lc.get('parent_module', '')
        if parent:
            kw['module_class'] = parent

        if 'allowed_partition_dims' in lc:
            kw['allowed_dims'] = [
                _parse_legacy_dim(d) for d in lc['allowed_partition_dims']
            ]

        replica_allowed = lc.get('replica_allowed', True)
        if not replica_allowed:
            # Legacy semantics: replica_allowed=False means the op *must*
            # be partitioned along one of the allowed dims.
            pass  # Handled downstream in the solver filtering.

        pcs.append(PartitionConstraintV2(**kw))

    return ConstraintSet(partition_constraints=pcs)
