#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""
Tests for the V2 constraint specification layer.
"""

import pytest
import yaml
import tempfile
import os
from pathlib import Path

from nnscaler.autodist.constraints import (
    PartitionConstraintV2,
    SchedulerConstraint,
    ConstraintSet,
    ConstraintValidationError,
    convert_legacy_constraints,
    VALID_SCHEDULERS,
)
from nnscaler.autodist.constraint_validator import (
    ValidationResult,
    _extract_layer_index,
)


# =========================================================================
# PartitionConstraintV2 — validation
# =========================================================================

class TestPartitionConstraintV2Validation:

    def test_valid_constraint_op_name(self):
        pc = PartitionConstraintV2(op_name='torch.nn.functional.linear')
        assert pc.op_name == 'torch.nn.functional.linear'

    def test_valid_constraint_module_class(self):
        pc = PartitionConstraintV2(module_class='LlamaAttention')
        assert pc.module_class == 'LlamaAttention'

    def test_valid_constraint_layer_range(self):
        pc = PartitionConstraintV2(
            op_name='linear', layer_range=(0, 16),
        )
        assert pc.layer_range == (0, 16)

    def test_valid_constraint_all_fields(self):
        pc = PartitionConstraintV2(
            op_name='linear',
            module_class='Attention',
            layer_range=(0, 8),
            param_name_pattern=r'.*weight.*',
            allowed_dims=[(0, 2)],
            max_partition_degree=4,
            stage_id=0,
            recompute=True,
        )
        assert pc.max_partition_degree == 4

    def test_missing_matching_criterion_raises(self):
        with pytest.raises(ConstraintValidationError, match='at least one matching'):
            PartitionConstraintV2(force_replicate=True)

    def test_force_replicate_with_allowed_dims_raises(self):
        with pytest.raises(ConstraintValidationError, match='force_replicate.*conflicts'):
            PartitionConstraintV2(
                op_name='linear',
                force_replicate=True,
                allowed_dims=[(0, 1)],
            )

    def test_allowed_forbidden_overlap_raises(self):
        with pytest.raises(ConstraintValidationError, match='overlap'):
            PartitionConstraintV2(
                op_name='linear',
                allowed_dims=[(0, 1), (0, 2)],
                forbidden_dims=[(0, 2)],
            )

    def test_invalid_max_partition_degree_raises(self):
        with pytest.raises(ConstraintValidationError, match='max_partition_degree'):
            PartitionConstraintV2(
                op_name='linear',
                max_partition_degree=0,
            )

    def test_invalid_layer_range_order_raises(self):
        with pytest.raises(ConstraintValidationError, match='layer_range start'):
            PartitionConstraintV2(
                op_name='linear',
                layer_range=(10, 5),
            )

    def test_invalid_regex_raises(self):
        with pytest.raises(ConstraintValidationError, match='Invalid param_name_pattern'):
            PartitionConstraintV2(
                op_name='linear',
                param_name_pattern='[invalid',
            )


# =========================================================================
# PartitionConstraintV2 — matching
# =========================================================================

class TestPartitionConstraintV2Matching:

    def _make_module_stack(self, names):
        """Build a mock module_stack from class name strings."""
        stack = {}
        for i, name in enumerate(names):
            # Create a simple type with the given __name__
            t = type(name, (), {})
            stack[f'module_{i}'] = t
        return stack

    def test_op_name_exact_match(self):
        pc = PartitionConstraintV2(op_name='linear')
        assert pc.matches_operator('linear', {})
        assert not pc.matches_operator('relu', {})

    def test_module_class_match(self):
        pc = PartitionConstraintV2(module_class='Attention')
        stack = self._make_module_stack(['Model', 'Attention'])
        assert pc.matches_operator('any_op', stack)

    def test_module_class_no_match(self):
        pc = PartitionConstraintV2(module_class='Attention')
        stack = self._make_module_stack(['Model', 'MLP'])
        assert not pc.matches_operator('any_op', stack)

    def test_layer_range_inclusive_exclusive(self):
        pc = PartitionConstraintV2(op_name='linear', layer_range=(0, 4))
        assert pc.matches_operator('linear', {}, layer_index=0)
        assert pc.matches_operator('linear', {}, layer_index=3)
        assert not pc.matches_operator('linear', {}, layer_index=4)
        assert not pc.matches_operator('linear', {}, layer_index=None)

    def test_param_name_pattern(self):
        pc = PartitionConstraintV2(
            op_name='linear',
            param_name_pattern=r'layers\.\d+\.attention\.weight',
        )
        assert pc.matches_operator(
            'linear', {},
            param_fqns=['model.layers.3.attention.weight'],
        )
        assert not pc.matches_operator(
            'linear', {},
            param_fqns=['model.layers.3.mlp.weight'],
        )

    def test_combined_criteria_and_logic(self):
        pc = PartitionConstraintV2(
            op_name='linear',
            module_class='Attention',
            layer_range=(0, 8),
        )
        stack = self._make_module_stack(['Attention'])
        # All match
        assert pc.matches_operator('linear', stack, layer_index=3)
        # Op name wrong
        assert not pc.matches_operator('relu', stack, layer_index=3)
        # Module wrong
        assert not pc.matches_operator(
            'linear', self._make_module_stack(['MLP']), layer_index=3,
        )
        # Layer out of range
        assert not pc.matches_operator('linear', stack, layer_index=10)

    def test_matched_tracking(self):
        pc = PartitionConstraintV2(op_name='linear')
        assert not pc.is_matched
        pc.matches_operator('relu', {})
        assert not pc.is_matched  # didn't match
        # Direct mark
        pc.mark_matched()
        assert pc.is_matched


# =========================================================================
# SchedulerConstraint
# =========================================================================

class TestSchedulerConstraint:

    def test_valid_default(self):
        sc = SchedulerConstraint()
        assert sc.max_bubble_ratio == 0.2

    def test_valid_with_schedulers(self):
        sc = SchedulerConstraint(
            allowed_schedulers=['1f1b', '1f1b_plus'],
            max_bubble_ratio=0.15,
            max_stages=4,
        )
        assert sc.max_stages == 4

    def test_invalid_scheduler_name(self):
        with pytest.raises(ConstraintValidationError, match='Unknown schedulers'):
            SchedulerConstraint(allowed_schedulers=['nonexistent'])

    def test_invalid_bubble_ratio(self):
        with pytest.raises(ConstraintValidationError, match='max_bubble_ratio'):
            SchedulerConstraint(max_bubble_ratio=1.5)

    def test_invalid_max_stages(self):
        with pytest.raises(ConstraintValidationError, match='max_stages'):
            SchedulerConstraint(max_stages=0)


# =========================================================================
# ConstraintSet
# =========================================================================

class TestConstraintSet:

    def test_empty_set(self):
        cs = ConstraintSet()
        cs.validate_all()
        assert len(cs.partition_constraints) == 0

    def test_unmatched_detection(self):
        cs = ConstraintSet(partition_constraints=[
            PartitionConstraintV2(op_name='linear'),
            PartitionConstraintV2(op_name='relu'),
        ])
        # Neither was matched
        assert len(cs.get_unmatched_constraints()) == 2
        # Simulate matching
        cs.partition_constraints[0].mark_matched()
        assert len(cs.get_unmatched_constraints()) == 1

    def test_coverage_report(self):
        cs = ConstraintSet(partition_constraints=[
            PartitionConstraintV2(op_name='linear'),
        ])
        cs.partition_constraints[0].mark_matched()
        report = cs.get_coverage_report(total_ops=50)
        assert report['total_constraints'] == 1
        assert report['matched_constraints'] == 1
        assert report['total_operators'] == 50


# =========================================================================
# YAML serialization round-trip
# =========================================================================

class TestYamlRoundTrip:

    def test_partition_constraint_round_trip(self):
        original = PartitionConstraintV2(
            op_name='linear',
            module_class='Attention',
            layer_range=(0, 16),
            allowed_dims=[(0, 2), (1, 0)],
            max_partition_degree=4,
            stage_id=1,
            recompute=True,
        )
        yaml_dict = original.to_yaml()
        restored = PartitionConstraintV2.from_yaml(yaml_dict)
        assert restored.op_name == original.op_name
        assert restored.module_class == original.module_class
        assert restored.layer_range == original.layer_range
        assert restored.allowed_dims == original.allowed_dims
        assert restored.max_partition_degree == original.max_partition_degree
        assert restored.stage_id == original.stage_id
        assert restored.recompute == original.recompute

    def test_scheduler_constraint_round_trip(self):
        original = SchedulerConstraint(
            allowed_schedulers=['1f1b', '1f1b_plus'],
            max_bubble_ratio=0.15,
            max_stages=4,
            min_microbatches=8,
            prefer_interleaved=True,
        )
        yaml_dict = original.to_yaml()
        restored = SchedulerConstraint.from_yaml(yaml_dict)
        assert restored.allowed_schedulers == original.allowed_schedulers
        assert restored.max_bubble_ratio == original.max_bubble_ratio
        assert restored.max_stages == original.max_stages
        assert restored.prefer_interleaved == original.prefer_interleaved

    def test_constraint_set_round_trip(self):
        cs = ConstraintSet(
            partition_constraints=[
                PartitionConstraintV2(op_name='linear', allowed_dims=[(0, 1)]),
                PartitionConstraintV2(module_class='MLP', force_replicate=True),
            ],
            scheduler_constraint=SchedulerConstraint(
                allowed_schedulers=['1f1b'],
            ),
        )
        yaml_dict = cs.to_yaml()
        restored = ConstraintSet.from_yaml(yaml_dict)
        assert len(restored.partition_constraints) == 2
        assert restored.partition_constraints[0].op_name == 'linear'
        assert restored.partition_constraints[1].force_replicate is True
        assert restored.scheduler_constraint is not None
        assert restored.scheduler_constraint.allowed_schedulers == ['1f1b']

    def test_constraint_set_from_yaml_file(self):
        content = {
            'partition_constraints': [
                {'op_name': 'linear', 'allowed_dims': ['0,2']},
            ],
            'scheduler_constraint': {
                'allowed_schedulers': ['1f1b', 'gpipe'],
                'max_bubble_ratio': 0.1,
            },
        }
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.yaml', delete=False,
        ) as f:
            yaml.dump(content, f)
            path = f.name
        try:
            cs = ConstraintSet.from_yaml_file(path)
            assert len(cs.partition_constraints) == 1
            assert cs.scheduler_constraint.max_bubble_ratio == 0.1
        finally:
            os.unlink(path)


# =========================================================================
# Legacy conversion
# =========================================================================

class TestLegacyConversion:

    def test_convert_legacy(self):
        legacy = [
            {
                'name': 'torch.nn.functional.linear',
                'parent_module': 'LlamaAttention',
                'allowed_partition_dims': ['0,2'],
                'replica_allowed': True,
            },
            {
                'name': 'torch.nn.functional.embedding',
                'parent_module': '',
                'allowed_partition_dims': ['0,0', '0,1'],
                'replica_allowed': False,
            },
        ]
        cs = convert_legacy_constraints(legacy)
        assert len(cs.partition_constraints) == 2
        assert cs.partition_constraints[0].op_name == 'torch.nn.functional.linear'
        assert cs.partition_constraints[0].module_class == 'LlamaAttention'
        assert cs.partition_constraints[0].allowed_dims == [(0, 2)]
        # Second constraint: no module_class (empty string is not set)
        assert cs.partition_constraints[1].allowed_dims == [(0, 0), (0, 1)]


# =========================================================================
# Helpers
# =========================================================================

class TestHelpers:

    def test_extract_layer_index(self):
        assert _extract_layer_index({'model.layers.5.attn': None}) == 5
        assert _extract_layer_index({'encoder.layer.12': None}) == 12
        assert _extract_layer_index({'blocks.0.ffn': None}) == 0
        assert _extract_layer_index({'embeddings': None}) is None
        assert _extract_layer_index({}) is None


# =========================================================================
# ValidationResult
# =========================================================================

class TestValidationResult:

    def test_summary_passed(self):
        vr = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            coverage={
                'total_constraints': 2,
                'matched_constraints': 2,
                'unmatched_constraints': 0,
                'total_operators': 100,
                'constrained_operators': 30,
                'unconstrained_operators': 70,
            },
            search_space_estimate={'reduction_factor': 5.2},
        )
        s = vr.summary()
        assert 'PASSED' in s
        assert '5.2x' in s

    def test_summary_failed(self):
        vr = ValidationResult(
            is_valid=False,
            errors=['bad stage_id'],
            warnings=['unused constraint'],
            coverage={
                'total_constraints': 1,
                'matched_constraints': 0,
                'unmatched_constraints': 1,
                'total_operators': 50,
                'constrained_operators': 0,
                'unconstrained_operators': 50,
            },
        )
        s = vr.summary()
        assert 'FAILED' in s
        assert 'bad stage_id' in s
        assert 'unused constraint' in s
