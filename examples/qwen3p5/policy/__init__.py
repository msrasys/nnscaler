"""Declarative JSON parallelization strategy schema, validator, and `nnscaler.policies.fn` compiler."""

from examples.qwen3p5.policy.json_pas import (
    StrategyConfig,
    StrategyValidationError,
    build_compute_config,
    build_pas_policy,
    compile_op_plans,
    load_strategy_config,
)

__all__ = [
    "StrategyConfig",
    "StrategyValidationError",
    "build_compute_config",
    "build_pas_policy",
    "compile_op_plans",
    "load_strategy_config",
]
