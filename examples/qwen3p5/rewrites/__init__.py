"""Instance-level subgraph rewrites and weightless/eager model builders for Qwen3.5."""

from examples.qwen3p5.rewrites.model_rewrites import (
    Qwen3_5CompileWrapper,
    RewriteConfig,
    apply_qwen3_5_rewrites,
    build_dummy_samples,
    build_qwen3_5_model,
    permute_gdn_qkv_weights_,
)

__all__ = [
    "Qwen3_5CompileWrapper",
    "RewriteConfig",
    "apply_qwen3_5_rewrites",
    "build_dummy_samples",
    "build_qwen3_5_model",
    "permute_gdn_qkv_weights_",
]
