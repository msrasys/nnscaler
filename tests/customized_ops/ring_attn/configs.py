#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""
Configuration file for ring attention tests.
This file contains predefined test configurations for both correctness and performance testing.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class RingAttnConfig:
    """Configuration for ring attention test cases"""
    batch_size: int
    num_heads: int
    head_dim: int
    max_seqlen: int
    dtype: str = "bf16"
    name: str = ""
    num_kv_heads: Optional[int] = None  # For GQA/MQA support
    causal: bool = True  # Most attention patterns are causal
    window_size: Tuple[int, int] = (-1, -1)  # Sliding window attention (-1, -1) means no window

    def __post_init__(self):
        # Set num_kv_heads to num_heads if not specified (standard MHA)
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads

        if not self.name:
            gqa_suffix = f"_gqa{self.num_kv_heads}" if self.num_kv_heads != self.num_heads else ""
            causal_suffix = "" if self.causal else "_noncausal"
            window_suffix = f"_w{self.window_size[0]}-{self.window_size[1]}" if self.window_size != (-1, -1) else ""
            self.name = f"b{self.batch_size}_h{self.num_heads}_d{self.head_dim}_s{self.max_seqlen}_{self.dtype}{gqa_suffix}{causal_suffix}{window_suffix}"

        # Generate cu_seqlens for variable length sequences
        # Create sequences with different lengths for more realistic testing
        seq_lens = [
            self.max_seqlen // 8,      # Short sequence
            self.max_seqlen // 4,      # Medium sequence
            self.max_seqlen // 2,      # Long sequence
            self.max_seqlen - self.max_seqlen // 8 - self.max_seqlen // 4 - self.max_seqlen // 2  # Remaining
        ]
        self.cu_seqlens = [0]
        for seq_len in seq_lens:
            self.cu_seqlens.append(self.cu_seqlens[-1] + seq_len)

    @property
    def total_tokens(self) -> int:
        """Total number of tokens across all sequences"""
        return self.cu_seqlens[-1]

    @property
    def is_gqa(self) -> bool:
        """Check if this is a GQA (Grouped Query Attention) configuration"""
        return self.num_kv_heads < self.num_heads

    @property
    def is_mqa(self) -> bool:
        """Check if this is an MQA (Multi-Query Attention) configuration"""
        return self.num_kv_heads == 1

    @property
    def num_groups(self) -> int:
        """Number of query heads per KV head (group size)"""
        return self.num_heads // self.num_kv_heads


# Small test cases for quick correctness validation
SMALL_CONFIGS = {
    "tiny": RingAttnConfig(2, 8, 64, 1024, "bf16", "tiny", causal=True),
    "small": RingAttnConfig(4, 12, 128, 4096, "bf16", "small", causal=True),
    "small_fp16": RingAttnConfig(4, 12, 128, 4096, "fp16", "small_fp16", causal=False),  # One non-causal config
    "small_window": RingAttnConfig(4, 12, 128, 4096, "bf16", "small_window", causal=True, window_size=(512, 0)),  # Sliding window
}

# Medium test cases for standard testing
MEDIUM_CONFIGS = {
    "medium": RingAttnConfig(4, 24, 128, 8192, "bf16", "medium", causal=True),
    "medium_large_head": RingAttnConfig(4, 12, 256, 8192, "bf16", "medium_large_head", causal=False),  # One non-causal config
    "medium_many_heads": RingAttnConfig(4, 32, 128, 8192, "bf16", "medium_many_heads", causal=True),
    "medium_fp16": RingAttnConfig(4, 24, 128, 8192, "fp16", "medium_fp16", causal=True),
    "medium_window": RingAttnConfig(4, 24, 128, 8192, "bf16", "medium_window", causal=True, window_size=(512, 0)),  # Sliding window
}

# Large test cases for performance benchmarking
LARGE_CONFIGS = {
    "large": RingAttnConfig(4, 32, 128, 16384, "bf16", "large", causal=True),
    "large_seq": RingAttnConfig(4, 24, 128, 32768, "bf16", "large_seq", causal=True),
    "large_head": RingAttnConfig(4, 24, 256, 16384, "bf16", "large_head", causal=False),  # One non-causal config
    "xlarge": RingAttnConfig(8, 32, 128, 32768, "bf16", "xlarge", causal=True),
    "large_window": RingAttnConfig(4, 32, 128, 16384, "bf16", "large_window", causal=True, window_size=(512, 0)),  # Sliding window
}

# Realistic model configurations (kept minimal, most covered by medium/large configs)
MODEL_CONFIGS = {
}

# GQA (Grouped Query Attention) configurations based on Qwen models
GQA_CONFIGS = {
    # Qwen3-235B-A22B: 64 heads, 4 kv_heads, 128 head_dim
    "qwen3_235b_a22b": RingAttnConfig(
        batch_size=2,
        num_heads=64,
        head_dim=64,
        max_seqlen=16384,
        dtype="bf16",
        name="qwen3_235b_a22b",
        num_kv_heads=4,
        causal=True
    ),

    # Qwen3-30B-A3B: 40 heads, 8 kv_heads, 128 head_dim
    "qwen3_30b_a3b": RingAttnConfig(
        batch_size=4,
        num_heads=32,
        head_dim=64,
        max_seqlen=16384,
        dtype="bf16",
        name="qwen3_30b_a3b",
        num_kv_heads=4,
        causal=True
    ),

    # Qwen3-4B: 32 heads, 4 kv_heads, 80 head_dim
    "qwen3_4b": RingAttnConfig(
        batch_size=4,
        num_heads=32,
        head_dim=80,
        max_seqlen=16384,
        dtype="bf16",
        name="qwen3_4b",
        num_kv_heads=4,
        causal=True
    ),

    # Qwen3-32B: 64 heads, 8 kv_heads, 128 head_dim
    "qwen3_32b": RingAttnConfig(
        batch_size=2,
        num_heads=64,
        head_dim=128,
        max_seqlen=16384,
        dtype="bf16",
        name="qwen3_32b",
        num_kv_heads=8,
        causal=True
    ),

    # Qwen3-14B: 40 heads, 8 kv_heads, 128 head_dim
    "qwen3_14b": RingAttnConfig(
        batch_size=4,
        num_heads=40,
        head_dim=128,
        max_seqlen=16384,
        dtype="bf16",
        name="qwen3_14b",
        num_kv_heads=8,
        causal=True
    ),
}

# MQA is already covered by medium/large configs, so removed duplicate MQA_CONFIGS

# All configurations combined
ALL_CONFIGS = {
    **SMALL_CONFIGS,
    **MEDIUM_CONFIGS,
    **LARGE_CONFIGS,
    **MODEL_CONFIGS,
    **GQA_CONFIGS,
}

# Default configurations for different test types
DEFAULT_CORRECTNESS_CONFIGS = ["tiny", "small", "medium"]
DEFAULT_PERFORMANCE_CONFIGS = ["medium", "large"]
DEFAULT_MULTI_GPU_CONFIGS = ["small", "medium"]
DEFAULT_GQA_CONFIGS = ["qwen3_4b", "qwen3_14b", "qwen3_32b"]


def get_config(name: str) -> RingAttnConfig:
    """Get a configuration by name"""
    if name in ALL_CONFIGS:
        return ALL_CONFIGS[name]
    else:
        raise ValueError(f"Unknown configuration: {name}. Available: {list(ALL_CONFIGS.keys())}")


def list_configs(category: str = "all") -> List[str]:
    """List available configurations by category"""
    if category == "all":
        return list(ALL_CONFIGS.keys())
    elif category == "small":
        return list(SMALL_CONFIGS.keys())
    elif category == "medium":
        return list(MEDIUM_CONFIGS.keys())
    elif category == "large":
        return list(LARGE_CONFIGS.keys())
    elif category == "model":
        return list(MODEL_CONFIGS.keys())
    elif category == "gqa":
        return list(GQA_CONFIGS.keys())
    elif category == "correctness":
        return DEFAULT_CORRECTNESS_CONFIGS
    elif category == "performance":
        return DEFAULT_PERFORMANCE_CONFIGS
    elif category == "multi_gpu":
        return DEFAULT_MULTI_GPU_CONFIGS
    elif category == "gqa_default":
        return DEFAULT_GQA_CONFIGS
    else:
        raise ValueError(f"Unknown category: {category}")


def get_configs_by_category(category: str) -> dict:
    """Get all configurations in a category"""
    config_names = list_configs(category)
    return {name: get_config(name) for name in config_names}


def get_gqa_configs() -> dict:
    """Get all GQA (Grouped Query Attention) configurations"""
    return {name: config for name, config in ALL_CONFIGS.items() if config.is_gqa and not config.is_mqa}


def get_mqa_configs() -> dict:
    """Get all MQA (Multi-Query Attention) configurations"""
    return {name: config for name, config in ALL_CONFIGS.items() if config.is_mqa}


def get_mha_configs() -> dict:
    """Get all MHA (Multi-Head Attention) configurations"""
    return {name: config for name, config in ALL_CONFIGS.items() if not config.is_gqa}


def filter_configs_by_attention_type(attention_type: str) -> dict:
    """Filter configurations by attention type: 'mha', 'gqa', or 'mqa'"""
    if attention_type.lower() == "mha":
        return get_mha_configs()
    elif attention_type.lower() == "gqa":
        return get_gqa_configs()
    elif attention_type.lower() == "mqa":
        return get_mqa_configs()  # Will return empty dict since no dedicated MQA configs
    else:
        raise ValueError(f"Unknown attention type: {attention_type}. Supported: 'mha', 'gqa', 'mqa'")