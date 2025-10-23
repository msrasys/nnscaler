#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import os
import sys
import subprocess

import pytest
import torch

from configs import (
    DEFAULT_CORRECTNESS_CONFIGS,
    DEFAULT_MULTI_GPU_CONFIGS,
    DEFAULT_GQA_CONFIGS,
    get_config,
    list_configs
)


def _get_project_root():
    """Get the absolute path to nnscaler root directory"""
    current_dir = os.path.dirname(__file__)  # tests/customized_ops/ring_attn/
    return os.path.abspath(os.path.join(current_dir, "../../../"))


def get_bash_arguments(num_gpus_per_node, **kwargs):
    """Generate command line arguments for running the test script"""
    args = [
        "python3",
        "-m",
        "torch.distributed.launch",
        "--nproc-per-node=" + str(num_gpus_per_node),
    ]

    project_root = _get_project_root()
    script_path = os.path.join(project_root, "tests", "customized_ops", "ring_attn", "run_ring_attn_correctness.py")
    args.append(script_path)

    for k, v in kwargs.items():
        args.append(f"{k}={v}")
    return args


@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
@pytest.mark.parametrize("config_name", DEFAULT_CORRECTNESS_CONFIGS)
def test_ring_attn_correctness(dtype, config_name):
    """Test ring attention correctness with different configurations"""
    num_gpus = 2  # Default to 2 GPUs for correctness tests
    if num_gpus > torch.cuda.device_count():
        pytest.skip(f"Test requires {num_gpus} GPUs, but found {torch.cuda.device_count()}")

    config = get_config(config_name)

    subprocess.run(
        get_bash_arguments(
            num_gpus_per_node=num_gpus,
            dtype=dtype,
            config_name=config_name,
        ),
        check=True,
        cwd=_get_project_root()
    )


@pytest.mark.parametrize("num_gpus", [2, 4])
@pytest.mark.parametrize("config_name", DEFAULT_MULTI_GPU_CONFIGS)
def test_ring_attn_multi_gpu(num_gpus, config_name):
    """Test ring attention with different numbers of GPUs"""
    if num_gpus > torch.cuda.device_count():
        pytest.skip(f"Test requires {num_gpus} GPUs, but found {torch.cuda.device_count()}")

    subprocess.run(
        get_bash_arguments(
            num_gpus_per_node=num_gpus,
            dtype="bf16",
            config_name=config_name,
        ),
        check=True,
        cwd=_get_project_root()
    )


@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
def test_ring_attn_all_configs(dtype):
    """Test all available configurations (comprehensive test)"""
    num_gpus = 2
    if num_gpus > torch.cuda.device_count():
        pytest.skip(f"Test requires {num_gpus} GPUs, but found {torch.cuda.device_count()}")

    # Test a selection of configurations
    test_configs = ["tiny", "small", "medium"]

    for config_name in test_configs:
        config = get_config(config_name)
        # Skip very large configs for comprehensive test
        if config.max_seqlen > 16384:
            continue

        subprocess.run(
            get_bash_arguments(
                num_gpus_per_node=num_gpus,
                dtype=dtype,
                config_name=config_name,
            ),
            check=True,
            cwd=_get_project_root()
        )


@pytest.mark.parametrize("dtype", ["bf16"])
@pytest.mark.parametrize("config_name", DEFAULT_GQA_CONFIGS)
def test_ring_attn_gqa_correctness(dtype, config_name):
    """Test ring attention GQA correctness with Qwen model configurations"""
    num_gpus = 2
    if num_gpus > torch.cuda.device_count():
        pytest.skip(f"Test requires {num_gpus} GPUs, but found {torch.cuda.device_count()}")

    config = get_config(config_name)

    # Ensure it's actually a GQA config
    assert config.is_gqa, f"Configuration {config_name} should be GQA"
    assert config.num_kv_heads < config.num_heads, f"Configuration {config_name} should have fewer KV heads"

    subprocess.run(
        get_bash_arguments(
            num_gpus_per_node=num_gpus,
            dtype=dtype,
            config_name=config_name,
        ),
        check=True,
        cwd=_get_project_root()
    )


@pytest.mark.parametrize("dtype", ["bf16"])
@pytest.mark.parametrize("config_name", ["small_window", "medium_window"])
def test_ring_attn_sliding_window(dtype, config_name):
    """Test ring attention with sliding window configurations"""
    num_gpus = 2
    if num_gpus > torch.cuda.device_count():
        pytest.skip(f"Test requires {num_gpus} GPUs, but found {torch.cuda.device_count()}")

    config = get_config(config_name)

    # Ensure it's actually a sliding window config
    assert config.window_size != (-1, -1), f"Configuration {config_name} should have sliding window"

    subprocess.run(
        get_bash_arguments(
            num_gpus_per_node=num_gpus,
            dtype=dtype,
            config_name=config_name,
        ),
        check=True,
        cwd=_get_project_root()
    )


if __name__ == "__main__":
    # Run specific test if called directly
    test_ring_attn_correctness("bf16", "small")

    # Example of running GQA test
    # test_ring_attn_gqa_correctness("bf16", "qwen3_4b")

    # Example of running sliding window test
    # test_ring_attn_sliding_window("bf16", "small_window")