#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""
Base test framework for ring attention tests.
This module provides common functionality for both ring_attn and ring_attn_varlen tests.
"""

import os
import sys
import subprocess
from abc import ABC, abstractmethod
from typing import Dict, Any, List

import pytest
import torch

from configs import (
    DEFAULT_CORRECTNESS_CONFIGS,
    DEFAULT_MULTI_GPU_CONFIGS,
    DEFAULT_GQA_CONFIGS,
    get_config,
    list_configs
)


class RingAttnTestBase(ABC):
    """Base class for ring attention tests"""

    @property
    @abstractmethod
    def runner_script_name(self) -> str:
        """Return the name of the runner script (e.g., 'run_correctness.py')"""
        pass

    @property
    @abstractmethod
    def test_name_prefix(self) -> str:
        """Return the prefix for test names (e.g., 'ring_attn' or 'ring_attn_varlen')"""
        pass

    def _get_project_root(self):
        """Get the absolute path to nnscaler root directory"""
        current_dir = os.path.dirname(__file__)  # tests/customized_ops/ring_attn/
        return os.path.abspath(os.path.join(current_dir, "../../../"))

    def get_bash_arguments(self, num_gpus_per_node: int, **kwargs) -> List[str]:
        """Generate command line arguments for running the test script"""
        args = [
            "python3",
            "-m",
            "torch.distributed.launch",
            "--nproc-per-node=" + str(num_gpus_per_node),
        ]

        project_root = self._get_project_root()
        script_path = os.path.join(
            project_root, "tests", "customized_ops", "ring_attn",
            self.runner_script_name
        )
        args.append(script_path)

        for k, v in kwargs.items():
            args.append(f"{k}={v}")
        return args

    def run_test_subprocess(self, num_gpus: int, **kwargs):
        """Run test using subprocess with the configured runner script"""
        if num_gpus > torch.cuda.device_count():
            pytest.skip(f"Test requires {num_gpus} GPUs, but found {torch.cuda.device_count()}")

        subprocess.run(
            self.get_bash_arguments(
                num_gpus_per_node=num_gpus,
                **kwargs
            ),
            check=True,
            cwd=self._get_project_root()
        )

    # Common test methods that can be used by both ring_attn and ring_attn_varlen

    def test_correctness_basic(self, dtype: str, config_name: str):
        """Test correctness with different configurations"""
        num_gpus = 2  # Default to 2 GPUs for correctness tests
        config = get_config(config_name)

        self.run_test_subprocess(
            num_gpus=num_gpus,
            dtype=dtype,
            config_name=config_name,
        )

    def test_multi_gpu_scaling(self, num_gpus: int, config_name: str):
        """Test with different numbers of GPUs"""
        self.run_test_subprocess(
            num_gpus=num_gpus,
            dtype="bf16",
            config_name=config_name,
        )

    def test_comprehensive_configs(self, dtype: str):
        """Test all available configurations (comprehensive test)"""
        num_gpus = 2

        # Test a selection of configurations
        test_configs = ["tiny", "small", "medium"]

        for config_name in test_configs:
            config = get_config(config_name)
            # Skip very large configs for comprehensive test
            if config.max_seqlen > 16384:
                continue

            self.run_test_subprocess(
                num_gpus=num_gpus,
                dtype=dtype,
                config_name=config_name,
            )

    def test_gqa_correctness(self, dtype: str, config_name: str):
        """Test GQA correctness with Qwen model configurations"""
        num_gpus = 2
        config = get_config(config_name)

        # Ensure it's actually a GQA config
        assert config.is_gqa, f"Configuration {config_name} should be GQA"
        assert config.num_kv_heads < config.num_heads, f"Configuration {config_name} should have fewer KV heads"

        self.run_test_subprocess(
            num_gpus=num_gpus,
            dtype=dtype,
            config_name=config_name,
        )

    def test_sliding_window(self, dtype: str, config_name: str):
        """Test with sliding window configurations"""
        num_gpus = 2
        config = get_config(config_name)

        # Ensure it's actually a sliding window config
        assert config.window_size != (-1, -1), f"Configuration {config_name} should have sliding window"

        self.run_test_subprocess(
            num_gpus=num_gpus,
            dtype=dtype,
            config_name=config_name,
        )


def create_parametrized_tests(test_class: RingAttnTestBase):
    """
    Factory function to create parametrized test methods for a test class.
    This reduces code duplication between ring_attn and ring_attn_varlen tests.
    """

    # Correctness tests with different dtypes and configs
    @pytest.mark.parametrize("dtype", ["bf16", "fp16"])
    @pytest.mark.parametrize("config_name", DEFAULT_CORRECTNESS_CONFIGS)
    def test_correctness(dtype, config_name):
        """Test correctness with different configurations"""
        instance = test_class()
        instance.test_correctness_basic(dtype, config_name)

    # Multi-GPU tests
    @pytest.mark.parametrize("num_gpus", [2, 4])
    @pytest.mark.parametrize("config_name", DEFAULT_MULTI_GPU_CONFIGS)
    def test_multi_gpu(num_gpus, config_name):
        """Test with different numbers of GPUs"""
        instance = test_class()
        instance.test_multi_gpu_scaling(num_gpus, config_name)

    # Comprehensive tests
    @pytest.mark.parametrize("dtype", ["bf16", "fp16"])
    def test_all_configs(dtype):
        """Test all available configurations (comprehensive test)"""
        instance = test_class()
        instance.test_comprehensive_configs(dtype)

    # GQA tests
    @pytest.mark.parametrize("dtype", ["bf16"])
    @pytest.mark.parametrize("config_name", DEFAULT_GQA_CONFIGS)
    def test_gqa_correctness(dtype, config_name):
        """Test GQA correctness with Qwen model configurations"""
        instance = test_class()
        instance.test_gqa_correctness(dtype, config_name)

    # Sliding window tests
    @pytest.mark.parametrize("dtype", ["bf16"])
    @pytest.mark.parametrize("config_name", ["small_window", "medium_window"])
    def test_sliding_window(dtype, config_name):
        """Test with sliding window configurations"""
        instance = test_class()
        instance.test_sliding_window(dtype, config_name)

    return {
        f'test_{test_class().test_name_prefix}_correctness': test_correctness,
        f'test_{test_class().test_name_prefix}_multi_gpu': test_multi_gpu,
        f'test_{test_class().test_name_prefix}_all_configs': test_all_configs,
        f'test_{test_class().test_name_prefix}_gqa_correctness': test_gqa_correctness,
        f'test_{test_class().test_name_prefix}_sliding_window': test_sliding_window,
    }