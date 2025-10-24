# Ring Attention Performance Benchmarks

This directory contains a unified performance benchmarking framework for all Ring Attention variants, built using a shared architecture that eliminates code duplication and provides consistent interfaces.

## 🏗️ Architecture

The benchmark framework consists of:

### Core Framework
- **`benchmark_base.py`**: Shared benchmark framework extending the test framework
- **Configuration System**: Unified configuration management via `../tests/customized_ops/ring_attn/configs.py`

### Attention Implementations
- **`benchmark_ring_attn.py`**: Standard Ring Attention benchmarks
- **`benchmark_ring_attn_varlen.py`**: Variable Length Ring Attention benchmarks  
- **`benchmark_zigzag_attn.py`**: Zigzag Ring Attention benchmarks (causal-only)

## 🚀 Quick Start

### 1. List Available Configurations

```bash
cd benchmark

# List configurations for any benchmark variant
python benchmark_ring_attn_varlen.py --list-configs
python benchmark_ring_attn.py --list-configs
python benchmark_zigzag_attn.py --list-configs
```

### 2. Run Basic Benchmarks

```bash
# Ring Attention Variable Length
torchrun --nproc_per_node=2 benchmark_ring_attn_varlen.py --config medium

# Standard Ring Attention  
torchrun --nproc_per_node=2 benchmark_ring_attn.py --config small

# Zigzag Ring Attention (causal-only)
torchrun --nproc_per_node=2 benchmark_zigzag_attn.py --config tiny
```

### 3. Advanced Usage

```bash
# Custom timing parameters
torchrun --nproc_per_node=2 benchmark_ring_attn_varlen.py --config medium --timing-method warmup --warmup-runs 5 --timing-runs 10

# Detailed profiling
torchrun --nproc_per_node=2 benchmark_ring_attn.py --config large --timing-method profiler

# Custom configurations (legacy support)
torchrun --nproc_per_node=2 benchmark_ring_attn.py --seqlen 8192 --nheads 16 --head-dim 128 --batch-size 4
```

## 📋 Available Configurations

The benchmark framework uses a comprehensive configuration system with predefined configurations for different testing scenarios.

### Configuration Categories

#### Small Configs (Quick Testing)
- **`tiny`**: 2×8×64, seq=1024, tokens=1K, bf16 [Causal]
- **`small`**: 4×12×128, seq=4096, tokens=4K, bf16 [Causal]
- **`small_fp16`**: 4×12×128, seq=4096, tokens=4K, fp16 [Non-causal]
- **`small_window`**: 4×12×128, seq=4096, tokens=4K, bf16 [Causal] [Window=512,0]

#### Medium Configs (Standard Testing)
- **`medium`**: 4×24×128, seq=8192, tokens=8K, bf16 [Causal]
- **`medium_large_head`**: 4×12×256, seq=8192, tokens=8K, bf16 [Non-causal]
- **`medium_many_heads`**: 4×32×128, seq=8192, tokens=8K, bf16 [Causal]
- **`medium_fp16`**: 4×24×128, seq=8192, tokens=8K, fp16 [Causal]
- **`medium_window`**: 4×24×128, seq=8192, tokens=8K, bf16 [Causal] [Window=512,0]

#### Large Configs (Performance Testing)
- **`large`**: 4×32×128, seq=16384, tokens=16K, bf16 [Causal]
- **`large_seq`**: 4×24×128, seq=32768, tokens=32K, bf16 [Causal]
- **`large_head`**: 4×24×256, seq=16384, tokens=16K, bf16 [Non-causal]
- **`xlarge`**: 8×32×128, seq=32768, tokens=32K, bf16 [Causal]
- **`large_window`**: 4×32×128, seq=16384, tokens=16K, bf16 [Causal] [Window=512,0]

#### GQA Configs (Grouped Query Attention)
- **`qwen3_235b_a22b`**: 2×64×64, seq=16384, tokens=16K, bf16 (GQA 64→4) [Causal]
- **`qwen3_30b_a3b`**: 4×32×64, seq=16384, tokens=16K, bf16 (GQA 32→4) [Causal]
- **`qwen3_4b`**: 4×32×80, seq=16384, tokens=16K, bf16 (GQA 32→4) [Causal]
- **`qwen3_32b`**: 2×64×128, seq=16384, tokens=16K, bf16 (GQA 64→8) [Causal]
- **`qwen3_14b`**: 4×40×128, seq=16384, tokens=16K, bf16 (GQA 40→8) [Causal]

#### Zigzag Configs (Causal-Only)
- **`zigzag_tiny`**: 2×8×64, seq=1024, tokens=1K, bf16 [Causal]
- **`zigzag_small`**: 4×12×128, seq=4096, tokens=4K, bf16 [Causal]
- **`zigzag_medium`**: 4×24×128, seq=8192, tokens=8K, bf16 [Causal]
- **`zigzag_large`**: 4×32×128, seq=16384, tokens=16K, bf16 [Causal]
- **`zigzag_fp16`**: 4×12×128, seq=4096, tokens=4K, fp16 [Causal]
- **`zigzag_gqa`**: 4×32×128, seq=8192, tokens=8K, bf16 (GQA 32→8) [Causal]

### Default Configuration Sets
- **Correctness Testing**: `["tiny", "small", "medium"]`
- **Performance Testing**: `["medium", "large"]`
- **Multi-GPU Testing**: `["small", "medium"]`
- **GQA Testing**: `["qwen3_4b", "qwen3_14b", "qwen3_32b"]`
- **Zigzag Testing**: `["zigzag_tiny", "zigzag_small", "zigzag_medium"]`

## 🔧 Features

### Unified Framework
- **Shared Base Class**: All benchmarks extend `RingAttnBenchmarkBase` for consistency
- **Code Reuse**: Leverages test framework components (`test_base.py`, `runner_base.py`)
- **Consistent Interface**: Same command-line options across all attention variants

### Multiple Timing Methods
- **`simple`**: Basic CUDA timing measurements (fastest)
- **`warmup`**: Multiple runs with warm-up (recommended for accurate results)
- **`profiler`**: torch.profiler with detailed kernel analysis

### Comprehensive Metrics
- **Performance**: Forward/backward timing, throughput (tokens/sec)
- **Scalability**: Speedup analysis, parallel efficiency
- **Memory**: GPU memory usage tracking
- **Comparative**: Single vs. parallel mode analysis

### Configuration Support
- **Predefined Configs**: 20+ predefined configurations covering different scales
- **Legacy Parameters**: Backward compatibility with custom parameters
- **Attention Variants**: Support for standard, variable-length, and zigzag attention
- **GQA Support**: Grouped Query Attention configurations based on Qwen models

## 🧪 Usage Examples

### Basic Performance Testing
```bash
# Quick benchmarks with different attention types
torchrun --nproc_per_node=2 benchmark_ring_attn_varlen.py --config tiny --timing-method simple
torchrun --nproc_per_node=2 benchmark_ring_attn.py --config small --timing-method warmup
torchrun --nproc_per_node=2 benchmark_zigzag_attn.py --config medium --dtype fp16
```

### Comparative Analysis
```bash
# Compare different attention mechanisms on same config
torchrun --nproc_per_node=2 benchmark_ring_attn.py --config medium --timing-method warmup
torchrun --nproc_per_node=2 benchmark_ring_attn_varlen.py --config medium --timing-method warmup  
torchrun --nproc_per_node=2 benchmark_zigzag_attn.py --config medium --timing-method warmup
```

### Advanced Profiling
```bash
# Detailed profiler analysis
torchrun --nproc_per_node=2 benchmark_ring_attn_varlen.py --config large --timing-method profiler

# Custom timing parameters for high precision
torchrun --nproc_per_node=2 benchmark_ring_attn.py --config medium --timing-method warmup --warmup-runs 10 --timing-runs 20
```

### GQA Performance Testing
```bash
# Test Grouped Query Attention configurations
torchrun --nproc_per_node=2 benchmark_ring_attn.py --config qwen3_4b --timing-method warmup
torchrun --nproc_per_node=2 benchmark_ring_attn_varlen.py --config qwen3_14b --timing-method warmup
```

### Legacy Support (Custom Parameters)
```bash
# Override specific parameters while using predefined base
torchrun --nproc_per_node=2 benchmark_ring_attn.py --config medium --seqlen 16384 --nheads 32

# Full custom configuration
torchrun --nproc_per_node=2 benchmark_ring_attn_varlen.py --seqlen 8192 --nheads 16 --head-dim 128 --batch-size 4 --dtype bf16
```

## 📈 Output Interpretation

The benchmark framework provides comprehensive performance analysis:

### Performance Metrics
```
================================================================================
RING ATTENTION VARIABLE LENGTH PERFORMANCE BENCHMARK (WARMUP METHOD)
Configuration: medium - medium
  Sequence length: 8192
  Batch size: 4
  Heads: 24
  Head dim: 128
  Data type: bf16
  World size: 2 GPUs
  Total tokens: 8,192
  (Warmup runs: 3, Timing runs: 5)
================================================================================
Single Mode:
  Forward time:  0.001234 seconds
  Backward time: 0.002345 seconds
  Total time:    0.003579 seconds
  Throughput:    2288764 tokens/sec

Parallel Mode:
  Forward time:  0.000987 seconds
  Backward time: 0.001654 seconds
  Total time:    0.002641 seconds
  Throughput:    3102234 tokens/sec

Speedup:
  Forward speedup:     1.25x
  Backward speedup:    1.42x
  Total speedup:       1.35x
  Throughput improvement: 1.35x

Efficiency:
  Theoretical speedup: 2x
  Actual speedup:      1.35x
  Parallel efficiency: 67.7%
================================================================================
```

### Key Metrics Explained
- **Forward/Backward Time**: Separate timing for forward and backward passes
- **Throughput**: Tokens processed per second (higher = better)
- **Speedup**: Performance ratio vs single GPU (higher = better)
- **Parallel Efficiency**: Actual speedup / theoretical speedup (closer to 100% = better)

### Profiler Output (when using `--timing-method profiler`)
When using the profiler method, you get additional detailed analysis:
- Kernel-level timing breakdown
- Memory bandwidth utilization
- CUDA kernel execution patterns
- Optimization recommendations

## 🎯 Attention Variant Characteristics

### Ring Attention (`benchmark_ring_attn.py`)
- **Format**: Standard batch format `[batch_size, seq_len, num_heads, head_dim]`
- **Use Case**: General purpose attention for standard transformer models
- **Constraints**: Supports both causal and non-causal attention, sliding windows

### Ring Attention Variable Length (`benchmark_ring_attn_varlen.py`)
- **Format**: Packed format `[total_tokens, num_heads, head_dim]` with `cu_seqlens`
- **Use Case**: Optimized for variable-length sequences, eliminates padding waste
- **Constraints**: Supports causal/non-causal attention, sliding windows

### Zigzag Attention (`benchmark_zigzag_attn.py`)
- **Format**: Standard batch format `[batch_size, seq_len, num_heads, head_dim]`
- **Use Case**: Specialized for causal attention with optimized communication pattern
- **Constraints**: **Only supports causal=True and window_size=(-1,-1)**

## 🔗 Integration with Test Framework

The benchmark framework is tightly integrated with the correctness test framework:

### Shared Components
- **Configuration System**: Same `configs.py` used for both correctness and performance testing
- **Base Classes**: Reuses `RingAttnRunnerBase` from `runner_base.py`
- **Distributed Setup**: Shared GPU detection and distributed initialization
- **Error Handling**: Consistent tolerance and validation logic

### Workflow Integration
```bash
# 1. Run correctness tests first
cd /path/to/MagicCube
pytest tests/customized_ops/ring_attn/test_ring_attn_varlen.py --config tiny

# 2. Then run performance benchmarks  
cd benchmark
torchrun --nproc_per_node=2 benchmark_ring_attn_varlen.py --config tiny
```

## ⚠️ Requirements & Setup

### System Requirements
- **Multi-GPU Setup**: Most benchmarks require 2+ GPUs (use `torchrun --nproc_per_node=N`)
- **GPU Memory**: Large configs may require high-memory GPUs (A100, H100 recommended)
- **CUDA**: Compatible CUDA installation (11.8+ recommended)
- **Python Environment**: PyTorch with NCCL support for distributed training

### Optional Components
- **TransformerEngine**: Install TE 2.2.0+ for optimal performance (auto-detected)
- **Flash Attention**: Required for base attention implementations
- **InfiniBand**: Recommended for multi-node setups (reduces communication latency)

### Environment Setup
```bash
# From MagicCube root directory
cd benchmark

# Verify imports work correctly
python -c "
from benchmark_base import RingAttnBenchmarkBase
print('✓ Benchmark framework ready')
"

# Test configuration system
python benchmark_ring_attn_varlen.py --list-configs
```

## 🚨 Troubleshooting

### Common Issues

#### GPU/Memory Issues
```bash
# OOM errors: Use smaller configs or reduce batch size
torchrun --nproc_per_node=2 benchmark_ring_attn.py --config tiny  # Instead of large

# Insufficient GPUs: Check available GPUs
python -c "import torch; print(f'Available GPUs: {torch.cuda.device_count()}')"
```

#### Import/Path Issues
```bash
# Import errors: Ensure running from correct directory
cd /path/to/MagicCube/benchmark
python benchmark_ring_attn.py --help

# Configuration import errors
python -c "
import sys, os
sys.path.insert(0, '../tests/customized_ops/ring_attn')
from configs import get_config
print('✓ Config system working')
"
```

#### Distributed Training Issues
```bash
# NCCL errors: Check GPU compatibility and CUDA setup
export NCCL_DEBUG=INFO  # For detailed NCCL debugging

# Port conflicts: Use different port
torchrun --master_port=29501 --nproc_per_node=2 benchmark_ring_attn.py --config tiny
```

### Performance Debugging
```bash
# Test basic functionality without distributed training
CUDA_VISIBLE_DEVICES=0 python -c "
from benchmark_ring_attn import RingAttnBenchmark
print('✓ Benchmark classes load correctly')
"

# Verify attention implementations work
cd ../tests/customized_ops/ring_attn
pytest test_ring_attn.py::TestRingAttn::test_ring_attn_tiny -v
```

**Note**: Actual efficiency depends on hardware, network, and system configuration.

## 📚 Related Documentation

### Core Documentation
- **Ring Attention Implementation**: `../nnscaler/customized_ops/ring_attention/README.md`
- **Test Framework**: `../tests/customized_ops/ring_attn/README.md`  
- **Development Guide**: `../dev_docs/README_refactoring.md`
- **Testing Results**: `../dev_docs/benchmark_testing_results.md`

---

**For implementation details**: See `../nnscaler/customized_ops/ring_attention/`  
**For correctness testing**: See `../tests/customized_ops/ring_attn/`  