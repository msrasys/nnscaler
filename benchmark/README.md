# Ring Attention Performance Benchmarks

This directory contains performance benchmarking tools for Ring Attention with variable length sequences.

## 📊 Benchmark Scripts

### `benchmark_ring_attn_varlen.py`
Main performance benchmarking script with comprehensive timing and profiling capabilities.

**Features:**
- Predefined configurations for different model sizes
- Multiple timing methods (simple, warmup, profiler)
- Distributed training support via torchrun
- Custom parameter override support
- TransformerEngine integration with fallback

## 🚀 Quick Start

### 1. List Available Configurations

```bash
cd benchmark
python benchmark_ring_attn_varlen.py --list-configs
```

### 2. Run Basic Benchmark

```bash
# Use predefined configuration
torchrun --nproc_per_node=2 benchmark_ring_attn_varlen.py --config medium

# Use custom parameters  
torchrun --nproc_per_node=2 benchmark_ring_attn_varlen.py --seqlen 8192 --nheads 16 --head-dim 128

# Detailed profiling
torchrun --nproc_per_node=2 benchmark_ring_attn_varlen.py --config large --timing-method profiler
```

### 3. Timing Methods

- **`simple`**: Basic CUDA timing measurements
- **`warmup`**: Multiple runs with warm-up (recommended for accurate results)
- **`profiler`**: torch.profiler with detailed analysis

## 📋 Available Configurations

### Small Configs (Quick Testing)
- `tiny`: 2×8×64, seq=1024, tokens=1K
- `small`: 4×12×128, seq=4096, tokens=4K

### Medium Configs (Standard Testing)
- `medium`: 4×24×128, seq=8192, tokens=8K
- `medium_large_head`: 4×12×256, seq=8192, tokens=8K
- `medium_many_heads`: 4×32×128, seq=8192, tokens=8K

### Large Configs (Performance Testing)
- `large`: 4×32×128, seq=16384, tokens=16K
- `large_seq`: 4×24×128, seq=32768, tokens=32K
- `xlarge`: 8×32×128, seq=32768, tokens=32K

### Model Configs (Realistic Workloads)
- `llama_7b_like`: 4×32×128, seq=16384, tokens=16K
- `llama_13b_like`: 4×40×128, seq=16384, tokens=16K
- `llama_30b_like`: 4×52×128, seq=16384, tokens=16K
- `llama_65b_like`: 4×64×128, seq=16384, tokens=16K

## 🔧 Usage Examples

### Basic Performance Testing
```bash
# Quick medium-scale benchmark
torchrun --nproc_per_node=2 benchmark_ring_attn_varlen.py --config medium --timing-method warmup

# Test different data types
torchrun --nproc_per_node=2 benchmark_ring_attn_varlen.py --config small --dtype fp16
torchrun --nproc_per_node=2 benchmark_ring_attn_varlen.py --config small --dtype bf16
```

### Advanced Profiling
```bash
# Detailed profiler analysis
torchrun --nproc_per_node=2 benchmark_ring_attn_varlen.py --config large --timing-method profiler

# Custom timing parameters
torchrun --nproc_per_node=2 benchmark_ring_attn_varlen.py --config medium --timing-method warmup --warmup-runs 5 --timing-runs 10
```

### Custom Configurations
```bash
# Override specific parameters
torchrun --nproc_per_node=2 benchmark_ring_attn_varlen.py --config medium --seqlen 16384 --nheads 32

# Full custom configuration
torchrun --nproc_per_node=2 benchmark_ring_attn_varlen.py --seqlen 8192 --nheads 16 --head-dim 128 --batch-size 4 --dtype bf16
```

## 📈 Output Interpretation

The benchmark provides several performance metrics:

### Timing Results
- **Forward Pass Time**: Time for attention computation
- **Backward Pass Time**: Time for gradient computation  
- **Memory Usage**: Peak GPU memory consumption
- **Speedup**: Parallel vs single GPU performance ratio

### Profiler Results (when using `--timing-method profiler`)
- **Kernel-level timing**: Detailed CUDA kernel execution times
- **Memory bandwidth**: Data transfer rates
- **Utilization**: GPU compute utilization percentages

## 🔗 Integration

### With Test Framework
```bash
# Run correctness tests first
cd ..
pytest tests/customized_ops/ring_attn/test_correctness.py -v

# Then run performance benchmarks
cd benchmark
torchrun --nproc_per_node=2 benchmark_ring_attn_varlen.py --config medium
```

### With Configuration System
The benchmark script automatically imports configurations from `tests/customized_ops/ring_attn/configs.py`, ensuring consistency between correctness tests and performance benchmarks.

## ⚠️ Requirements

- **Multi-GPU setup**: Most benchmarks require 2+ GPUs
- **Sufficient memory**: Large configs may require high-memory GPUs
- **TransformerEngine**: Optional but recommended for best performance

## 🚨 Troubleshooting

### Common Issues
- **OOM errors**: Use smaller configs or reduce batch size
- **Import errors**: Ensure running from MagicCube root directory
- **NCCL errors**: Check GPU compatibility and CUDA setup

### Debug Commands
```bash
# Test basic functionality
python benchmark_ring_attn_varlen.py --help

# Validate imports
python -c "
import sys, os
sys.path.insert(0, '../tests/customized_ops/ring_attn')
from configs import get_config
print('✓ Config system working')
"
```

---

*For correctness testing, see: `tests/customized_ops/ring_attn/`*  
*For implementation details, see: `nnscaler/customized_ops/ring_attention/`*