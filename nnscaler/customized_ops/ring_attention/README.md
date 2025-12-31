# Ring Attention Implementation

High-performance ring attention mechanisms for nnscaler, supporting multiple attention variants and distributed training.

## 📖 Overview

This module implements multiple efficient attention mechanisms designed to distribute computation evenly in long sequence processing:

- **Ring Attention**: Standard ring attention supporting arbitrary sequence lengths
- **Ring Attention Variable Length**: Variable-length sequence optimized ring attention  
- **Zigzag Attention**: Zigzag pattern ring attention optimized for causal attention

All implementations are deeply integrated with nnscaler's parallel computing framework, supporting automatic distributed training.

## 🏗️ Architecture Design

```
nnscaler/customized_ops/ring_attention/
├── __init__.py                           # Package import interface
├── ring_attn.py                         # Standard ring attention
├── ring_attn_varlen.py                  # Variable length ring attention
├── zigzag_attn.py                       # Zigzag ring attention
├── varlen_utils.py                      # Variable length utility functions
└── core/                                # Core implementations
    ├── ring_attn_implementation.py      # Standard ring attention core
    ├── ring_attn_varlen_implementation.py # Variable length core implementation
    ├── zigzag_attn_implementation.py    # Zigzag attention core implementation
    └── utils.py                         # Common utility functions
```

## 🚀 Quick Start

### Standard Ring Attention

```python
from nnscaler.customized_ops.ring_attention import wrap_ring_attn_func

# Basic usage
output = wrap_ring_attn_func(
    q,  # [batch_size, seq_len, num_heads, head_dim]
    k,  # [batch_size, seq_len, num_heads, head_dim] 
    v,  # [batch_size, seq_len, num_heads, head_dim]
    causal=True,              # Causal attention mask
    window_size=(-1, -1),     # Sliding window size, (-1,-1) means global attention
    softmax_scale=None,       # Softmax scale factor, defaults to 1/sqrt(head_dim)
    dropout_p=0.0            # Dropout probability
)
```

### Variable Length Ring Attention

```python
from nnscaler.customized_ops.ring_attention import wrap_ring_attn_varlen_func

# Variable length sequence attention
output = wrap_ring_attn_varlen_func(
    q,                       # [total_tokens, num_heads, head_dim]
    k,                       # [total_tokens, num_heads, head_dim]
    v,                       # [total_tokens, num_heads, head_dim]
    cu_seqlens_q,           # Cumulative sequence lengths [batch_size + 1]
    cu_seqlens_k,           # Cumulative sequence lengths [batch_size + 1]
    bias=None,              # Optional attention bias
    causal=True,            # Causal attention mask
    window_size=(-1, -1),   # Sliding window size
    softmax_scale=None,     # Softmax scale factor
    dropout_p=0.0          # Dropout probability
)
```

### Zigzag Ring Attention

```python
from nnscaler.customized_ops.ring_attention import wrap_zigzag_attn_func

# Zigzag attention (causal attention only)
output = wrap_zigzag_attn_func(
    q,  # [batch_size, seq_len, num_heads, head_dim]
    k,  # [batch_size, seq_len, num_heads, head_dim]
    v,  # [batch_size, seq_len, num_heads, head_dim]
    causal=True,              # Must be True
    window_size=(-1, -1),     # Must be (-1, -1), sliding window not supported
    softmax_scale=None,
    dropout_p=0.0
)
```

## 🔧 Core Features

### Performance Optimization
- **Flash Attention integration**: Efficient implementation based on flash_attn
- **TransformerEngine support**: Automatic detection and usage of TE 2.2.0+
- **CUDA kernel optimization**: GPU-optimized low-level implementations
- **Distributed friendly**: Seamless integration with torch.distributed

### Flexible Configuration
- **Attention patterns**: Support for causal and non-causal attention
- **Sliding window**: Configurable local attention windows
- **GQA support**: Grouped Query Attention optimization
- **Custom scaling**: Flexible softmax scaling strategies

## 🧮 Algorithm Principles

### Ring Attention Mechanism

Ring Attention decomposes attention computation into multiple blocks:

1. **Sequence chunking**: Divide long sequences into blocks distributed across devices
2. **Ring communication**: Devices pass key/value blocks by all-gather and reduce-scatter
3. **Incremental computation**: Each device computes attention with received key/value blocks

### Variable Length Optimization

Special optimizations for variable length sequences:

```python
# Cumulative sequence length example
cu_seqlens = [0, 128, 256, 512]  # 3 sequences with lengths 128, 128, 256
# Corresponding token tensor shape: [512, num_heads, head_dim]
```

### Zigzag Pattern

Zigzag Attention uses a special communication pattern for higher efficiency in causal attention scenarios:

- **Causal constraint**: Only supports causal=True cases
- **Optimized communication**: Ring communication optimized for causal masks
- **Memory friendly**: Further reduces unnecessary computation and communication

## 🔗 nnscaler Integration

### Automatic Parallelization

```python
from nnscaler.parallel import parallelize, ComputeConfig
from nnscaler.customized_ops.ring_attention import wrap_ring_attn_func

class AttentionModel(torch.nn.Module):
    def forward(self, q, k, v):
        return wrap_ring_attn_func(q, k, v, causal=True)

# nnscaler automatically handles distribution
config = ComputeConfig(
    plan_ngpus=4,
    runtime_ngpus=4
)
parallel_model = parallelize(model, config=config)
```

### Computation Graph Optimization

nnscaler automatically provides:
- **Communication optimization**: Minimize inter-device communication overhead
- **Memory planning**: Optimize memory usage patterns
- **Operator fusion**: Fuse with other operators for optimization
- **Gradient synchronization**: Automatic gradient communication in backward pass

## 🧪 Testing Framework

Comprehensive test coverage ensures implementation correctness and performance:

```bash
# Run all attention tests
pytest tests/customized_ops/ring_attn/ -v

# Specific attention variant tests
pytest tests/customized_ops/ring_attn/test_ring_attn.py -v
pytest tests/customized_ops/ring_attn/test_ring_attn_varlen.py -v  
pytest tests/customized_ops/ring_attn/test_zigzag_attn.py -v
```

### Test Types

- **Correctness tests**: Compare outputs with standard attention
- **Multi-GPU scalability**: Behavior validation across different device counts
- **GQA compatibility**: Grouped Query Attention correctness
- **Sliding window**: Local attention pattern validation
- **Edge cases**: Stability testing under extreme conditions

## 🛠️ Development Guide

### Adding New Attention Variants

1. **Core implementation**: Add implementation file in `core/` directory
2. **Wrapper function**: Create corresponding wrap function
3. **Test coverage**: Add comprehensive test cases
4. **Documentation**: Update README and API documentation

### Performance Optimization Tips

- **TransformerEngine**: Install TE 2.2.0+ for optimal performance
- **CUDA version**: Use CUDA 11.8+ for latest optimizations
- **Memory configuration**: Adjust batch size and sequence length based on GPU memory
- **Communication optimization**: Use InfiniBand networks to reduce communication latency

## 🚨 Known Limitations

### Ring Attention
- **alibi_slopes**: ALiBi positional encoding not currently supported
- **return_attn_probs**: Returning attention weights not supported

### Zigzag Attention  
- **causal**: Only supports causal attention (causal=True)
- **window_size**: Sliding window not supported (must be (-1,-1))

### General Limitations
- **Dynamic shapes**: Sequence length cannot change dynamically during training
- **Mixed precision**: May require special handling in certain configurations

## 📚 References

- **Ring Attention Paper**: [Ring Attention with Blockwise Transformers](https://arxiv.org/abs/2310.01889)
- **Flash Attention**: [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)
- **Llama3 Paper**: [The Llama3 Herd of Models](https://arxiv.org/pdf/2407.21783)
- **nnscaler Documentation**: [nnscaler Parallel Computing Framework](https://github.com/microsoft/nnscaler)
- **TransformerEngine**: [NVIDIA TransformerEngine](https://github.com/NVIDIA/TransformerEngine)

---

**Note**: This implementation is optimized for large-scale distributed training. For single-GPU scenarios, standard Flash Attention is recommended for optimal performance.