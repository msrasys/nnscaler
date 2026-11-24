# Qwen3.5 Family Compilation & Declarative Parallelism (`examples/qwen3p5`)

Hardware-decoupled compilation, custom `IRDimops` operator registrations, surgical HuggingFace graph rewrites, static IR/collective auditing, and declarative JSON parallelization strategies for the **Qwen3.5** model family (`0.8B`–`27B` Dense, `35B-A3B`–`397B-A17B` Sparse MoE, and Multimodal VL).

All compilation and static verification steps run on a **single CPU node with 0 GPUs and 0 checkpoint weights** (`trace_strategy="meta"`, `param_init_strategy="shard"`, `load_module=False`).

---

## Supported Architectures & Presets

| Preset (`--model`) | Architecture | Layers | Hybrid Attention | MLP / MoE |
|---|---|---|---|---|
| `qwen3.5-0.8b` .. `qwen3.5-27b` | `Qwen3_5ForCausalLM` | 24–64 | 3:1 Gated DeltaNet (`linear_attn`) + Full GQA (`self_attn`) | Dense SwiGLU |
| `qwen3.5-35b-a3b` .. `qwen3.5-397b-a17b` | `Qwen3_5MoeForCausalLM` | 40–60 | 3:1 Gated DeltaNet + Full GQA | Routed Top-K SwiGLU Experts + Shared Expert |
| `micro-dense`, `micro-moe`, `micro-vl` | Dense / MoE / `Qwen3_5ForConditionalGeneration` | 4 | 3:1 GDN + GQA (+ ViT 2D Patch Merger & 3D mRoPE) | Fast CI / parity presets |

---

## Quick Start (Zero-GPU Compilation CLI)

Compile `Qwen3.5-0.8B` for 4 GPUs (`PP=2, TP=2`):

```bash
PYTHONPATH=. python -m examples.qwen3p5.compile \
  --model qwen3.5-0.8b \
  --strategy examples/qwen3p5/strategies/dense_pp2_tp2.json \
  --out-dir /tmp/qwen3p5_0p8b_pp2_tp2
```

Compile `Qwen3.5-397B-A17B` (60 layers, 512 routed experts) for 32 GPUs (`PP=4, EP/TP=8`) on a CPU-only machine without allocating ~800 GB of weights:

```bash
PYTHONPATH=. python -m examples.qwen3p5.compile \
  --model qwen3.5-397b-a17b \
  --strategy examples/qwen3p5/strategies/moe_ep8_pp4.json \
  --out-dir /tmp/qwen3p5_397b_pp4_ep8
```

Compile a Multimodal VL model (`Qwen3_5ForConditionalGeneration`) with `TP=2`:

```bash
PYTHONPATH=. python -m examples.qwen3p5.compile \
  --model micro-vl \
  --multimodal \
  --strategy examples/qwen3p5/strategies/vl_tp2.json \
  --out-dir /tmp/qwen3p5_vl_tp2
```

You can also pass `--model /path/to/config.json` (with optional `--num-layers N` or `--expand-moe-ops {true,false}` overrides).

---

## Verification Bundle Layout

Every `compile.py` invocation produces an auditable bundle in `--out-dir`:

- `00_model_and_strategy.json`: Resolved architecture dimensions, CLI overrides, normalized JSON strategy, and verification summary.
- `02_ir_audit.json` & `02_ir_pre_pas.txt`: Pre-partition IR audit (`opaque_ops == []`, custom op counts, unlocked `N+ C^` CE loss) and full pre-PAS IR text dump.
- `03_adapters.json` & `03_ir_post_pas.txt`: Post-partition audit reporting policy plan/warnings, `gencode*.py` AST collective counts (`all_reduce`, `all_gather`, `reduce_scatter`, `send`/`recv`), and post-PAS IR dump.
- `04_generated/**/gencode*.py`: Generated per-rank PyTorch execution modules ready for multi-GPU runtime import (`load_parallel_module`).

---

## Declarative Strategy JSON & `expand_moe_ops`

Strategy files live in `examples/qwen3p5/strategies/*.json` and compile directly into `nnscaler.policies.fn` `OpPlan` rules:

- **`rewrites`**:
  - `"expand_moe_ops": false` (**default**): Wraps routed Top-K expert MLP inside `qwen3_5_moe_route` + `qwen3_5_moe_grouped_gemm`, yielding clean Expert Parallelism (`dim=0` of `E`) with `AllReduce` on `h+` and eliminating dynamic token-histogram graph breaks.
  - `"expand_moe_ops": true` (opt-in): Decomposes routed MoE into explicit `qwen3_5_moe_permute` -> `qwen3_5_grouped_linear` -> `qwen3_5_moe_unpermute`, exposing intermediate expert projections for fine-grained `EP`, `ETP`, or hybrid `EP × ETP`.
  - `"chunked_lm_head_ce": false`: Keeps `lm_head` + token-summed `F.cross_entropy(..., reduction="sum") / valid_count` exposed for Vocab-Parallel TP (`v+`) and Data/Token Parallelism (`N+`). Set `true` only when vocabulary sharding is disabled (`TP=1`) and memory-saving token chunking is preferred.
- **`rules`**: Ordered pattern matchers (`fqn_regex`, `op_name_regex`, `is_Forward`) emitting `partition` (`[input_or_weight_idx, dim]` or `"replicate"`/`"auto"`) with `"on_indivisible": "error"` (strict fail-fast) or `"fallback_replicate"` (safe automatic fallback when e.g. GQA KV heads `< tp_size`).

---

## Running Tests

Run all core patch tests, CPU HuggingFace golden forward/backward parity checks, and end-to-end weightless compilations on any CPU or GPU machine:

```bash
PYTHONPATH=. python -m examples.qwen3p5.tests.run_all_tests
```

Include the full 60-layer `Qwen3.5-397B-A17B` 32-GPU weightless compilation test:

```bash
PYTHONPATH=. python -m examples.qwen3p5.tests.run_all_tests --include-397b
```
