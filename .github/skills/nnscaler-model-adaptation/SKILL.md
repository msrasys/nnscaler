---
name: nnscaler-model-adaptation
description: Zero-GPU, hardware-decoupled adaptation and parallelization workflow for onboarding new HuggingFace or PyTorch models (Dense, Sparse MoE, Hybrid Linear/SSM/MLA Attention, Multimodal VLMs) onto nnScaler. Use when adapting a new model architecture to nnScaler, diagnosing FX/IR tracing or operator annotation bottlenecks, wrapping custom Triton/CUDA kernels with fake_fn and IRDimops, designing surgical model rewrites, or configuring declarative JSON parallelization (TP/PP/EP/ETP/ZeRO) and static IR audits.
---

# nnScaler Model Adaptation & Declarative Parallelization Playbook

Follow this **5-stage hardware-decoupled engineering workflow** to onboard any frontier model architecture (Dense, Sparse MoE, Hybrid Linear/SSM/MLA Attention, Multimodal VLM) onto `nnScaler` **on a 0-GPU CPU machine** before touching a multi-GPU cluster.

Reference implementation: `examples/qwen3p5/` (`ops/`, `rewrites/`, `policy/json_pas.py`, `audit/ir_auditor.py`, `compile.py`, `tests/`).

---

## Core Principle: Decouple Compilation from Cluster Runtime

Never require multi-GPU hardware or multi-hundred-GB checkpoints just to adapt, partition, and validate a model:

1. **Micro-Preset First**: Define a 4-layer micro config (`hidden_size=128`, `vocab_size=512`, small expert/vision dims) alongside official presets (`0.8B` .. `400B+`) so CPU eager forward+backward parity tests run in `< 2s`.
2. **Weightless `meta` Compilation**: Instantiate frontier configs under `with torch.device("meta"):` and call `nnscaler.parallelize(...)` with:
   ```python
   ComputeConfig(
       plan_ngpus,
       runtime_ngpus,
       use_end2end=True,
       trace_strategy="meta",
       param_init_strategy=ParamInitStrategy.SHARD,
       pas_config=...,
   )
   # parallelize(..., load_module=False, broadcast_strategy="none", gen_savedir=out_dir)
   ```

---

## Stage 1: Zero-GPU Probe & Bottleneck Triage

Instantiate a micro-preset of the target HuggingFace class (`AutoModelForCausalLM.from_config(...)` or multimodal equivalent), wrap it in a thin end-to-end training wrapper returning scalar `loss`, and run `_gen_graph` + `audit_pre_pas` (`examples/qwen3p5/audit/ir_auditor.py`).

Classify every tracing failure or suboptimal IR node into one of **4 canonical categories**:

| Category | Symptoms in FX / Pre-PAS IR | Standard Resolution |
|---|---|---|
| **1. Missing standard PyTorch op** | Unregistered `torch.*` / `_operator.*` / `Tensor.*` method or opaque `IRFwOperation` | Add `IRDimops` class in `nnscaler/graph/function/function.py` and register in `nnscaler/graph/parser/mapping.py`. |
| **2. Data-dependent control flow** | `.item()`, `.tolist()`, `.nonzero()`, or Python `for` loop over routed tokens (`meta` trace crashes or bakes dummy routing) | Replace Python loop / `.tolist()` in-memory with a static-boundary custom op (`@register_op(..., fake_fn=...)`) that executes dynamic indexing eagerly at runtime. |
| **3. Locked parallelism / loss axes** | `F.cross_entropy(..., reduction='mean')` (`N^ C^ -> 1`), flat `[Q\|K\|V]` fused projection, or biased `F.linear(x, w, bias)` (`k^` locked) | Rewrite loss to `reduction='sum'` divided by `valid_count`; permute fused QKV weights into TP-contiguous head groups; decompose Row-TP biased linear into `F.linear(x, w, None) + bias`. |
| **4. External CUDA/Triton `autograd.Function`** | `fla`, `causal_conv1d`, `flash_attn_varlen`, `grouped_gemm` fail on CPU/`meta` or appear as opaque `IRPyFunc` | Wrap in a module-level Python function (never register `.apply` directly when `fake_fn` is needed) with `@register_op(anno_fn, fake_fn=...)` and a pure-PyTorch CPU fallback. |

---

## Stage 2: Custom Operator Registration (`ops/`)

When registering novel operators with `nnscaler.graph.parser.register.register_op`:

1. **Always wrap `autograd.Function.apply` in a module-level `def`**:
   - `register_op` forbids `fake_fn` on raw `autograd.Function` classes/applies.
2. **Write exact `IRDimops` annotations**:
   - Use **composite dimensions** `(hk g^)` when two inputs share a grouped head ratio (e.g., GQA or Gated DeltaNet where `Hv = Hk * g`), locking the intra-group factor `g^` while leaving `hk` partitionable across `tp_size`.
   - Mark contracted/summed dimensions with `+` (`E+` for expert parallelism, `k+` for Row-TP) and non-splittable dimensions with `^`.
   - Use `TransformRule(itransform, otransform, modifier)` when partitioning an axis must also update scalar kwargs (e.g., updating `local_expert_start` and `local_expert_end` when sharding `[E, ...]` across EP ranks; see `examples/qwen3p5/ops/moe_ops.py:build_fused_ep_transform_rule`).
3. **Provide both a `fake_fn` and a differentiable CPU fallback**:
   - `fake_fn` must return output shells with the runtime function's exact shapes, dtypes, and per-output `requires_grad` flags. `torch.empty_like(x)` / `x.new_empty(...)` do not inherit gradient flags: set them explicitly from the output's differentiable input/weight dependencies and `torch.is_grad_enabled()`. Keep integer indices, boolean masks, and intentionally detached outputs non-differentiable. Validate metadata against the CPU fallback with all inputs frozen, each differentiable input enabled separately, and under `torch.no_grad()`; test fake outputs on both CPU and meta.
   - Do not repair missing gradients only by filtering codegen tensor/gradient lists: incorrect fake metadata can select forward-only communication and silently cut gradients on multiple ranks even when single-rank autograd works. Correct the metadata and regenerate the IR and code.
   - Fake integer outputs consumed by real indexing during CPU tracing must contain initialized, in-range representative values, not uninitialized `torch.empty` data. Keep sequence boundaries consistent when consumed by real sequence operations. Metadata-only meta compilation cannot detect these value errors; do not let dummy values specialize data-dependent routing or shape decisions.
   - Inside the runtime function, dispatch to optimized CUDA/Triton/CUTLASS kernels only for supported devices, dtypes, layouts, and shapes, and fall back to a pure-PyTorch differentiable reference on CPU. An HF `@use_kernel_func_from_hub_with_fallback` wrapper may choose an installed FLA implementation without checking the input device; explicitly select the CPU reference rather than assuming the decorator does so. Record which implementation tests actually execute: reference-path success does not validate accelerated kernels.
4. **Default to Fused Mega-Ops for MoE (`expand_moe_ops: false`)**:
   - Provide a fused routed-MLP op (`qwen3_5_moe_route` + `qwen3_5_moe_grouped_gemm`) as the default to avoid dynamic token-histogram graph breaks, and expose decomposed atomic ops (`permute` -> `grouped_linear` -> `unpermute`) behind an opt-in `"expand_moe_ops": true` config flag when fine-grained `ETP` or `EP × ETP` exploration is required.
   - Preserve autograd dependencies for empty experts and ranks with no routed tokens; do not return a detached zero or skip branches needed by backward communication. Test zero-token batches and empty local experts for zero-valued (not missing) gradients on differentiable inputs/weights and consistent backward collective ordering across EP ranks.

---

## Stage 3: Surgical In-Memory Rewrites & CPU Golden Parity Gate (`rewrites/`)

**Never copy-paste entire HuggingFace `modeling_*.py` files.** Keep HuggingFace upgrades effortless by patching live `nn.Module` instances in-memory before tracing (`examples/qwen3p5/rewrites/model_rewrites.py`):

- Bind replacement `.forward` methods onto specific submodule instances via `types.MethodType(_rewritten_forward, submodule)`.
- Permute fused projection weights/biases in-place (with `torch.no_grad()`, skipping data permutation only when `param.is_meta` while preserving shape) and record permutation metadata so checkpoints remain compatible.
- Wrap the root model in a thin `CompileWrapper(nn.Module)` that unpacks `samples: dict[str, torch.Tensor]` and computes token-normalized loss:
  ```python
  loss_sum = F.cross_entropy(shift_logits, shift_labels, reduction="sum", ignore_index=-100)
  loss = loss_sum / valid_token_count.clamp(min=1).to(loss_sum.dtype)
  ```
  *(Avoid `mean` CE, which locks `N^ C^ -> 1` and triggers a multi-GB full-vocabulary `AllGather` across GPUs.)*

### Mandatory Two-Tier CPU Numerical Gate (`rtol/atol < 1e-4`)

Before running any compiler pass, write fast CPU tests (`examples/qwen3p5/tests/test_ops_and_rewrites_parity.py`):

1. **Per-Op Split Equivalence**: Run each custom op unsharded vs. manually sliced along its `TP`/`EP`/`ETP` dimension (`torch.cat` or `+` sum on outputs), call `.backward(grad_out)`, and assert `torch.testing.assert_close` on outputs and all input/weight `.grad` tensors.
2. **End-to-End HF Golden Parity**: Instantiate unmodified HF vs. rewritten HF on the micro-preset with identical `torch.manual_seed`, run forward + `.backward()`, and assert identical loss and `embed_tokens.weight.grad`.
   - *Tip*: If HF `@use_kernel_func_from_hub_with_fallback` attempts CUDA dispatch on CPU tensors, temporarily unwrap it with `inspect.unwrap(fn)` inside a context manager for the golden reference run.

---

## Stage 4: Declarative JSON Parallelization (`policy/json_pas.py`)

Reuse `examples/qwen3p5/policy/json_pas.py` to compile JSON strategy configs directly into `nnscaler.policies.fn` `OpPlan`s without writing model-specific imperative graph passes:

1. **Inspect Pre-PAS FQNs & Signatures**:
   - Run a 1-stage or `TP=2` compile and inspect `02_ir_pre_pas.txt` to read exact module FQNs (e.g., `layers.\d+.self_attn.(q_proj|k_proj|v_proj)$`, `shared_expert.down_proj$`) and op signatures.
2. **Author Declarative Strategy JSON (`strategies/*.json`)**:
   - **`mesh`**: `plan_ngpus`, `runtime_ngpus`, `pipeline_nstages`, `pipeline_size` (sets `pas_config['pipeline_size']`; `tp_size = plan_ngpus // pipeline_size`), `pipeline_nmicros`, `pipeline_scheduler`, `zero_stage`.
   - **`stage_splits`**: Map contiguous layer ranges (`"0..14"`) to `stage_id` (`0..pipeline_nstages-1`).
   - **`recompute_modules`**: Regex list of submodule scopes (`["layers\\.\\d+"]`) for activation checkpointing.
   - **`rules`**: Ordered regex rules mapping ops to `tp_col`, `tp_row`, `head_parallel`, `expert_parallel`, `expert_tp_col`, `expert_tp_row`, `vocab_parallel`, `replicate`, `auto`, or explicit `{"input": idx, "dim": dim}`.
   - **`on_indivisible`**:
     - Use `"fallback_replicate"` on GQA KV projections or head-parallel ops where small models may have `num_key_value_heads < tp_size`.
     - Use `"error"` (default) to fail fast with an actionable `StrategyValidationError` when an invalid shard dimension or indivisible mesh is requested.

---

## Stage 5: Static IR & AST Verification Bundle (Pre-Cluster Gate)

Before launching any multi-GPU job, compile both micro and full frontier presets (`30B`–`400B+`) on CPU via `audit_pre_pas` and `audit_post_pas` (`examples/qwen3p5/audit/ir_auditor.py`) and verify all **5 static gates**:

1. **Zero Opaque Forward Tensor Ops**: `pre_pas_summary["num_opaque_fw_ops"] == 0` (every tensor-transforming op is a known `IRDimops`).
2. **Zero Dynamic Shape Tensors**: `pre_pas_summary["num_dynamic_tensors"] == 0`.
3. **Zero `mean` CE Bottlenecks**: `pre_pas_summary["num_mean_ce_bottlenecks"] == 0`.
4. **Exact Architectural Rule Hit Counts**: Verify `policy_report["rule_hit_counts"]` matches the exact layer counts of the target architecture (e.g., `60` MoE layers, `45` linear-attn + `15` full-attn layers, `1` vocab-parallel `lm_head`).
5. **100% Rank Codegen & AST Collective Audit**:
   - `found_gencode_files == plan_ngpus` and every `gencode*.py` parses cleanly via `ast.parse`.
   - Syntax parsing alone is insufficient: verify `_train_step` has no undefined gradient references and every backward-call tensor/gradient pair matches. Check that custom-op boundaries requiring gradients use differentiable communication. Execute generated micro-model training steps and compare parameter gradients with the CPU reference before claiming training correctness.
   - Inspect both `IRFullTensor` and `IRSubTensor` outputs, including tensors nested in Python containers, when detecting opaque operations. Reject differentiable tensor outputs from `IRPyFunc`; a clean summary that checks only full tensors can miss gradient breaks. Compare every loss-dependent parameter's gradient and optimizer update over multiple generated training steps, not only the embedding gradient.
   - Inspect per-rank AST collective counts (`all_reduce`, `all_gather`, `reduce_scatter`, P2P `send`/`recv`) in `03_adapters.json` to detect unintended `AllGather` explosions or unsharded weights before allocating GPU cluster time.

---

## Common Pitfalls Checklist

- **Unregistered `unbind` on trainable tensors**: It can become an `IRPyFunc` returning a Python tuple and lose IR backward edges (for example QKV splits or HyperConnection scales). Use supported tensor indexing/splitting in a surgical rewrite, then verify the affected parameter gradients in generated training, not only eager parity.
- **`F.scaled_dot_product_attention(..., enable_gqa=True)`**: Ensure composite head grouping `(kh g^)` is active when `q.shape[-3] != k.shape[-3]`; note that if HF expands KV via `repeat_kv` before SDPA, `q/k/v` enter SDPA with full `num_attention_heads` (`Expand` uses `"auto"` fallback cleanly).
- **Biased `nn.Linear` under Row-TP (`tp_row`)**: `nnScaler` locks contraction dim `k^` on `F.linear(x, w, bias)` when `bias is not None` so bias is not added `tp_size` times during `AllReduce`. Always rewrite Row-TP target linears with bias into `F.linear(x, w, None) + bias`.
- **Zero-GPU `<tensor>.device` & `is_tracing(mask)`**: Modern HF `transformers` calls `torch.arange(..., device=x.device)` and `isinstance(mask, torch.fx.Proxy)`. `nnScaler`'s `BaseTraceStrategy._remap_cuda_devices` and `wrap_utils.isinstance_wrapper` handle this automatically—keep `trace_strategy="meta"` enabled for large models.
