# Architecture & Patch Plan: Agent-Driven Frontier Model Adaptation on Latest `nnscaler`

> **Context**: Re-architected for the fresh `msrasys/nnscaler` checkout (`4769c57d`, replacing `/home/v-xuasong/nnscaler_outdated`) and aligned with `/home/v-xuasong/nnscaler/examples/qwen3p5/spec.md`.
>
> **Core Goal**: Separate **compiler adaptation & strategy generation** (FX capture → IR construction → declarative parallelization → verification artifact export) from **AutoDist search** and **full-scale GPU training**, enabling an AI agent to adapt frontier models (`Qwen3.5-0.8B` up to `Qwen3.5-397B-A17B` and Multimodal) on lightweight hardware.

---

## 1. Re-Evaluating Previous (`nnscaler_outdated`) Patches

In `nnscaler_outdated`, we touched 6 core `nnscaler/` files (`+264 / -37` lines) plus `examples/qwen3p5_0p8B/`. Under the new goal—**no AutoDist search, no full-model training required, clean agent-driven PAS & verification artifacts**—several old patches were solving the wrong layer and should **not** be ported blindly:

| Old Patch in `nnscaler_outdated` | Verdict | Why (Under Latest `nnscaler` & New Goal) |
| --- | --- | --- |
| **`nnscaler/graph/function/function.py` (`MaskedFill` trailing broadcast)** | **KEEP & PORT** *(4 lines)* | Genuine FX→IR bug still present in latest `function.py:1129`. Needed whenever GDN or attention uses PyTorch eager/fallback causal masks (`[B, H, chunks, 64, 64]` vs `[64, 64]`). |
| **`nnscaler/graph/function/function.py` + `mapping.py` (`Softplus`)** | **KEEP & PORT** *(8 lines)* | Genuine missing core PyTorch op (`* -> *`). Every Qwen3.5 GDN layer (`0.8B`–`397B`) calls `F.softplus` for log-decay gate `g`. |
| **`nnscaler/codegen/emit.py` (`torch.memory_format`)** | **KEEP & PORT** *(2 lines)* | Still missing in latest `emit.py:67`. Required so `.to(..., memory_format=torch.contiguous_format)` emits valid Python in `gencode*.py`. |
| **`nnscaler/codegen/module/module.py` (`_autograd_imports`)** | **DROP (Redundant if ops are registered; optional fallback only)** | `register_op` already emits `import <module>` into `CustomizedOps.kOpCodeDef` (`register.py:265`). In the new framework, every custom `.apply` op should be explicitly registered (or wrapped in a Python function with `fake_fn`), rather than left as an unpartitionable `IRFwOperation`. |
| **`nnscaler/autodist/spmd_solver.py` + `apis.py` (`+264` lines shared-param solver & cost fix)** | **DROP COMPLETELY** | Solves `AutoDist` DP search & profiling cost mispricing. Latest `nnscaler/policies.py:1304` (`policies.fn`) **already supports static `OpPlan` policies** and handles shared/tied parameters (`len(splits) == 1` → true sharding with `0` `multiref`s; `len(splits) > 1` → automatic `multiref`; colocated PP stages supported via `#115`). |
| **`custom_ops.py` → `chunk_gated_delta_rule_input_gen`** | **DEMOTE / OPTIONAL** | `input_gen_fn` is only invoked by AutoDist's CUDA profiler (`profiler.py`). Keep domain constraints (`g <= 0`, valid `topk_idx`) in a **standalone unit-parity harness**, not coupled to AutoDist profiling. |
| **Registering raw `Function.apply` directly (`CausalConv1dFn.apply`, `ChunkGatedDeltaRuleFunction.apply`)** | **REFACTOR → Wrapper Function + `fake_fn`** | Latest `register.py:224` forbids `fake_fn` on `autograd.Function` (`ValueError: Autograd function cannot have fake runtime function`). Wrapping kernels in regular Python functions unlocks **`fake_fn`**, letting us compile `397B` on **CPU / `meta` tensors without CUDA kernels installed or GPU VRAM allocated**! |

**Result**: Core `nnscaler` modifications shrink from **`264+` lines across 6 files down to ~15 surgical lines in 3 parser/codegen files**, keeping the compiler clean and upstream-friendly.

---

## 2. Key Capabilities Already Built Into Latest `msrasys/nnscaler`

Inspecting the latest repo (`4769c57d`) reveals three major upstream features we should build directly upon instead of reinventing:

1. **Declarative Policy Engine (`nnscaler/policies.py` → `fn`, `OpPlan`, `OpPartition`, `group_nodes_by_modules`)**:
   - Upstream now records `node.module_stack`, `node.module_call_stack` (per-invocation IDs from `#107`), `node.fqn`, and `node.call_expr`.
   - `policies.fn(graph, cfg, policy)` accepts `Iterable[OpPlan]` (`stage_id`, `partition=OpPartition(input, dim) | None | 'auto'`, `recompute_id`, `offload_id`) and automatically inserts `staging`, `multiref`, `identity`, and `IRAdapterGener` collectives!
2. **Weightless / Zero-Checkpoint Compilation (`nnscaler/parallel.py`)**:
   - `ComputeConfig(plan_ngpus=N, runtime_ngpus=M, trace_strategy="meta", param_init_strategy=ParamInitStrategy.SHARD, use_end2end=True)` combined with `parallelize(..., load_module=False)` traces and generates `gencode0..N-1.py` **without saving `fullmodel.pt` (`AttrSaveLevel.M` only) and without initializing `torch.distributed` or allocating multi-GPU weights**.
3. **Pipeline & Shared-Parameter Fixes (`#114`, `#115`, `#142`, `#145`, `#146`, `#149`)**:
   - Value-partitioned segment outputs, colocated tied embeddings, and per-stage dataloader `getitem` sinking are already fixed in upstream `policies.fn`.

---

## 3. Proposed New Architecture (`examples/qwen3p5/`)

```txt
                   ┌──────────────────────────────────────────────┐
  HF Config JSON   │ 1. Weightless Model Builder & Rewrite Pass   │
 (0.8B .. 397B) ──▶│  - Meta/tiny-init instantiation (no 780GB!)  │
                   │  - Algebraic & Traceable Subgraph Rewrites   │
                   │    (sum/count loss, static MoE, GDN wrapper) │
                   └──────────────────────┬───────────────────────┘
                                          ▼
                   ┌──────────────────────────────────────────────┐
  Custom Op Specs  │ 2. CPU/Meta-Safe Custom Op Registry          │
  + fake_fn     ──▶│  - IRDimops + TransformRule + fake_fn        │
                   │  - Traces on CPU/Meta without CUDA OOM       │
                   └──────────────────────┬───────────────────────┘
                                          ▼
                   ┌──────────────────────────────────────────────┐
                   │ 3. Static IR Audit (Agent Feedback Sensor)   │
                   │  - Detects non-IRDimops, dynamic '?' shapes, │
                   │    frozen '^' axes, GQA/KV divisibility      │
                   └──────────────────────┬───────────────────────┘
                                          ▼
                   ┌──────────────────────────────────────────────┐
  Strategy JSON    │ 4. JSON-to-OpPlan Declarative PAS Compiler   │
 (Human / Agent)──▶│  - Maps FQN/module/op selectors → OpPlan     │
                   │  - Lowers via nnscaler.policies.fn (no search)│
                   └──────────────────────┬───────────────────────┘
                                          ▼
                   ┌──────────────────────────────────────────────┐
                   │ 5. Verification Artifact Bundle + Micro-Gate │
                   │  - Exports FX, Pre/Post-PAS IR, Adapters,    │
                   │    gencode*.py for TrainVerify               │
                   │  - Optional 1-step micro-model parity check  │
                   └──────────────────────────────────────────────┘
```

### Component A: Minimal Core `nnscaler` Patches (~15 Lines Total)
1. **`nnscaler/graph/function/function.py`**:
   - Fix `MaskedFill` using `_handle_broadcast(input, mask)` (right-aligned PyTorch broadcasting).
   - Add `Softplus(input, beta=1, threshold=20)` (`* -> *`).
2. **`nnscaler/graph/parser/mapping.py`**:
   - Map `__ftemplate('softplus')` and `__fcntemplate('softplus')` to `function.Softplus`.
3. **`nnscaler/codegen/emit.py`**:
   - Add `torch.memory_format` to `_safe_repr_value` literal pass-through types.

*(Optional compiler enhancement if needed during `397B` meta-tracing)*: Ensure `MetaStrategy.place_model` in `nnscaler/graph/tracer/trace_strategy.py` supports models already constructed on `meta` device (currently line `197` calls `module.cpu()`, which fails on meta parameters unless guarded by `if any(p.is_meta for p in model.parameters())`). **This 4-line fix in `trace_strategy.py` + `_gen_graph` will unlock true `0-byte` parameter memory tracing for `397B-A17B`!**

---

### Component B: Hardware-Decoupled Compilation Mode (Solving `397B` OOM)
To compile `Qwen3.5-35B-A3B`, `122B-A10B`, or `397B-A17B` for `plan_ngpus = 8, 16, 64` on a single CPU/GPU dev box:
1. **Config-Only Meta Instantiation**:
   - Build `Qwen3_5ForCausalLM` / `Qwen3_5MoeForCausalLM` from `AutoConfig` under `with torch.device("meta"):` (with non-meta rotary `inv_freq` / scalar constants materialized as tiny CPU tensors if needed by Python control flow).
2. **CPU/Meta-Compatible Custom Wrapper Ops (`fake_fn`)**:
   - Instead of registering `CausalConv1dFn.apply` and `ChunkGatedDeltaRuleFunction.apply` directly, wrap them in module-level functions:
     - `qwen3_5_causal_conv1d(...)` with `fake_fn` returning `torch.empty_like(x)` (and optional state).
     - `qwen3_5_chunk_gated_delta_rule(...)` with `fake_fn` returning `(q.new_empty(B, T, Hv, Dv), final_state)`.
     - **Configurable MoE Wrapper Granularity (`expand_moe_ops: bool`, default `false`)**:
       - **`expand_moe_ops: false` (Fused Mega-Op Mode, default)**: Wraps routing and routed-expert execution into two battle-tested ops (`qwen3_5_moe_route` + `qwen3_5_moe_grouped_gemm`, proven in `deepseek_coder_v2_lite`) to keep IR graphs compact and eliminate dynamic token-count shape/adapter bugs by default.
       - **`expand_moe_ops: true` (Expanded Atomic-Op Mode, opt-in)**: Keeps `router linear`, `softmax`, `topk`, `silu`, `mul`, and `shared_expert` as native `IRDimops`, wrapping **only** the irreducible dynamic kernels (`qwen3_5_moe_permute`, `qwen3_5_grouped_linear` (`gmm`), `qwen3_5_moe_unpermute`) with `fake_fn`s when fine-grained `EP × ETP` exploration or step-by-step `TrainVerify` inspection is desired.
   - During `trace_strategy="meta"` (or `"cpu"`), `ConcreteTracer` executes `fake_fn` in milliseconds with near-zero RAM/VRAM, while `gencode*.py` emits calls to the real fused runtime kernels!
3. **Compile-Only `parallelize` Invocation**:
   - `param_init_strategy="shard"`, `load_module=False`: skips saving `780 GB` `fullmodel.pt`, skips `torch.distributed`, and emits `gencode0..K-1.py` + `dist_param_map.pt` + `attr_meta`.

---

### Component C: JSON Declarative Strategy Schema (Bridging Agent → `policies.fn`)
Instead of `AutoDist` or brittle integer node IDs (`FwOp1929`), we implement `json_policy.py` in `examples/qwen3p5/` that translates a human- or agent-authored `strategy.json` into `list[nnscaler.policies.OpPlan]`:

```json
{
  "mesh": { "plan_ngpus": 8, "pipeline_nstages": 2, "pipeline_size": 2, "pipeline_nmicros": 4 },
  "tp_size": 4,
  "rewrites": {
    "loss_reduction": "sum-count",
    "wrap_gdn_kernels": true,
    "static_moe_grouped_gemm": true,
    "expand_moe_ops": false
  },
  "stage_splits": [
    { "layers": "0..19", "stage_id": 0 },
    { "layers": "20..39", "stage_id": 1 }
  ],
  "recompute_modules": ["Qwen3_5DecoderLayer"],
  "rules": [
    { "fqn_regex": "embed_tokens|lm_head", "op": "embedding|linear", "partition": { "input": 1, "dim": 0 } },
    { "fqn_regex": "self_attn\\.(q_proj|k_proj|v_proj)", "op": "linear", "partition": { "input": 1, "dim": 0 } },
    { "fqn_regex": "self_attn\\.o_proj", "op": "linear", "partition": { "input": 1, "dim": 1 } },
    { "fqn_regex": "linear_attn\\.in_proj_(qkv|z|b|a)", "op": "linear", "partition": { "input": 1, "dim": 0 } },
    { "fqn_regex": "linear_attn", "op": "qwen3_5_chunk_gated_delta_rule", "partition": { "input": 2, "dim": 2 } },
    { "fqn_regex": "linear_attn\\.out_proj", "op": "linear", "partition": { "input": 1, "dim": 1 } },
    { "fqn_regex": "mlp\\.(gate_proj|up_proj)", "op": "linear", "partition": { "input": 1, "dim": 0 } },
    { "fqn_regex": "mlp\\.down_proj", "op": "linear", "partition": { "input": 1, "dim": 1 } },
    { "fqn_regex": "mlp\\.experts", "op": "qwen3_5_grouped_linear|qwen3_5_moe_grouped_gemm", "partition": { "input": 1, "dim": 0 } },
    { "fqn_regex": ".*", "op": ".*", "partition": "auto" }
  ]
}
```

**Why this fits AI Agents + Verification**:
- **Stable across model scales**: The exact same `rules` work on `0.8B` (`24L`), `27B` (`64L`), and `35B-A3B` (`40L`) without knowing internal `cid`/`tid` numbers.
- **Pre-validated before codegen**: `json_policy.py` checks divisibility (`dim_size % tp_size == 0`, e.g., catching `num_kv_heads = 2` when `tp_size = 4`!) and emits structured JSON errors that the agent can immediately repair.
- **Zero solver latency**: Lowers in `< 2 seconds` via `nnscaler.policies.fn`.

---

### Component D: Refined Agent Auto-Adaptation Loop (No AutoDist, No Full Training)

```txt
[Step 1: Meta Model + Rewrite] ──▶ [Step 2: FX & Pre-PAS IR Capture]
                                                │
       ┌────────────────────────────────────────┘
       ▼
[Step 3: Static IR Audit Report (JSON)]
  - Any untraceable Python loops / dynamic `.nonzero()`? ──▶ Propose Module Rewrite + Wrapper Op
  - Any opaque `IRFwOperation` (non-IRDimops)?           ──▶ Author `register_op` + `fake_fn`
  - Any frozen `^` or indivisible head dims for target TP?──▶ Propose Algebraic Rewrite (`sum-count` / KV-rep)
       │
       ▼ (Once IR is clean & op unit-parity passes on tiny shapes)
[Step 4: Declarative Strategy JSON Generation]
  - Agent emits `strategy.json` (PP stages + TP/EP/SP rules + recompute)
       │
       ▼
[Step 5: Static PAS Compilation & Artifact Export]
  - Lower via `policies.fn` → `IRAdapterGener` → `ModuleCodeGen`
  - Dump standardized bundle for TrainVerify + run optional 2-layer micro-parity check
```

---

### Component E: Standardized Directory & Verification Artifact Layout

```txt
examples/qwen3p5/
├── spec.md                       # Project goal & regulations
├── design_and_patch_plan.md      # This architecture & patch blueprint
├── configs/                      # Saved HF config.json snapshots (0.8B..397B, MoE, VL) — KB-sized!
├── ops/
│   ├── gdn_ops.py                # Wrapper ops + IRDimops + fake_fn for causal_conv1d & FLA
│   ├── moe_ops.py                # Static router + 3D GroupedGEMM EP/TP TransformRule + fake_fn
│   └── vision_ops.py             # Stage 3: varlen ViT & masked_scatter wrappers
├── rewrites/
│   ├── loss_rewrite.py           # mean -> sum / valid_count algebraic rewrite
│   └── moe_rewrite.py            # Swaps Qwen3_5MoeSparseMoeBlock dynamic loop -> static moe_ops
├── policy/
│   ├── schema.py                 # JSON/YAML strategy schema & validator
│   ├── json_pas.py               # Compiles strategy.json -> list[OpPlan] -> policies.fn
│   └── strategies/               # Human/Agent strategies (dense_tp4.json, moe_ep4_pp2.json, ...)
├── audit/
│   ├── ir_auditor.py             # Structured report: non-IRDimops, frozen dims, adapter summary
│   └── artifact_dumper.py        # Exports FX, pre-PAS IR, post-PAS IR, adapter manifest, gencode
├── tests/
│   ├── test_op_split_parity.py   # Tiny-shape per-op 1-GPU vs 4-GPU split equivalence
│   └── test_micro_compile.py     # 2-layer Dense/MoE compile + micro-parity tests
└── compile.py                    # Single CLI entry: config + strategy.json -> artifacts/<model>/
```

Each `compile.py` invocation produces a self-contained **Verification Bundle** ready for downstream formal verification (`TrainVerify`):
- `00_model_and_strategy.json` (resolved HF config + strategy JSON + parameter shard map)
- `01_fx_graph.txt` & `01_fx_nodes.json` (captured FX graph with shapes/dtypes/FQNs)
- `02_ir_pre_pas.txt` & `02_ir_audit.json` (logical `IRGraph` before partitioning + op-legality audit)
- `03_ir_post_pas.txt` & `03_adapters.json` (physical partitioned `IRGraph` + every `IRAdapter` collective)
- `04_generated/gencode0..N-1.py` (per-rank executable Python modules)

---

## 4. Phased Implementation Roadmap

1. **Phase 0 — Core Surgical Patches (`nnscaler/`)**:
   - Port `MaskedFill` broadcast fix, `Softplus` op + mapping, `torch.memory_format` emit, and add the 4-line `MetaStrategy` meta-parameter guard (`tests/` unit-tested).
2. **Phase 1 — Framework Skeleton + Stage 1 Dense (`0.8B` → `27B`)**:
   - Implement `ops/gdn_ops.py` (wrapper + `fake_fn`), `policy/json_pas.py`, `audit/ir_auditor.py`, and `compile.py`.
   - Verify weightless compilation and artifact export across `0.8B`, `4B`, `9B`, and `27B` configs (`TP`, `PP`, tied vs. untied embeddings, GQA `KV` divisibility checks).
3. **Phase 2 — Stage 2 MoE (`35B-A3B`, `122B-A10B`, `397B-A17B`)**:
   - Implement `rewrites/moe_rewrite.py` + `ops/moe_ops.py` with **`expand_moe_ops: false` as the default** (fused `moe_route` + `moe_grouped_gemm` mega-op mode to avoid dynamic token-count adapter bugs) and opt-in **`expand_moe_ops: true`** (native router/SwiGLU `IRDimops` + atomic `moe_permute` / `grouped_linear` / `moe_unpermute` primitives for `EP × ETP` & `TrainVerify`), all backed by `fake_fn`.
   - Compile `35B-A3B` and `397B-A17B` end-to-end on `meta` tensors into full FX/IR/`gencode*.py` bundles under both modes, and validate math equivalence on a 2-layer `8-expert` micro-MoE.
4. **Phase 3 — Stage 3 Multimodal (`Qwen3_5ForConditionalGeneration`)**:
   - Register `masked_scatter` and static/bucketed ViT `cu_seqlens` wrappers to emit unified vision+text IR graphs.
