# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

nnScaler is a parallelization engine that compiles single-GPU PyTorch DNN models into multi-GPU parallel programs. Published at OSDI 2024 (internal codename: "CUBE"). Python >= 3.9, < 3.11 (3.10 recommended); PyTorch >= 2.0, < 2.4 (2.2.0 recommended).

## Build & Install

```bash
pip install -r requirements.txt
pip install -e .
```

The build compiles one C++ pybind11 extension: `nnscaler.autodist.dp_solver` (from `nnscaler/autodist/dp_solver.cpp`).

## Testing

```bash
# Run all tests (uses tox with Python 3.10):
tox

# Recreate tox venv after requirements change:
tox -r

# Run a single test file:
pytest tests/path/to/test_file.py

# Run a specific test:
pytest tests/path/to/test_file.py::test_function_name
```

CI (Azure DevOps) has no GPU. Tests must either:
- Use `@replace_all_device_with('cpu')` decorator from `tests/utils.py` to run on CPU
- Be skipped with `@pytest.mark.skipif(not torch.cuda.is_available(), reason='lack of gpu devices')`

Multi-process tests use `torchrun` via `tests/launch_torchrun.py` — it's slow, minimize its use. The `conftest.py` auto-cleans generated files (`gencode*.py`, model attributes) after each test.

## Code Style

Google Style Python Docstrings. Formatter: yapf (no custom config committed).

## Architecture

The compilation pipeline flows through these stages:

1. **Tracing/Parsing** (`graph/tracer/`, `graph/parser/`): Concrete tracer (torch.fx-based) converts a PyTorch model into an `IRGraph` via `FxModuleParser`.

2. **IR** (`ir/`): Core data model. `IRFullTensor` = logical full tensors; `IRSubTensor` = partitioned slices. `IRFwOperation`/`IRBpOperation` = forward/backward operators. `IRAdapter` = data redistribution nodes.

3. **DimOps** (`graph/function/dimops.py`): Dimension annotation system for describing how operators can be partitioned. Annotations like `'m k+, n k+ -> m n'` specify spatial (`''`), reduction (`'+'`), and non-partitionable (`'^'`) dimensions.

4. **Policy (PAS)** (`policies.py`): Users provide a Partition-Assign-Schedule policy that transforms the `IRGraph`. Steps: multiref for shared tensors, optional recomputation, pipeline staging, partition & assign to devices, schedule. Built-in policies are prefixed `pas_`.

5. **AutoDist** (`autodist/`): Automatic parallelization solver. SPMD solver explores tensor/operator parallelism; pipeline solver handles pipeline parallelism; C++ DP solver (`dp_solver.cpp`) accelerates search.

6. **Adapter Generation** (`graph/gener/`): Inserts communication adapters (allreduce, reduce-scatter, allgather) between partitioned operators. Uses RVD (Replicate-Value-Distribute) primitives.

7. **Scheduling** (`graph/schedule/`): Determines temporal execution order (e.g., 1F1B, interleaved 1F1B for pipeline parallelism).

8. **Execution Plan** (`execplan/`): Lowers scheduled graph into `ExecutionPlan`. Passes: `DiffFusion` (communication optimization), `Grouping` (computation grouping).

9. **Code Generation** (`codegen/`): Emits `gencode{rank}.py` files — one per GPU rank. `ModuleCodeGen` for spatial/module code, `ScheduleCodeGen` for temporal/schedule code.

10. **Runtime** (`runtime/`): `ParallelModule`/`CubeModule` wraps generated code as a PyTorch module. Handles gradient reduction (ZeRO stages 0/1/3), mixed-precision training, collectives.

11. **Trainer/CLI** (`cli/`): High-level `Trainer` with `TrainerArgs` (YAML or code), checkpointing, logging (TensorBoard/W&B), hook system. Entry point: `nnscaler-train`.

## Key APIs

- **`parallelize()`** (`nnscaler/parallel.py`): Modern API. Takes model + `ComputeConfig`, returns `ParallelModule`.
- **`@compile`** (`nnscaler/compiler.py`): Legacy decorator-based API.

## Environment Flags

Controlled via environment variables in `nnscaler/flags.py`:
- `TRACE_STRATEGY`: How to execute functions during tracing (`cpu`, `cuda`, `meta`, `cuda_run_cpu_offload`, `reuse_cache`). Default: `cuda_run_cpu_offload`.
- `USE_ZERO`: ZeRO optimization level (0/1/3).
- `ASYNC_REDUCER`: Overlap gradient sync with backward computation.
- `ASYNC_COMM`: Async communication.
- `USE_AMP`: Automatic mixed precision.
- `SINGLE_DEV_MODE`: Single-device debug mode (run with `python` instead of `torchrun`).

## Key Directories

| Directory | Purpose |
|---|---|
| `nnscaler/ir/` | Intermediate representation (tensors, operators, adapters) |
| `nnscaler/graph/` | Graph construction, tracing, parsing, scheduling, adapter generation |
| `nnscaler/algorithm/` | Partitioning algorithms for operators |
| `nnscaler/autodist/` | Automatic distribution solver (SPMD + pipeline + C++ DP solver) |
| `nnscaler/codegen/` | Code generation for parallelized modules |
| `nnscaler/execplan/` | Execution plan and optimization passes |
| `nnscaler/runtime/` | Runtime: ParallelModule, reducers, collectives, mixed-precision |
| `nnscaler/cli/` | Trainer, TrainerArgs, CLI entry point |
| `nnscaler/customized_ops/` | Custom ops: ring/zigzag/sliding-window attention, chunked cross-entropy |
| `nnscaler/policies.py` | Built-in PAS parallelization policies |
| `nnscaler/flags.py` | Environment variable compile/runtime flags |
| `examples/` | Example models (llama, nanogpt, deepseek, etc.) |
