# Combined 1F1B D：性能与拓扑复核记录

本记录对应分支 `yomia/combined-1f1b-d-performance-topology`。原始、可复算的
rank-max 延迟样本在：

- `docs/phase_acceptance_raw.json`
- `docs/phase_acceptance_raw.csv`
- 生成器：`benchmark/phase_acceptance_matrix.py`

不会提交 CUPTI/Chrome 大型 trace；`tests/parallel_module/test_phase_moe_profile.py`
可按需生成其文本摘要与 NVTX/`torch.profiler` 范围。

## C phase：实际测量与结论

### 协议

四方比较使用相同 2-GPU、单 stage EP2、相同模型/输入规模：

1. `serial`：`66616d64` 的非 phase 路径；
2. `c-baseline`：`71af7929`；
3. `d-old`：`6e1268b4`；
4. `d-new`：`66616d64`（PhaseExecutor/选择性同步提交）。

每个 cell 都是新 `torchrun`，每步在 CUDA 同步后记录**所有 rank 的最大
latency**；每轮取三个 timed step 的 median。每个 small/large scale 做八轮，
以随机初始顺序的 Latin rotation 平衡四个 ordinal 位置；两个 warmup 不计入。

| scale | serial median / MAD | C median / MAD | D-old median / MAD | D-new median / MAD |
|---|---:|---:|---:|---:|
| small | 38.408 / 1.779 ms | 69.516 / 5.762 ms | 72.097 / 12.093 ms | 63.625 / 9.405 ms |
| large | 85.509 / 3.444 ms | 147.448 / 18.473 ms | 128.593 / 6.972 ms | 137.425 / 16.812 ms |

这些数字**不能支持稳定性能收益的声明**。例如 D-old minus D-new 的 paired
median / MAD 为：small `+9.736 / 7.198 ms`、large `+6.099 / 21.295 ms`；二者
均未达到 3×MAD 门槛。D-new 仍然明显慢于 serial（small paired median
`-27.732 ms`，large `-50.246 ms`）。因此不把 PhaseExecutor 宣称为端到端
加速，保留它为有完整 A/B 开关的安全 fast path，并把数据保留供后续优化。

### 已验证的开销分解

`NN_SCALER_RUN_PHASE_PROFILE=1` 的 CUPTI-backed `torch.profiler`、NVTX 与
`cProfile` harness 显示：serial 在 profile workload 有 4 个 forward/4 个
backward executor 边界；phase 路径有 32/32。`c10d::alltoall_base_` 与
`aten::empty` 数量本身并不唯一属于 phase；重复的 autograd-engine backward、
Python boundary 和最终 CUDA wait 才是结构性主因。

`NN_SCALER_RUN_PHASE_MICROBENCH=1` 的小 CPU benchmark 对直接绑定的
`PhaseExecutor.forward/backward` 与 generic executor 得到约 `0.995x`（接近
持平），因此同样不声称 micro 加速。它的价值是替换 string/FIFO metadata
路径、给后续 profile 一个明确 A/B 控制，而非伪造性能结论。

### 安全实施

- `DeviceGroup.get_stream('default')` 返回真正 CUDA default stream。
- `StreamContext(stream=None)` 的语义是调用者 current stream；显式
  `stream='default'` 的语义是**无论调用者 current stream 是什么，都进入
  CUDA default stream**。2-GPU 测试在 non-default caller stream 内调用
  generated `train_step` 并验证每个 segment hook 观察到 default stream。
- 仅对真正 named consumer stream 的输入发射 `record_stream`；消除 self-wait。
- PhaseMoE 的默认路径没有 synthetic phase stream context；dedicated dispatch
  stream 仅作为 benchmark ablation。
- `Executor` FIFO 改为 `deque`，无 pending async work 时避免 completed-work
  scan。
- `PhaseExecutor` 为每个 `(microbatch, physical-stage, layer, phase)` 分配
  model-owned integer slot，缓存 detach/alias schema，保持每 phase 独立
  autograd graph/backward、hook、recompute、reducer 与 lifecycle。只有已知
  不消费 pending async tensor 的 attention/dispatch phase 跳过输入同步；
  expert/combine consumer 保留 wait-before-detach。完整 phase-wide bypass 曾
  泄漏 handler/lifecycle，已拒绝。

不融合 Attention+Dispatch 或 Expert+Combine：前者会合并可独立调度的
Dispatch/Attention backward island，后者会抹掉 combine issue 到 wait 的窗口；
目前依赖不允许把它们作为安全 super-phase 宣称。

## PP2×EP2 replica activation

RVD 的 `R` 表示**相等副本**，不能从静态 IR 自动推断 runtime rank input 是否
不同。PhaseMoE PP×EP policy 现在要求显式：

```python
make_pas(..., pp_replica_semantics='equal')
# 或
make_pas(..., pp_replica_semantics='independent')
```

遗漏声明会 fail-fast；这避免 policy 层静默把 rank-distinct lane 当 equal replica。
`equal` 保留 legacy RVD 语义。`independent` 会标记边界：若 producer/consumer
RVD grid 的每个 cell 的 device、`indmap`、`valmap` 完全 identity（含 active
backward mapping），`ConcurrentGener` 返回 `None`，不生成 adapter、不 fail-fast。
这有 unit 与 generated-code coverage。任意 device permutation、disjoint PP
mesh 或 value/layout 改变仍 fail-fast，绝不落入 destination `all_gather`。

通用 IR 对未标记 runtime-rank-distinct 数据无法可靠自动检测；所以 explicit
policy 是边界，而不是宣称 core 能猜出数据相等性。真正 disjoint bijection 尚未
支持，因而没有虚假的 PP2×EP2 非对称 numerical/optimizer reference claim。

后续 MVP 需要 stable：

```text
(boundary, direction, lane-id, tensor-slot, microbatch, global-ordinal)
```

并把 `RecvIssue/RecvWait/SendIssue/SendWait` 作为一等 IR，使用共享 PG 上同一
全序投影，才能安全实现 `src[lane] -> dst[lane]` 与 reverse gradient。当前
fail-fast 和设计覆盖正是为了阻止在此之前错误声称已支持。

## 全 physical stage 调度

`PhaseAwareSched.sched_1f1b_global_phase_aware` **覆盖所有 physical stage；
只有 ready window 存在时才交织**。它显式加入 phase-forward、mirrored-backward、
local F→B、upstream activation 与 downstream gradient 边；不再以“本地提前
F(m+1)”或“本地推迟 B(m)”猜测依赖。

对 PP4、unit-span phase、6 microbatch 的实际 schedule：rank 0/1 的 F/B
transition 均为 1（完整 F warmup 后完整 B cooldown），rank 2 为 19、rank 3
为 41。下界是 whole-phase 模型中的：

```text
start(F_s(m+1)) >= finish(F_{s-1}(m+1))
start(B_s(m))   >= finish(B_{s+1}(m))
```

并且每个 rank 一次只能执行一个 phase。rank 0 在 downstream gradient 回到前
已经耗尽可用 F；rank 1 同时受上述 upstream activation 与 downstream gradient
critical path 约束，故没有第二个 ready whole phase 可填空。该结论只适用于当前
“adapter 在 sender 后插入”的 whole-phase IR，不是声称 transport 层永无窗口。

显式零输入 `RecvIssue` 早发可能创造新窗口，但当前 `SchedulePlan` 没有独立
issue/wait node；在没有全局 lane ordinal 前不安全实现。本轮保留该限制并由
PP2/PP4 CPU sweep、2-GPU PP2、4-GPU PP2/PP4 repeated/numeric tests 覆盖。

## Multi-peer A

world NCCL PG 的 untagged P2P 有全局 ordering domain。生产路径仍是
`GlobalCommSchedule` 的单一全局拓扑排序；当发现 2+ peer pair 时 fail-closed
到每 device 单 communication chain。4-GPU、4-stage、3 peer-pair 的 repeated
no-deadlock、numeric、channel/cap 和 property tests 已重跑。

新增 opt-in `tests/runtime/test_pair_pg_diagnostic.py`：每个 rank 按相同排序
预创建所有 pair PG，使用 `group_peer`（而非 world rank），每 ordinal post 全部
本地 ops 后 wait，并验证 20 个 asymmetric ordinal。PP2 disjoint pair 与 PP4
overlapping pair 两种 topology 都通过。

这只证明静态 pair-PG 创建和 group-local peer API 可以工作；**没有**把它接入
动态 `GlobalCommSchedule`。历史动态 pipeline failure 的根因仍未被充分归因，
不能把“pair PG”称为已验证的替代方案。更好的下一步仍是 stable ordinal + joint
posting/first-class issue-wait；当前 fallback 的安全理由不变。

## 验证入口

- CPU：phase/codegen/generator/executor/global schedule suites。
- 2 GPU：phase numeric/lifecycle、explicit default stream、profile。
- 4 GPU：PP2/PP4 global phase、PP2×EP2 equal semantics、3-peer A regression、
  pair-PG diagnostic（opt-in）。
- Hygiene：`git diff --check`、`py_compile`、secret-pattern review、clean status。
