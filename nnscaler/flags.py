#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""
Environment flags for compiling options
"""

import os


def _to_bool(s: str, default=False) -> bool:
    val = os.environ.get(s, default=default)
    return bool(int(val))


def _to_int(s: str, default=0) -> int:
    val = os.environ.get(s, default=default)
    return int(val)


class CompileFlag:
    # ================ compiling ========================
    # worker sleep in seconds
    worker_sleep = _to_int('WORKER_SLEEP')
    disable_intra_rvd = _to_bool('DISABLE_INTRA_RVD')
    disable_inter_rvd =  _to_bool('DISABLE_INTER_RVD')
    disable_comm_fusion = _to_bool('DISABLE_COMM_FUSION')

    visualize_plan = _to_bool('VISUALIZE_PLAN')

    # ============ code generation ===============
    use_nnfusion = _to_bool('USE_NNFUSION')
    use_jit = _to_bool('USE_JIT')
    disable_code_line_info = _to_bool('DISABLE_CODE_LINE_INFO')  # will add original code information in generated code, note that this will make trace slow
    # how to execute the functions during trace, available choices ['cpu', 'cuda', 'meta', 'cuda_run_cpu_offload', 'reuse_cache']
    trace_strategy = os.environ.get('TRACE_STRATEGY', default='cuda_run_cpu_offload')
    # reduce scatter adapter can reduce the communication cost, and improve the performance
    # but sometimes it may cause communication bugs, so we provide an option to enable/disable it
    disable_reduce_scatter_adapter = _to_bool('DISABLE_REDUCE_SCATTER_ADAPTER', False)

    # whether to reschedule the operators (compute and communication ops) inside each
    # forward segment before code generation.
    # When enabled, a data-dependency graph is built among the ops of a segment so that
    # any topological order of the graph is a legal (causally-correct) execution order,
    # and the operators are reordered accordingly. Default is False (original op order).
    enable_op_reschedule = _to_bool('ENABLE_OP_RESCHEDULE', False)
    # scope of the operator rescheduling: 'segment' (reorder ops inside each forward
    # segment), 'sequence' (reorder the cross-segment execution sequence so that e.g.
    # asynchronous communication can be issued early), or 'both'. Default 'segment'.
    op_reschedule_scope = os.environ.get('OP_RESCHEDULE_SCOPE', default='segment')
    # whether to also reschedule pipeline (graph.sched) schedules at the 'sequence'
    # scope. The deliberate compute order is preserved (only communication adapters
    # move), but enabling it changes the generated pipeline code, so it is opt-in and
    # should be validated at runtime. Default False (pipeline schedules untouched).
    op_reschedule_pipeline = _to_bool('OP_RESCHEDULE_PIPELINE', False)
    # path to a schedule config file (produced by `dump_schedule`) that records the
    # desired operator order per forward segment. When set together with
    # `enable_op_reschedule`, the operators are reordered to follow the recorded order
    # (data dependencies are always enforced). Default '' (use the original order).
    op_reschedule_config = os.environ.get('OP_RESCHEDULE_CONFIG', default='')
    # path to dump the current operator schedule to (one entry per forward segment).
    # When set, the schedule is written after code generation planning so it can be
    # edited and fed back via `OP_RESCHEDULE_CONFIG`. Default '' (do not dump).
    dump_op_schedule = os.environ.get('DUMP_OP_SCHEDULE', default='')
    # path to dump a Graphviz visualization of the operator schedule to (ops laid out
    # linearly in their current order with dependency arrows). When set together with
    # `enable_op_reschedule`, both the "before" and "after" graphs are written
    # (with `.before`/`.after` inserted before the extension) for comparison.
    # Default '' (do not dump).
    dump_op_schedule_graph = os.environ.get('DUMP_OP_SCHEDULE_GRAPH', default='')

    # ============== runtime ====================
    dev_mode = _to_bool('SINGLE_DEV_MODE')  # allow to use python xx.py
    async_comm = _to_bool('ASYNC_COMM')
    line_timer = _to_bool('LINE_TIMER')

    # ============== reducer ==================
    # use zero optimization on optimizer status.
    # to cooperate with zero, user needs to call `model.parameters_for_optimizer()`
    # to get parameters for optimizer, and `model.gather_params()` after `optimizer.step()`
    use_zero = _to_int('USE_ZERO')
    # use async communication to overlap gradient synchronization and backward computation
    async_reducer = _to_bool('ASYNC_REDUCER')  # use async reducer
    # maximal reducer weight bytes for one allreduce (only effective for async):
    # default 0 means using the default value in reducer
    max_reducer_bucket = _to_int('MAX_REDUCER_BUCKET', default=0)
    # perform reducer op on gradients, can be sum, avg, mean, max, min. Default is sum
    reducer_op = os.environ.get('REDUCER_OP', default='sum')
    # zero_ngroups is the number of subgroups in each original ZeRO gruop (e.g., weights reducer)
    # ZeRO subgroup is obtained by dividing the original ZeRO group by zero_ngroups
    # it helps reduce communication cost of allgather weights in ZeRO, but increase the weights'
    # optimization states on each GPU.
    zero_ngroups = _to_int('ZERO_NUM_GROUPS', default=1)
    # whether to use reduce scatter for zero (default False).
    # By default we use `allreduce` for zero, which is due to
    # 1) `reduce_scatter` will make some parameters have stale gradient after synchronization,
    #    hence break the consistency of `.data` and `.grad` of parameters. Need to be careful when using optimizer.
    # 2) `reduce_scatter`` doesn't significantly improve performance comparing with `allreduce`.
    zero_use_reduce_scatter = _to_bool('ZERO_USE_REDUCE_SCATTER')
    # whether to use parameter level sharding in zero (default False).
    # This option controls the granularity of sharding parameters in ZeRO.
    # If set to True, gradients/parameters/optimizer states will be sharded at parameter level.
    # If set to False, they will be sharded at element level.
    # NOTE: parameter level sharding may introduce paddings
    # to make sure all devices have the same size of tensor, which may waste some memory.

    # You must set it to True when Muon optimizer is used.
    zero_param_level_sharding = _to_bool('ZERO_PARAM_LEVEL_SHARDING')

    # whether to generate weight reducers for replicated weights.
    # When True, replicated weights will also go through all-reduce and divide by nreplicas,
    # ensuring gradient consistency across ranks. Default is False (original behavior).
    reducer_replicated_params = _to_bool('REDUCER_REPLICATED_PARAMS')

    # use automate mixture precision training, where weights, gradients
    # and optimizer status are kept in its original data type (can be float32),
    # but some of the forward operators will be converted to float16.
    use_amp = _to_bool('USE_AMP')


class RuntimeFlag:

    # if True, skip model.zero_grad().
    # when applying gradient accumulation,
    # this flag should be set to True at the first accumulation step,
    # and set to False at other accumulation steps.
    # By default False, which means the gradients of parameters in the reducers
    # will be zeroed at the beginning of every iteration.
    skip_zero_grad: bool = False

    # if True, skip reducer.sync_grads().
    # when applying gradient accumulation,
    # this flag should be set to True at the last accumulation step,
    # .and set to False at other accumulation steps.
    # By default False, which means the gradients will be reduced at the end of every iteration.
    skip_reducer: bool = False
