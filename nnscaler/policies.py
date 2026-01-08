#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""
Policy Writing Guidelines

Users can write the policy following the steps:

1. Apply multiref
    If all consumers of a full tensor consume the same subtensor (the partitions are exactly the same), we can skip this step.
2. Apply recompute (if needed)
3. Graph staging (pipeline only)
4. Graph partition & assign
5. Apply schedule (pipeline only)

Note the steps 1, 2, 3 must be finished before any graph partition.

IRDataOperation is recommended to be replicated to all devices.
"""

import ast
from dataclasses import dataclass, field
import logging
from typing import Any, List, Literal, Optional, TYPE_CHECKING, Callable, Iterable, Union
import random

import torch
import more_itertools as mitr

from nnscaler.autodist.apis import parallelize_graph
from nnscaler.autodist.autodist_config import AutoDistConfig
from nnscaler.graph import IRGraph
from nnscaler.graph.function.anchor import IRGraphAnchor
from nnscaler.graph.function.dimops import IRDimops
from nnscaler.graph.function.pyfunc import IRPyFunc
from nnscaler.graph.segment import IRSegment
from nnscaler.ir.operator import IRBpOperation, IRDataOperation, IRFwOperation
from nnscaler.ir import IRCell, IRSubTensor, IRFullTensor
from nnscaler.ir.cten import IR
from nnscaler.runtime.function import identity, multiref
from nnscaler.utils import load_type


if TYPE_CHECKING:
    from nnscaler.parallel import ComputeConfig, ParallelModule


_logger = logging.getLogger(__name__)


def _tp(graph: IRGraph, node: IRDimops, devs: List[int], idx: int, dim: int):
    if len(devs) > 1:
        sub_nodes = graph.partition(
            node, node.algorithm('dim'), idx=idx, dim=dim, num=len(devs))
    else:
        sub_nodes = [node]
    for devid, sub_node in zip(devs, sub_nodes):
        graph.assign(sub_node, devid)
    return sub_nodes


def _replica(graph: IRGraph, node, devs: List[int]):
    sub_nodes = graph.replicate(node, times=len(devs))
    for devid, sub_node in zip(devs, sub_nodes):
        graph.assign(sub_node, devid)
    return sub_nodes


def is_tensor_in_output(t: IRFullTensor, graph: IRSegment) -> bool:
    for output in IRCell.get_objects_from_complex(graph.outputs()):
        if isinstance(output, IRSubTensor) and output.parent == t:
            return True
    return False


def auto_multiref(graph: IRGraph):
    for ftensor in graph.full_tensors():
        if ftensor.is_grad(): continue
        in_output = int(is_tensor_in_output(ftensor, graph))
        if len(graph.consumers(ftensor)) + in_output > 1:
            graph.multiref(ftensor, comment='auto_multiref')


def pas_dp(graph: IRGraph, cfg: 'ComputeConfig'):
    """
    pure data parallelism policy
    """
    ngpus = cfg.plan_ngpus
    if ngpus != 1:
        raise ValueError("Data parallelism only supports 1 plan GPU")

    # no partition is done, so we can skip multiref safely
    for node in graph.select(ntype=(IRFwOperation, IRDataOperation)):
        _replica(graph, node, [0])
    return graph


def pas_tp(graph: IRGraph, cfg: 'ComputeConfig'):
    """
    random tensor parallelism inside a scale unit, and dp across scale units
    """
    ngpus = cfg.plan_ngpus
    pas_cfg = cfg.pas_config
    enable_random_replicated = pas_cfg.get('enable_random_replicated', False)
    # get the current random state
    state = random.getstate()

    seed = cfg.pas_config.get('seed', 1)  # by default we fix the seed for test reproducibility
    random.seed(seed)
    devs = list(range(ngpus))

    auto_multiref(graph)

    for node in graph.select(ntype=(IRFwOperation, IRDataOperation)):
        if node.name == 'multiref' or isinstance(node, IRGraphAnchor):
            continue
        if isinstance(node, IRDimops):
            configs = node.transform_space()
            if len(configs) == 0 or (enable_random_replicated and random.random() < 0.5):
                _replica(graph, node, devs)
            else:
                configs = sorted(configs, reverse=True,
                                key=lambda config: node.input(config[0]).shape[config[1]])
                random.shuffle(configs)
                for (idx, dim) in configs:
                    if node.input(idx).shape[dim] % len(devs) != 0: continue
                    # only partition when all input tensors are constant on this dim
                    if not node.input(idx).dim_tracks[dim].is_constant: continue
                    if node.algorithm('dim').satisfy(idx=idx, dim=dim, num=len(devs)):
                        _tp(graph, node, devs, idx, dim)
                        break
                else:
                    _replica(graph, node, devs)
        else:
            _replica(graph, node, devs)

    # restore the random state
    random.setstate(state)
    return graph


def pas_pp(graph: IRGraph, cfg: 'ComputeConfig'):
    """
    pipeline parallelism inside a scale unit, and dp across scale units
    """
    nstages = cfg.pas_config.get('pipeline_nstages', 'auto')
    if nstages != 'auto' and nstages != cfg.plan_ngpus:
        raise ValueError("pas_pp requires pipeline_nstages == plan_ngpus")
    return pas_hybrid(graph, cfg)


def pas_data(graph: IRGraph, env_resource: 'ComputeConfig'):
    """
    tensor partition on batch dimension inside a scale unit, and dp across scale units
    """
    ngpus = env_resource.plan_ngpus
    auto_multiref(graph)

    batch_dim = 0
    for dl in graph.select(ntype=IRDataOperation):
        _replica(graph, dl, list(range(ngpus)))

    for node in graph.nodes():
        if isinstance(node, IRFwOperation):
            try:
                algo = node.algorithm('dim')
                idx = 0
                sub_nodes = graph.partition(
                    node, algo, idx=idx, dim=batch_dim, num=ngpus)
            except Exception:
                sub_nodes = graph.replicate(node, ngpus)

            for idx, node in enumerate(sub_nodes):
                graph.assign(node, idx)
    return graph


def pas_hybrid(graph: IRGraph, cfg: 'ComputeConfig'):
    """
    pipeline and tensor parallelism inside a scale unit, and dp across scale units
    """
    if not cfg.use_end2end:
        raise ValueError("Hybrid policy only supports end2end module")

    ngpus: int = cfg.plan_ngpus
    nstages = cfg.pas_config.get('pipeline_nstages', 'auto')
    if nstages == 'auto':
        nstages = cfg.plan_ngpus
    nmicros = cfg.pas_config['pipeline_nmicros']
    scheduler = cfg.pas_config.get('pipeline_scheduler', '1f1b')
    pp_size = cfg.pas_config.get('pp_size', nstages)

    if nstages % pp_size != 0:
        raise ValueError(f'invalid pp_size {pp_size} for nstages {nstages}')
    if ngpus % pp_size != 0:
        raise ValueError(f'invalid pp_size {pp_size} for ngpus {ngpus}')
    tp_size = ngpus // pp_size


    auto_multiref(graph)
    fnodes = graph.select(ntype=IRFwOperation)
    stages = mitr.divide(nstages, fnodes)
    stages = [list(s) for s in stages]
    for idx, stage in enumerate(stages):
        _logger.info(f'> stage {idx}: {stage[0]}')
    graph.staging([s[0] for s in stages])

    stages: List[IRSegment] = graph.select(ntype=IRSegment, flatten=False)
    stages = [s for s in stages if s.isfw()]
    assert len(stages) == nstages, "Internal Error"

    # stage-wise tensor parallelism
    curr_devices = list(range(ngpus))
    for idx, stage in enumerate(stages):
        idx = idx % pp_size
        devs = curr_devices[idx * tp_size: (idx + 1)* tp_size]
        for node in stage.nodes():
            try:
                _tp(graph, node, devs, idx=0, dim=0)
            except Exception as e:
                _replica(graph, node, devs)

    # replicate dataloader
    for dl in graph.select(ntype=IRDataOperation):
        _replica(graph, dl, devs=list(range(ngpus)))

    cfg.apply_pipeline_scheduler(graph, nstages, nmicros, scheduler)
    return graph


def pas_autodist(graph: IRGraph, cfg: 'ComputeConfig') -> IRGraph:
    from nnscaler.autodist.util import get_default_profile_path

    pas_cfg = cfg.pas_config

    update_freq = pas_cfg.get('update_freq', 1)
    if isinstance(update_freq, (tuple, list)):
        update_freq = update_freq[0]

    # optional parameters

    # Note we don't directly pass pipeline_nstages to autodist.
    # when `pipeline_nstages == 'auto'`, we will check if there are options incompatible with pipeline.
    # if we find incompabible options (here use_async_reducer and pipeline_pivots),
    # we will disable pipeline effectively by setting it to 1.
    pipeline_nstages = pas_cfg.get('pipeline_nstages', 'auto')

    if pipeline_nstages == 'auto':
        if not pas_cfg.get('pipeline_pivots'):
            pipeline_nstages = 1
        if not cfg.use_end2end or cfg.use_async_reducer:
            pipeline_nstages = 1
    elif pipeline_nstages > 1:
        # the user manually enabled pipeline, should not disable, so raise
        if not pas_cfg.get('pipeline_pivots'):
            raise ValueError("pipeline_pivots must be set to enable pipeline")
        if not cfg.use_end2end:
            raise ValueError("explore_pipeline cannot be enabled if use_end2end is False")
        if cfg.use_async_reducer:
            raise ValueError("explore_pipeline cannot be enabled if use_async_reducer is True")
    else:
        if pas_cfg.get('pipeline_pivots'):
            raise ValueError("pipeline_pivots must not be set because pipeline is disabled by pipeline_nstages<=1")

    pipeline_scheduler = pas_cfg.get('pipeline_scheduler', '1f1b')
    if pipeline_scheduler != '1f1b':
        raise ValueError(f"Only 1f1b scheduler is supported in autodist.")

    mesh_col = pas_cfg.get('max_partition_degree', cfg.plan_ngpus)
    if cfg.plan_ngpus % mesh_col != 0:
        raise ValueError(f"plan_ngpus {cfg.plan_ngpus} should be divisible by max_partition_degree {mesh_col}")
    mesh_row = cfg.plan_ngpus // mesh_col
    if pipeline_nstages == 1 and mesh_row != 1:
        raise ValueError("mesh_row should be 1 if pipeline is not enabled")
    memory_constraint = pas_cfg.get('mem_constraint', -1)
    task_name = pas_cfg.get('task_name', '_')
    use_memory_efficient_fp16 = pas_cfg.get('use_memory_efficient_fp16', False)
    use_memory_efficient_bf16 = pas_cfg.get('use_memory_efficient_bf16', False)
    use_fp16 = pas_cfg.get('use_fp16', use_memory_efficient_fp16)
    use_bf16 = pas_cfg.get('use_bf16', use_memory_efficient_bf16)
    profile_dir = pas_cfg.get('profile_dir', None)
    if profile_dir is None:
        profile_dir = get_default_profile_path()
    re_profile = pas_cfg.get('re_profile', False)
    verbose = pas_cfg.get('verbose', False)
    load_plan_path = pas_cfg.get('load_plan_path', None)
    save_plan_path = pas_cfg.get('save_plan_path', None)
    partition_constraints_path = pas_cfg.get('partition_constraints_path', '')
    recompute_modules = pas_cfg.get('recompute_modules', '')
    recompute_ratio = pas_cfg.get('recompute_ratio', 1.0)
    pipeline_pivots = pas_cfg.get('pipeline_pivots', '')
    max_pipeline_bubble_ratio = pas_cfg.get('max_pipeline_bubble_ratio', 0.2)
    max_pipeline_unbalance_ratio = pas_cfg.get('max_pipeline_unbalance_ratio', 0.5)
    use_apex_fused_adam_v2 = pas_cfg.get('use_apex_fused_adam_v2', False)
    parallel_profile = pas_cfg.get('parallel_profile', True)
    transient_mem_coef = pas_cfg.get('transient_mem_coef', 2)
    disable_shared_param_constraint = pas_cfg.get('disable_shared_param_constraint', False)
    solver = pas_cfg.get('solver', 'dp')

    task_name = f'{task_name}_{cfg.plan_ngpus}gpus_{update_freq}update_freq'
    if memory_constraint == -1:
        # consider memory fragmentation and other buffers, use 80% of the memory
        memory_constraint = int(0.8 * torch.cuda.mem_get_info()[1] / 1024 /
                                1024 / 1024)
    if cfg.use_zero:
        zero_stage = 1
        zero_ngroups = cfg.zero_ngroups
    else:
        zero_stage = 0
        zero_ngroups = 1
    if use_fp16 or use_bf16:
        support_inkernel_cast = use_apex_fused_adam_v2
        if use_memory_efficient_fp16 or use_memory_efficient_bf16:
            # Check fairseq/optim/fused_adam.py
            # If memory efficient:
            # Considered in opt_resident_mem: fp32 moment1, fp32 moment2.
            # Considered in opt_transient_mem: fp32 weight, fp32 gradient,
            # because fp16 weight and gradient are casted to fp32.
            # Here weight_mem is in fp16, so multiply by (2+2).
            opt_resident_coef = 4
            opt_transient_coef = 0 if support_inkernel_cast else 4
        else:
            # If not memory efficient:
            # Considered in opt_resident_mem: fp32 moment1, fp32 moment2, fp32 weight.
            # Considered in opt_transient_mem: fp32 gradient,
            # because fp16 gradient are casted to fp32.
            # Here weight_mem is in fp16, so multiply by (2+2+2).
            opt_resident_coef = 6
            # inkernel cast between fp32 weight and fp16 grad has not support
            opt_transient_coef = 2 if support_inkernel_cast else 2
    else:
        # Considered in opt_resident_mem: fp32 moment1, fp32 moment2
        # Considered in opt_transient_mem: 0
        # Here weight_mem is in fp32, so multiply by (1+1).
        opt_resident_coef = 2
        opt_transient_coef = 0

    autodist_cfg = AutoDistConfig(
        mesh_row=mesh_row,
        mesh_col=mesh_col,
        update_freq=update_freq,
        task_name=task_name,
        is_train=not cfg.inference_only,
        ignore_small_tensor_threshold=524288,  # 0.5 MB is a good threshold to reduce search time and make the result correct, will refine later
        memory_granularity=524288,             # 0.5 MB is a good threshold to reduce search time and make the result correct, will refine later
        consider_mem=True,
        partition_constraints_path=partition_constraints_path,
        memory_constraint=memory_constraint,
        opt_resident_coef=opt_resident_coef,
        opt_transient_coef=opt_transient_coef,
        verbose=verbose,
        re_profile=re_profile,
        profile_dir=profile_dir,
        world_size=cfg.runtime_ngpus,
        recompute_modules=recompute_modules,
        recompute_ratio=recompute_ratio,
        zero_stage=zero_stage,
        zero_ngroups=zero_ngroups,
        load_plan_path=load_plan_path,
        save_plan_path=save_plan_path,
        pipeline_pivots=pipeline_pivots,
        pipeline_nstages=pipeline_nstages,
        max_pipeline_bubble_ratio=max_pipeline_bubble_ratio,
        max_pipeline_unbalance_ratio=max_pipeline_unbalance_ratio,
        parallel_profile=parallel_profile,
        transient_mem_coef=transient_mem_coef,
        disable_shared_param_constraint=disable_shared_param_constraint,
        solver=solver,
    )

    return parallelize_graph(graph, autodist_cfg)


@dataclass(unsafe_hash=True, frozen=True)
class OpPartition:
    """
    OpPartition represents a partition plan for an operator dimension.
    """
    input: int
    dim: int


@dataclass
class OpPlan:
    """
    OpPlan represents the distributed plan for an operator.
    """
    op: IRFwOperation
    recompute_id: int = -1  # -1 means no recompute
    stage_id: int = -1       # pipeline stage id, -1 means following the previous op's stage

    # user defined meta data for hooks
    # which will be passed to the pre_hook and post_hook functions
    # Note: Only types that can be safely `repr`-ed can be used here. (e.g., str, int, float, tuple, list, dict)
    hook_meta: Any = None

    # function to be called before the op is executed
    # which will be inserted in the runtime code before the op call.
    # op's inputs will be passed to the hook.
    # The signature will be like
    # def pre_hook(module: ParallelModule, meta: Any, inputs: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
    pre_hook: Optional[Callable[['ParallelModule', Any, tuple[Any, ...], dict[str, Any]], None]] = None

    # function to be called after the op is executed
    # which will be inserted in the runtime code after the op call.
    # op's inputs and outputs will be passed to the hook.
    # the signature will be like
    # def post_hook(module: ParallelModule, meta: Any, inputs: Tuple[Any, ...], kwargs: Dict[str, Any], output: Any) -> None:
    post_hook: Optional[Callable[['ParallelModule', Any, tuple[Any, ...], dict[str, Any], Any], None]] = None

    # OpPartition: user specified partition plan
    #   You only need to specify one partition plan here.
    #   For example, torch.matmul has annotation of `m k+, k+ n -> m n`,
    #   If you want to partition the matmul on the k dimension,
    #   you can set OpPartition(input=0, dim=1) or OpPartition(input=1, dim=0).
    #   They are equivalent.
    # None: replicated
    # 'auto': auto partition based on the input tensor partition info
    #   1. if any of the input tensors is value partitioned, we replicate the op
    #      TODO: is it too strict?
    #   2. if any of the input tensors is partitioned on a dim,
    #      we will try to partition the op on the same dim first,
    #      if the partition is invalid, we replicate the op
    #   3. if all the input tensor is replicated, we replicate the op
    partition: OpPartition | None | Literal['auto'] = None  # partition plan
    # for future extension
    # don't use it now.
    partitions: List[OpPartition | None] = field(default_factory=list)  # multiple partition plans

    def __post_init__(self):
        if self.partition is not None and len(self.partitions) > 0:
            raise ValueError("Only one of partition and partitions can be set")

        if len(self.partitions) > 1:
            raise NotImplementedError("Multiple partitions are not supported yet")

        if len(self.partitions) == 1:
            self.partition = self.partitions[0]
            self.partitions = []


def get_layer_index(fqn: str) -> int:
    """
    Extract the layer index from full qualified name.
    If there are multiple integers in the name, raise ValueError.
    """
    nums = [int(s) for s in fqn.split(".") if s.isdigit()]
    if len(nums) != 1:
        raise ValueError(f"Name {fqn} should only contain one integer")
    return nums[0]


def get_called_self_module_name(node_call_expr: str) -> str:
    """
    Get the called module name from the node's call expr by ast.
    For example:
    self.up_proj(x) -> up_proj
    self.act_fn(self.gate_proj(x)) -> act_fn
    self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)) -> down_proj
    torch.tanh(x) -> ''  # because it's not called from self
    self.up_proj(x).transpose() -> '' # because it's an attribute call

    Other cases return empty string.

    NOTE: regex is not easy to make it work

    """

    if not node_call_expr:
        return ''
    call_expr: ast.Call = ast.parse(node_call_expr, mode='eval').body  # type: ignore
    if isinstance(call_expr, ast.Call):  # self.up_proj(x)
        if isinstance(call_expr.func, ast.Attribute):  # self.up_proj
            if isinstance(call_expr.func.value, ast.Name) and call_expr.func.value.id == 'self':
                return call_expr.func.attr  # up_proj
    return ''


def get_pas_ops(graph: IRGraph) -> List[IRFwOperation]:
    """
    Get all operators in the graph that can set operator plan.
    When we write a policy, only ops returned from this function need to be considered.

    Args:
        graph: the input IRGraph

    Returns:
        List[IRFwOperation]: list of IRFwOperation nodes
    """
    return graph.select(ntype=IRFwOperation)


def fn(
        graph: IRGraph, cfg: 'ComputeConfig',
        policy: Union[
            Callable[[IRGraph, 'ComputeConfig'], IRGraph],
            Callable[[IRGraph, 'ComputeConfig'], Iterable[OpPlan]],
        ]
) -> IRGraph:
    """
    General policy function based on user-defined policy.
    The user-defined policy can either return the final IRGraph, or
    return a list of OpPlan to describe the distributed plan for each operator.

    To write a new-style policy, the most important part is to locate the operator node in the graph.
    Here are some tips:
    1. use `node.name` to get the operator name.
    2. use `node.fn` to get the operator function.
    3. use `node.module_stack` to get the module stack info.
    4. use `node.module_class_chain` to get the module class chain.
    5. use `node.call_expr` to get the call expression string. And you can user `ast.parse` to parse it.
    6. use `get_layer_index` to get the layer index in a torch.nn.ModuleList.
    7. use `get_called_self_module_name` to get the called self module name from the call expression.
    8. use `node.inputs()` the get the input tensors of the operator.
        We can further check whether the input tensor is a parameter by `tensor.is_param`,
        or get the full name of the parameter by `tensor.name`, etc.
    9. insert anchors in code with `nnscaler.anchor` to help locate the operator (intrusive way).

    A good way to locate the operator will be like:
    1. Locate the module first by module_class_chain (`target_module in node.module_class_chain`)
    2. If the module are used multiple times (e.g., in ModuleList),
       locate further by layer index (`get_layer_index`) or `node.fqn`.
    3. Once the module is located,
       we can further locate the operator by
       `node.name`,`node.call_expr`, `node.fn`, `node.inputs()` (especially the `is_param`/`name` of input)
       or other properties.

    Args:
        graph: the input IRGraph
        cfg: the compute config
        policy: the user-defined policy function. It can either return the final IRGraph,
                or return an iterable of OpPlan for each operator.

    Returns:
        the distributed IRGraph
    """
    result = policy(graph, cfg)
    if isinstance(result, IRGraph):  # traditional policy
        return result

    op_plans = {r.op: r for r in result}
    ngpus: int = cfg.plan_ngpus

    recompute_groups: dict[int, list[IRFwOperation]] = {}
    recompute_last_id: int = -1
    recompute_group_stages: dict[int, int] = {}

    pp_stages: list[list[IRFwOperation]] = [[]]
    pp_cur_stage_id = 0

    # key: IRFullTensor
    # value:
    #   key: stage_id
    #   value: set of OpPartition in this stage
    tensor_splits: dict[IRFullTensor, dict[int, set[OpPartition]]] = {}
    # store the last split info for each tensor to help handle auto partition
    # None: replicated
    # 'value': value partitioned
    # int: the partitioned dim
    output_tensor_last_split: dict[IRFullTensor, int | None | Literal['value']] = {}

    fw_nodes = dict.fromkeys(graph.select(ntype=IRFwOperation))

    for node in fw_nodes:
        if node not in op_plans:
            op_plans[node] = OpPlan(op=node)  # default: no partition, stage 0, no recompute

        node.hook_meta = op_plans[node].hook_meta
        node.pre_hook = op_plans[node].pre_hook
        node.post_hook = op_plans[node].post_hook

        op_plan = op_plans[node]

        # set pipeline stage id if not set
        if op_plan.stage_id == -1:
            op_plan.stage_id = pp_cur_stage_id

        # currently we only support partition for IRDimops
        if not isinstance(op_plan.op, IRDimops):
            if op_plan.partition == 'auto':
                op_plan.partition = None
            if op_plan.partition is not None:
                raise ValueError("Only IRDimops can be partitioned.")

        # list of partitions for the op
        # [] means no partition(replicated)
        op_partitions = [op_plan.partition] if op_plan.partition is not None else []

        if op_partitions == ['auto']:
            # auto partition based on input tensor partition info
            op_partitions = []  # reset to collect partitions
            for idx, input in enumerate(op_plan.op.inputs()):
                if not isinstance(input, IRSubTensor):
                    continue
                ftensor = input.parent
                last_partition_dim = output_tensor_last_split.get(ftensor, None)
                if last_partition_dim == 'value':
                    # value partitioned input, replicate the op
                    op_partitions = []
                    break
                elif last_partition_dim is not None:
                    op_partitions.append(OpPartition(input=idx, dim=last_partition_dim))

        # final partition plan for the op
        # key: input idx, value: partitioned dim
        op_partition_map: dict[int, int] = {}
        if op_partitions:
            # we partition the op based on the first partition plan
            # and then check the rest partitions are satisfied or not
            op_first_partition = op_partitions[0]
            partitioned_nodes = op_plan.op.algorithm('dim')\
                .instantiate(idx=op_first_partition.input, dim=op_first_partition.dim, num=ngpus)
            subnode = partitioned_nodes[0]  # first subnode carries all necessary partition info

            # collect input partition info
            # key: input idx, value: partitioned dim
            result_partitions: dict[int, int] = {}
            for idx, input in enumerate(subnode.inputs()):
                if not isinstance(input, IRSubTensor):
                    continue
                split_dims = input.splitdims()
                assert len(split_dims) <= 1, "Internal Error: multiple splitdims in one input"
                if split_dims:
                    result_partitions[idx] = split_dims[0]

            # check the rest partitions
            # Note if we only have one partition plan, the check is skipped, we can always partition it
            # In fact, if `auto` is not specified, we always have at most one partition plan
            for op_partition in op_partitions[1:]:
                if op_partition.input not in result_partitions or \
                        result_partitions[op_partition.input] != op_partition.dim:
                    _logger.warning(
                        f"Operator {op_plan.op} cannot be partitioned as specified: {op_partition}"
                        f", replicate it instead."
                    )
                    op_partitions = []
                    op_partition_map = {}
                    break
            else:
                # all partitions are satisfied
                # then we can update input/output partition info

                # make sure the first item in op_partition_map is the first partition plan
                op_partition_map[op_first_partition.input] = op_first_partition.dim
                op_partition_map.update(result_partitions)

                for output in subnode.outputs():
                    if not isinstance(output, IRSubTensor):
                        continue
                    ftensor = output.parent
                    if output.valmap != (0, 1):
                        output_tensor_last_split[ftensor] = 'value'
                    else:
                        split_dims = output.splitdims()
                        assert len(split_dims) <= 1, "Internal Error: multiple splitdims in one output"
                        if split_dims:
                            output_tensor_last_split[ftensor] = split_dims[0]

        if op_plan.partition == 'auto':
            if not op_partition_map:
                op_plan.partition = None
            else:
                # use the first partition plan,
                # which is consistent with the logic above
                first_input_idx = list(op_partition_map.keys())[0]
                op_plan.partition = OpPartition(
                    input=first_input_idx,
                    dim=op_partition_map[first_input_idx]
                )

        # update tensor_splits for input tensors
        for idx, input in enumerate(op_plan.op.inputs()):
            if not isinstance(input, IRSubTensor):
                continue
            ftensor = input.parent
            if ftensor not in tensor_splits:
                tensor_splits[ftensor] = {}
            if idx not in op_partition_map:
                tensor_splits[ftensor].setdefault(op_plan.stage_id, set()).add(None)
            else:
                tensor_splits[ftensor].setdefault(op_plan.stage_id, set()).add(
                    OpPartition(input=idx, dim=op_partition_map[idx]))

        if op_plan.recompute_id != -1:
            if op_plan.recompute_id in recompute_group_stages:
                if recompute_group_stages[op_plan.recompute_id] != op_plan.stage_id:
                    raise ValueError("All ops in a recompute group must be in the same stage")
            else:
                recompute_group_stages[op_plan.recompute_id] = op_plan.stage_id

            if op_plan.recompute_id != recompute_last_id and op_plan.recompute_id in recompute_groups:
                raise ValueError("Nodes in a recompute group must be continuous.")

            recompute_groups.setdefault(op_plan.recompute_id, []).append(op_plan.op)

        recompute_last_id = op_plan.recompute_id

        # update pipeline stages
        if op_plan.stage_id == pp_cur_stage_id:
            pp_stages[pp_cur_stage_id].append(op_plan.op)
        elif op_plan.stage_id == pp_cur_stage_id + 1:
            pp_cur_stage_id += 1
            pp_stages.append([op_plan.op])
        else:
            raise ValueError("Pipeline stage ids must be continuous integers starting from 0")

    if len(op_plans) != len(fw_nodes):
        assert len(op_plans) > len(fw_nodes)
        for op_plan in op_plans.values():
            if op_plan.op not in fw_nodes:
                raise ValueError(f"OpPlan contains operator {op_plan.op} not in the graph or not a forward operator")

    pp_segs = [graph]
    nstages = len(pp_stages)
    pp_enabled = nstages > 1
    # not all schedulers support pp_size < nstages
    pp_size = cfg.pas_config.get('pipeline_size', nstages)
    nmicros = cfg.pas_config.get('pipeline_nmicros', None)
    scheduler = cfg.pas_config.get('pipeline_scheduler', '1f1b')
    tp_size = ngpus // pp_size

    if pp_enabled:
        if not cfg.use_end2end:
            raise ValueError("Pipeline parallelism requires use_end2end to be True")
        if pp_size <= 1:
            raise ValueError("pipeline_size must be greater than 1 when pipeline is enabled")
        if not nmicros:
            raise ValueError("nmicros must be set when pipeline is enabled")
        if nstages % pp_size != 0:
            raise ValueError(f'invalid pipeline_size {pp_size} for nstages {nstages}')
        if ngpus % pp_size != 0:
            raise ValueError(f'invalid pipeline_size {pp_size} for ngpus {ngpus}')
    else:
        if pp_size != 1:
            raise ValueError("pipeline_size must be 1 when pipeline is disabled")

    # set recompute groups
    for group in recompute_groups.values():
        if len(group) <= 1:
            continue
        graph.recompute(group)

    # add multiref for shared parameters across stages
    # note that we have constrained that shared parameters cannot be partitioned in SPMDSolver, other input tensors
    # belonging to the same operator can be partitioned. For example, in some LLMs, the embedding matrix is shared
    # with the output layer. In this case, the batch dim / seq dim of the activation tensor can be partitioned.
    for ftensor, stage_info in tensor_splits.items():
        if not ftensor.is_param():
            continue
        splits = set(k.dim if k is not None else None for v in stage_info.values() for k in v)
        find_replicated = None in splits
        splits = list(splits)
        # For safety, we will add multiref when detecting shared param are all replicated for pipeline parallelism.
        # The reason is that stages may have different number of devices, it is hard to synchronize gradients directly
        # by inserting reducers although weights are all REPLICAED.
        if len(splits) > 1 or (pp_enabled and find_replicated):
            _logger.info(f'add multiref for shared param {ftensor}')
            graph.multiref(ftensor, comment='shared param')

    # set pipeline stages
    if pp_enabled:
        graph.staging([s[0] for s in pp_stages])
        pp_segs: list[IRSegment] = graph.select(ntype=IRSegment, flatten=False)

        for stage_id, stage in enumerate(pp_segs):
            for node in stage.select(ntype=IRFwOperation):
                if node in fw_nodes:
                    continue
                if node.fn == multiref: # skip multiref nodes
                    continue
                assert node.fn == identity, "Internal Error: non-identity node added in staging"
                # force identity nodes to be replicated
                # these nodes are usually added for data transfer between stages in graph.staging
                # TODO: is it possible to have TP here?
                op_plans[node] = OpPlan(op=node, stage_id=stage_id, partition=None)

    # add multiref to an activation tensor when the states of the tensor and its grad are different
    # among consumers and current segment's outputs
    for ftensor, stage_info in tensor_splits.items():
        # Parameter are already handled above
        if ftensor.is_grad() or ftensor.is_param():
            continue

        # check if this tensor is in the output of each stage
        is_seg_output: dict[int, bool] = {}
        for idx, stage in enumerate(pp_segs):
            is_seg_output[idx] = IR.contains_object(
                stage.outputs(),
                lambda x: isinstance(x, IRSubTensor) and x.parent == ftensor
            )

        for idx, splits in stage_info.items():
            stage = pp_segs[idx]
            split_list = list(splits)
            if len(split_list) > 1 or (
                is_seg_output[idx] and split_list[0] is not None # treat segment output as a consumer
            ):
                _logger.debug(f'add multiref for {ftensor} in stage {stage}')
                stage.multiref(ftensor, comment='activation')

    # stage-wise tensor parallelism
    curr_devices = list(range(ngpus))
    for op_plan in op_plans.values():
        idx = op_plan.stage_id % pp_size
        devs = curr_devices[idx * tp_size: (idx + 1)* tp_size]
        if op_plan.partition is not None:
            _tp(graph, op_plan.op, devs, idx=op_plan.partition.input, dim=op_plan.partition.dim)
        else:
            _replica(graph, op_plan.op, devs)

    # replicate dataloader
    for dl in graph.select(ntype=IRDataOperation):
        _replica(graph, dl, devs=list(range(ngpus)))

    if pp_enabled:
        cfg.apply_pipeline_scheduler(graph, nstages, nmicros, scheduler)

    return graph


def pas_fsdp(graph, cfg: 'ComputeConfig'):
    """
    A simple FSDP policy:
    1. all operators are replicated
    2. user specified modules with `cfg.pas_config.recompute_modules` are recomputed
    3. shard policy is configured in cfg.use_zero and cfg.zero_ngroups
    4. CPU offload is not supported
    """
    if cfg.plan_ngpus != 1:
        raise ValueError("FSDP policy only supports 1 plan GPU")
    if not cfg.use_zero:
        raise ValueError("FSDP policy requires use_zero to be 1/3")
    # use 'recomputes' instead of 'recompute_modules'
    # to avoid confliction with autodist config
    recompute_modules = cfg.pas_config.get('recomputes', '')
    # parse recompute_modules
    # user can also provide a list of Module classes.
    if isinstance(recompute_modules, str):
        recompute_modules = recompute_modules.strip()
        if not recompute_modules:
            recompute_modules = []
        else:
            recompute_modules = [m.strip() for m in recompute_modules.split(',')]

    if recompute_modules:
        recompute_modules = [load_type(rm) for rm in recompute_modules]
    else:
        recompute_modules = []

    cur_recompute_id = -1
    cur_recompute_module_fqn = None
    for node in get_pas_ops(graph):
        recompute_module: torch.nn.Module
        for rm in recompute_modules:
            if rm in node.module_class_chain:
                recompute_module = rm
                break
        else:
            cur_recompute_module_fqn = None
            continue

        mod_fqn = node.get_module_fqn(recompute_module)
        if cur_recompute_module_fqn is None or cur_recompute_module_fqn != mod_fqn:
            cur_recompute_id += 1
            cur_recompute_module_fqn = mod_fqn
        yield OpPlan(node, recompute_id=cur_recompute_id)
