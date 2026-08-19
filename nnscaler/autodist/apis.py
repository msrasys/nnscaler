#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import copy
import json
import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, TYPE_CHECKING

from nnscaler.graph import IRGraph
from nnscaler.graph.segment import IRSegment
from nnscaler.ir.operator import IRFwOperation

from .autodist_config import AutoDistConfig
from .descs import *
from .model_graph import ModelGraph, estimate_mem_lower_bound
from .pipeline_solver import calc_optimal_pp_plan
from .spmd_solver import analysis_pretty_printer, calc_optimal_spmd_plan

if TYPE_CHECKING:
    from nnscaler.policies import OpPlan

_logger = logging.getLogger(__name__)

__all__ = [
    'parallelize_graph',
]


def check_env(autodist_config: AutoDistConfig):
    arch_dir = Path(autodist_config.profile_dir)
    if not arch_dir.exists():
        _logger.info(f'create folder: {arch_dir}')
        arch_dir.mkdir(parents=True, exist_ok=True)


def pre_estimate_mem(graph: ModelGraph):
    '''
    Estimate a rough lower bound of memory consumption per device. Exit if the model is too large
    for allocated resources.
    '''

    def to_mb(size):
        return size // 1024 // 1024

    def to_gb(size):
        return to_mb(size) // 1024

    # calculate sizes of activations, buffers and parameters, exit if the model is
    # too large for allocated resources
    param_mem, buffer_mem, activation_mem = graph.query_mem(0, graph.op_num - 1)
    _logger.info(
        f'param mem {to_mb(param_mem)} MB, buff mem {to_mb(buffer_mem)} MB, activation mem {to_mb(activation_mem)} MB'
    )
    plan_ngpus = graph.autodist_config.mesh_desc.ngpus
    if graph.autodist_config.zero_stage == 1:
        zero_group_size = graph.autodist_config.world_size // graph.autodist_config.zero_ngroups
    elif graph.autodist_config.zero_stage == 0:
        zero_group_size = plan_ngpus
    else:
        raise RuntimeError(
            f'invalid zero stage {graph.autodist_config.zero_stage}')
    min_single_dev_mem = estimate_mem_lower_bound(
        param_mem=param_mem,
        buffer_mem=buffer_mem,
        activation_mem=activation_mem,
        plan_ngpus=plan_ngpus,
        zero_group_size=zero_group_size,
        cfg=graph.autodist_config,
    )
    min_single_dev_mem += graph.min_recompute_mem
    _logger.info(
        f'estimated minimum memory per device {to_mb(min_single_dev_mem)} MB')
    mem_constraint = graph.autodist_config.memory_constraint
    if min_single_dev_mem > mem_constraint * 1024 * 1024 * 1024:
        raise RuntimeError(
            f'est min mem: {to_gb(min_single_dev_mem)} GB vs mem constraint: {mem_constraint} GB, '
            + 'model is too large for current resources, try to ' +
            'reduce batch size, add more devices or increase zero group size')


def calc_parallel_plan(graph: IRGraph,
                       autodist_config: AutoDistConfig) -> PipelineSearchOutput:
    _logger.info(autodist_config)
    check_env(autodist_config)

    autodist_graph = ModelGraph(ir_graph=graph, autodist_config=autodist_config)
    pre_estimate_mem(autodist_graph)

    recompute_groups = autodist_graph.recompute_groups
    recompute_groups = [
        [node.cid for node in group] for group in recompute_groups
    ]

    if autodist_config.pipeline_enabled:
        pp_out = calc_optimal_pp_plan(autodist_graph, autodist_config, uniform_tp=True)
    else:
        pp_out = calc_optimal_spmd_plan(autodist_graph, autodist_config)
    pp_out.desc.recompute_groups = recompute_groups
    pp_out.stage_mems = [mem for mem in pp_out.stage_mems]
    return pp_out


def _write_plan_json(plan_json, f):
    """Write plan JSON with compact one-line partition_descs entries.

    Each entry in partition_descs is rendered as a single line like:
        {"cid": 11, "partition": [[[0, 0], 2]], "fqn": "...", "op": "..."}
    while the rest of the JSON uses standard indent=2 pretty-printing.
    """
    plan_json = copy.deepcopy(plan_json)
    markers = {}
    for spmd in plan_json.get('desc', {}).get('spmd_descs', []):
        for i, entry in enumerate(spmd.get('partition_descs', [])):
            if isinstance(entry, dict) and 'cid' in entry:
                marker = f'__COMPACT_{entry["cid"]}__'
                markers[f'"{marker}"'] = json.dumps(entry, separators=(', ', ': '))
                spmd['partition_descs'][i] = marker
    text = json.dumps(plan_json, indent=2)
    if markers:
        pattern = re.compile('|'.join(re.escape(m) for m in markers))
        text = pattern.sub(lambda match: markers[match.group(0)], text)
    f.write(text)


def parallelize_graph(
    graph: IRGraph,
    autodist_config: AutoDistConfig,
) -> Iterable['OpPlan']:
    """Convert an AutoDist search result into plans consumed by ``policies.fn``.

    This function only creates ``OpPlan`` objects; graph staging, partitioning,
    device assignment, and scheduling are handled by ``policies.fn``.

    The current ``OpPlan`` path supports only plans where every pipeline stage
    uses the same number of devices. Each operator must be either replicated or
    partitioned along one input dimension using all devices in its stage.

    Args:
        graph: The unsegmented IR graph to plan.
        autodist_config: Configuration used to search for or load an AutoDist plan.

    Returns:
        Operator plans describing stage placement, recomputation, and partitioning.
    """
    segments: List[IRSegment] = graph.select(ntype=IRSegment)
    if segments:
        raise RuntimeError('assume there is no segment in the graph')

    if autodist_config.load_plan_path:
        _logger.info(f'load plan from {autodist_config.load_plan_path}')
        with open(autodist_config.load_plan_path, 'r') as f:
            search_out_json = json.load(f)
        search_out = PipelineSearchOutput.from_json(search_out_json)
    else:
        search_out = calc_parallel_plan(graph, autodist_config)

        if autodist_config.save_plan_path:
            _logger.info(f'save plan to {autodist_config.save_plan_path}')
            # build cid-to-node mapping for annotating plan with fqn/op
            cid2node_for_save: Dict[int, IRFwOperation] = {}
            for node in graph.nodes():
                if isinstance(node, IRFwOperation):
                    cid2node_for_save[node.cid] = node
            plan_json = search_out.to_json(cid2node=cid2node_for_save)
            with open(autodist_config.save_plan_path, 'w') as f:
                _write_plan_json(plan_json, f)

    _logger.info(f'use plan with e2e time/s {1000 * search_out.e2e_time:.2f}ms')
    pp_desc = search_out.desc

    cid2node: Dict[int, IRFwOperation] = dict()
    for node in graph.nodes():
        if isinstance(node, IRFwOperation):
            cid2node[node.cid] = node

    nstages = len(pp_desc.spmd_descs)
    if autodist_config.pipeline_nstages != 'auto' and nstages != autodist_config.pipeline_nstages:
        raise RuntimeError("pipeline_nstages doesn't match the number of stages (based on your pipeline_pivots config) in the plan")

    if pp_desc.mesh_desc.ngpus != autodist_config.mesh_desc.ngpus:
        raise RuntimeError(
            f'plan uses {pp_desc.mesh_desc.ngpus} devices, but autodist config has '
            f'{autodist_config.mesh_desc.ngpus}'
        )

    # key: node cid
    # value: stage id
    planned_stages: dict[int, int] = {}
    if pp_desc.mesh_desc.ngpus % nstages != 0:
        raise RuntimeError(
            f'autodist plan uses {pp_desc.mesh_desc.ngpus} devices across {nstages} stages, '
            'but fn requires the same number of devices per stage'
        )
    tp_size = pp_desc.mesh_desc.ngpus // nstages

    for stage_id, spmd_desc in enumerate(pp_desc.spmd_descs):
        if not spmd_desc.partition_descs:
            raise RuntimeError(f'autodist plan stage {stage_id} is empty')
        if spmd_desc.mesh_desc.ngpus != tp_size:
            raise RuntimeError(
                f'autodist plan stage {stage_id} uses {spmd_desc.mesh_desc.ngpus} devices, '
                f'but fn automatically assigns {tp_size} devices per stage'
            )

        for cid in spmd_desc.partition_descs:
            if cid not in cid2node:
                raise RuntimeError(f'node {cid} not found in {cid2node}, make sure the plan is correct')
            if cid in planned_stages:
                raise RuntimeError(f'node {cid} appears in multiple stages in the autodist plan')
            planned_stages[cid] = stage_id

        stage_info_str = f'stage {stage_id} with {tp_size} devices and mem {search_out.stage_mems[stage_id]:.2f} GB'
        _logger.info(f'\nautodist plan analysis for {stage_info_str}:\n\n{analysis_pretty_printer(spmd_desc.analysis)}')

    recompute_ids: dict[int, int] = {}
    for recompute_id, group in enumerate(pp_desc.recompute_groups):
        for cid in group:
            if cid not in cid2node:
                raise RuntimeError(f'recompute node {cid} not found in {cid2node}, make sure the plan is correct')
            if cid in recompute_ids:
                raise RuntimeError(f'node {cid} appears in multiple recompute groups')
            recompute_ids[cid] = recompute_id

    from nnscaler.policies import OpPartition, OpPlan

    op_plans = []
    for node in graph.select(ntype=IRFwOperation):
        planned_stage_id = planned_stages.get(node.cid)
        partition = None
        if planned_stage_id is not None:
            partition_desc = pp_desc.spmd_descs[planned_stage_id].partition_descs[node.cid]
            if len(partition_desc.desc) != 1:
                raise RuntimeError(f'node {node} is partitioned along multiple dims')

            (input_idx, dim), partition_num = partition_desc.desc[0]
            if partition_num != tp_size:
                raise RuntimeError(
                    f'node {node.cid} uses partition degree {partition_num}, but stage '
                    f'{planned_stage_id} has {tp_size} devices'
                )
            if (input_idx, dim) == (-1, -1):
                partition = None
            elif input_idx >= 0 and dim >= 0:
                partition = OpPartition(input=input_idx, dim=dim)
            else:
                raise RuntimeError(f'invalid partition description {partition_desc.desc} for node {node.cid}')

        op_plans.append(OpPlan(
            op=node,
            recompute_id=recompute_ids.get(node.cid, -1),
            stage_id=planned_stage_id or -1,
            partition=partition,
        ))

    return op_plans
