#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""
Dump and restore the execution graph before code generation.

The dumped graph includes forward, backward, communication (adapter),
and weight reduction nodes with their dependencies, enabling reasoning
about the computation and communication structure of the distributed plan.

Two dump formats are supported:

1. **JSON** (human-readable analysis): Contains node metadata, dependencies,
   tensor info, communication primitives, and per-device execution order.
   Suitable for analysis and visualization but not for code generation.

2. **Dill/pickle** (full state): Preserves the complete ``ExecutionPlan``
   object (including all IR nodes, tensors, and ID generator state) so that
   code generation can be re-run from the dump without recompilation.
"""

from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import json
import logging
import os

import dill

from nnscaler.ir.cten import IRCell, IRObject
from nnscaler.ir.tensor import IRSubTensor, IRFullTensor
from nnscaler.ir.unique import IDGenerator
from nnscaler.ir.operator import IRFwOperation, IRBpOperation, IRDataOperation
from nnscaler.ir.adapter import IRAdapter, IRWeightReducer
from nnscaler.ir.adapter.prim import (
    IRAdapterPrim, CommPrim, SpatialPrim, ValuePrim,
)
from nnscaler.graph.graph import IRGraph, IRSegment

from nnscaler.execplan.execplan import ExecutionPlan, ExeReuseCell


_logger = logging.getLogger(__name__)


def _classify_node(node: IRCell) -> str:
    """Classify a node into a high-level category."""
    if isinstance(node, ExeReuseCell):
        return _classify_node(node.cell)
    if isinstance(node, IRDataOperation):
        return 'data'
    if isinstance(node, IRWeightReducer):
        return 'weight_reducer'
    if isinstance(node, IRAdapter):
        return 'communication'
    if isinstance(node, IRSegment):
        return 'forward_segment' if node.isfw() else 'backward_segment'
    if isinstance(node, IRBpOperation):
        return 'backward'
    if isinstance(node, IRFwOperation):
        return 'forward'
    return 'unknown'


def _get_comm_types(node: IRCell) -> List[str]:
    """Get the communication primitive types for an adapter node."""
    if isinstance(node, ExeReuseCell):
        return _get_comm_types(node.cell)
    if not isinstance(node, IRAdapter):
        return []
    return [type(prim).__name__ for prim in node.prims]


def _tensor_info(t: IRSubTensor) -> Dict[str, Any]:
    """Extract serializable info from a subtensor."""
    info: Dict[str, Any] = {
        'tid': t.parent.tid,
        'name': t.parent.name,
        'shape': list(t.shape),
        'full_shape': list(t.parent.shape),
        'dtype': str(t.dtype),
        'requires_grad': t.requires_grad,
        'devices': list(t.device) if t.device else [],
    }
    if t.is_param():
        info['is_param'] = True
    if hasattr(t.parent, 'is_buffer') and t.parent.is_buffer():
        info['is_buffer'] = True
    if t.is_grad():
        info['is_grad'] = True
    if t.parent.is_loss():
        info['is_loss'] = True
    # Include partition info (indmap/valmap) for completeness
    if t.indmap is not None:
        info['indmap'] = [list(dim_range) for dim_range in t.indmap]
    if t.valmap is not None:
        info['valmap'] = list(t.valmap)
    return info


def _node_id(node: IRCell) -> str:
    """Generate a unique string ID for a node."""
    if isinstance(node, ExeReuseCell):
        return f'reuse_{node.cell.cid}_{id(node)}'
    return f'{node.cid}'


def _unwrap(node: IRCell) -> IRCell:
    """Unwrap ExeReuseCell to get the underlying cell."""
    if isinstance(node, ExeReuseCell):
        return node.cell
    return node


def _serialize_node(node: IRCell) -> Dict[str, Any]:
    """Serialize a single execution node."""
    unwrapped = _unwrap(node)
    category = _classify_node(node)

    info: Dict[str, Any] = {
        'id': _node_id(node),
        'cid': unwrapped.cid,
        'name': unwrapped.name,
        'category': category,
        'signature': unwrapped.signature or '',
        'devices': list(node.device) if node.device else [],
        'is_forward': node.isfw(),
    }

    if isinstance(node, ExeReuseCell):
        info['is_reuse'] = True
        info['reuse_cell_cid'] = node.cell.cid

    # Communication details
    if category == 'communication':
        comm_types = _get_comm_types(node)
        info['comm_primitives'] = comm_types
        adapter = unwrapped
        if isinstance(adapter, IRAdapter):
            info['differentiable'] = adapter.differentiable
            if adapter.recompute is not None:
                info['recompute_group'] = adapter.recompute
            info['comm_details'] = []
            total_volume = 0
            for prim in adapter.prims:
                prim_info = {
                    'type': type(prim).__name__,
                    'local': prim.local,
                    'devices': list(prim.device),
                }
                if isinstance(prim, CommPrim):
                    prim_info['kwargs'] = {
                        k: v if not isinstance(v, (list, tuple)) else list(v)
                        for k, v in prim.kwargs.items()
                    }
                    try:
                        vol = prim.volume()
                        prim_info['volume'] = vol
                        total_volume += vol
                    except (NotImplementedError, Exception):
                        pass
                info['comm_details'].append(prim_info)
            if total_volume > 0:
                info['total_comm_volume'] = total_volume

    # Recompute info for segments/ops
    if isinstance(unwrapped, IRFwOperation) and unwrapped.recompute is not None:
        info['recompute_group'] = unwrapped.recompute

    # Mirror (forward/backward pair) info
    if unwrapped.mirror is not None and isinstance(unwrapped.mirror, IRCell):
        info['mirror_cid'] = unwrapped.mirror.cid

    # Input/output tensor info
    inputs = []
    for t in node.inputs():
        if isinstance(t, IRSubTensor):
            inputs.append(_tensor_info(t))
        elif isinstance(t, IRObject):
            inputs.append({'name': t.name, 'type': 'object'})
    info['inputs'] = inputs

    outputs = []
    for t in node.outputs():
        if isinstance(t, IRSubTensor):
            outputs.append(_tensor_info(t))
        elif isinstance(t, IRObject):
            outputs.append({'name': t.name, 'type': 'object'})
    info['outputs'] = outputs

    # Weight reducer details
    if isinstance(unwrapped, IRWeightReducer):
        info['reducer_weights'] = []
        for t in unwrapped.inputs():
            if isinstance(t, IRSubTensor):
                info['reducer_weights'].append({
                    'tid': t.parent.tid,
                    'name': t.parent.name,
                    'shape': list(t.shape),
                    'dtype': str(t.dtype),
                })

    # Sub-nodes for segments
    if isinstance(unwrapped, IRSegment):
        sub_nodes = []
        for sub in unwrapped.nodes(flatten=True):
            sub_info = {
                'cid': sub.cid,
                'name': sub.name,
                'signature': sub.signature or '',
                'category': _classify_node(sub),
            }
            if isinstance(sub, IRFwOperation) and sub.recompute is not None:
                sub_info['recompute_group'] = sub.recompute
            # Include tensor IDs for graph edge computation
            in_tids = []
            for t in sub.inputs():
                if isinstance(t, IRSubTensor):
                    in_tids.append(t.parent.tid)
            out_tids = []
            for t in sub.outputs():
                if isinstance(t, IRSubTensor):
                    out_tids.append(t.parent.tid)
            sub_info['input_tids'] = in_tids
            sub_info['output_tids'] = out_tids
            sub_nodes.append(sub_info)
        info['sub_nodes'] = sub_nodes

        # Compute edges between sub-nodes within this segment
        info['sub_edges'] = _compute_sub_edges(sub_nodes)

    return info


def _compute_sub_edges(sub_nodes: List[Dict[str, Any]]) -> List[List[int]]:
    """Compute edges between sub-nodes from tensor IDs.

    Returns a list of [from_index, to_index] pairs (indices into sub_nodes).
    Uses compact format to keep JSON small for large graphs.
    """
    # Map: tid -> list of producer sub-node indices
    tid_producers: Dict[int, List[int]] = {}
    edges: List[List[int]] = []

    for idx, sn in enumerate(sub_nodes):
        # Check inputs against known producers
        for tid in sn.get('input_tids', []):
            if tid in tid_producers:
                for prod_idx in tid_producers[tid]:
                    edges.append([prod_idx, idx])
        # Register outputs
        for tid in sn.get('output_tids', []):
            tid_producers.setdefault(tid, []).append(idx)

    return edges


def _compute_dependencies(device_nodes: List[IRCell]) -> List[Dict[str, Any]]:
    """Compute data dependencies between nodes based on tensor producer/consumer relationships.

    Returns a list of edges: {from_id, to_id, tensor_tid, tensor_name}
    """
    # Map: (parent_tid) -> list of producer node IDs
    tensor_producers: Dict[int, List[str]] = {}
    edges: List[Dict[str, Any]] = []

    for node in device_nodes:
        node_nid = _node_id(node)
        # Check inputs: find producers of each input tensor
        for t in node.inputs():
            if isinstance(t, IRSubTensor):
                ptid = t.parent.tid
                if ptid in tensor_producers:
                    for producer_nid in tensor_producers[ptid]:
                        edges.append({
                            'from': producer_nid,
                            'to': node_nid,
                            'tensor_tid': ptid,
                            'tensor_name': t.parent.name,
                        })

        # Register outputs as produced by this node
        for t in node.outputs():
            if isinstance(t, IRSubTensor):
                ptid = t.parent.tid
                tensor_producers.setdefault(ptid, []).append(node_nid)

    return edges


def dump_execution_graph(
    execplan: ExecutionPlan,
    outfile: Optional[str] = None,
) -> Dict[str, Any]:
    """Dump the complete execution graph from an ExecutionPlan as JSON.

    The dumped graph includes:
    - All execution nodes (forward, backward, communication, weight reduction, data)
    - Data dependencies between nodes (tensor-based edges)
    - Per-device execution sequences
    - Node metadata (category, devices, shapes, communication primitives)
    - Tensor partition info (indmap, valmap) and communication volume

    This is a human-readable analysis dump. For re-generating code from a dump,
    use :func:`save_execution_plan` / :func:`load_execution_plan` instead.

    Args:
        execplan: The finalized execution plan.
        outfile: Optional file path to write the JSON dump.
            If None, only returns the dict.

    Returns:
        A dict representing the full execution graph, serializable to JSON.
    """
    graph = execplan.graph

    result: Dict[str, Any] = {
        'metadata': {
            'train': graph.train,
            'num_devices': len(execplan.devices()),
            'devices': execplan.devices(),
            'has_scheduler': graph.sched is not None,
        },
        'graph_inputs': [],
        'graph_outputs': [],
        'graph_attributes': [],
        'per_device': {},
    }

    # Graph-level inputs
    for t in graph.inputs():
        if isinstance(t, IRSubTensor):
            result['graph_inputs'].append(_tensor_info(t))
        elif isinstance(t, IRObject):
            result['graph_inputs'].append({'name': t.name, 'type': 'object'})

    # Graph-level outputs
    for t in IRSegment.get_objects_from_complex(execplan.outputs()):
        if isinstance(t, IRSubTensor):
            result['graph_outputs'].append(_tensor_info(t))
        elif isinstance(t, IRObject):
            result['graph_outputs'].append({'name': t.name, 'type': 'object'})

    # Graph attributes (parameters and buffers)
    for ft in graph.attributes():
        attr_info: Dict[str, Any] = {
            'tid': ft.tid,
            'name': ft.name,
            'shape': list(ft.shape),
            'dtype': str(ft.dtype),
            'requires_grad': ft.requires_grad,
        }
        if ft.is_param():
            attr_info['is_param'] = True
        if hasattr(ft, 'is_buffer') and ft.is_buffer():
            attr_info['is_buffer'] = True
        result['graph_attributes'].append(attr_info)

    # Per-device execution details
    for devid in execplan.devices():
        device_nodes = execplan.seq(devid)
        serialized_nodes = [_serialize_node(n) for n in device_nodes]
        edges = _compute_dependencies(device_nodes)

        # Summary statistics
        categories = [_classify_node(n) for n in device_nodes]
        summary = {
            'total_nodes': len(device_nodes),
            'forward_nodes': sum(1 for c in categories if c in ('forward', 'forward_segment')),
            'backward_nodes': sum(1 for c in categories if c in ('backward', 'backward_segment')),
            'communication_nodes': sum(1 for c in categories if c == 'communication'),
            'weight_reducer_nodes': sum(1 for c in categories if c == 'weight_reducer'),
            'data_nodes': sum(1 for c in categories if c == 'data'),
        }

        result['per_device'][devid] = {
            'summary': summary,
            'execution_order': [_node_id(n) for n in device_nodes],
            'nodes': {_node_id(n): s for n, s in zip(device_nodes, serialized_nodes)},
            'edges': edges,
        }

    if outfile is not None:
        with open(outfile, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        _logger.info(f'Execution graph dumped to {outfile}')

    return result


# ===================== Dill-based dump/load =====================


def save_execution_plan(
    execplan: ExecutionPlan,
    outfile: str,
    save_json: bool = True,
) -> None:
    """Save the full ExecutionPlan state to disk via dill.

    This preserves all IR state (nodes, tensors, adapters, partition info,
    ID generator state) so that code generation can be re-run later from
    the saved file without recompilation.

    Optionally also writes a companion ``.json`` analysis file alongside
    the dill file.

    Args:
        execplan: The finalized execution plan.
        outfile: File path for the dill dump (e.g. ``execplan.pkl``).
        save_json: If True, also write a ``<outfile>.json`` companion
            with the human-readable analysis dump.
    """
    # Use the same pickling context as IRGraph.dumps() to decouple cell refs
    class _PicklingContext:
        def __enter__(self):
            IRObject.__getstate__ = IRObject.getstate_for_dump
        def __exit__(self, *exc):
            IRObject.__getstate__ = lambda self: self.__dict__.copy()

    with _PicklingContext():
        state = {
            'id_generator': IDGenerator().get_states(),
            'execplan': execplan,
        }
        with open(outfile, 'wb') as f:
            dill.dump(state, f)
    _logger.info(f'Execution plan saved to {outfile}')

    if save_json:
        json_file = outfile + '.json'
        dump_execution_graph(execplan, outfile=json_file)


def load_execution_plan(infile: str) -> ExecutionPlan:
    """Load an ExecutionPlan from a dill dump created by :func:`save_execution_plan`.

    This restores the full IR state including the ID generator, so subsequent
    code generation will produce correct and consistent output.

    Args:
        infile: Path to the dill file.

    Returns:
        The restored ExecutionPlan.
    """
    with open(infile, 'rb') as f:
        state = dill.load(f)

    IDGenerator().load_states(state['id_generator'])
    execplan: ExecutionPlan = state['execplan']

    # Restore cell references on all IRObjects (same as IRGraph.from_dill)
    _reset_cell_refs(execplan.graph)

    # Also restore cell refs for dispatched nodes in device sequences
    for devid in execplan.devices():
        for node in execplan.seq(devid):
            unwrapped = node.cell if isinstance(node, ExeReuseCell) else node
            for t in node.inputs() + node.outputs():
                if isinstance(t, IRObject):
                    t.cell = node
            if isinstance(unwrapped, IRSegment):
                _reset_cell_refs(unwrapped)

    _logger.info(f'Execution plan loaded from {infile}')
    return execplan


def _reset_cell_refs(segment: IRSegment) -> None:
    """Restore bidirectional cell<->tensor links after dill deserialization."""
    for t in segment.inputs():
        if isinstance(t, IRObject):
            t.cell = segment
    for node in segment.nodes():
        for t in node.inputs() + node.outputs():
            if isinstance(t, IRObject):
                t.cell = node
        if isinstance(node, IRSegment):
            _reset_cell_refs(node)
    for t in IRSegment.get_objects_from_complex(segment.outputs()):
        if isinstance(t, IRObject):
            t.cell = segment


# ===================== Code generation from dump =====================


def gencode_from_execution_plan(
    execplan: ExecutionPlan,
    num_ranks: int,
    outdir: str = '.',
    filename_pattern: str = 'gencode{rank}.py',
    runtime_ndevs: Optional[int] = None,
) -> List[str]:
    """Generate distributed training code files from an ExecutionPlan.

    This runs the same ``ModuleCodeGen`` + ``ScheduleCodeGen`` pipeline
    that ``@nnscaler.compile`` uses, but from a pre-built (or restored)
    ExecutionPlan rather than requiring a live compilation session.

    If the execution plan contains ungrouped forward operations (i.e.,
    bare ``IRFwOperation`` at the top level instead of ``IRSegment``),
    the grouping pass will be applied automatically before code generation.

    Args:
        execplan: The finalized execution plan.
        num_ranks: Number of rank files to generate (typically
            ``local_world_size`` during compilation).
        outdir: Directory to write the generated files into.
        filename_pattern: Pattern for output filenames, must contain
            ``{rank}`` placeholder. Default ``'gencode{rank}.py'``.
        runtime_ndevs: If set and greater than the number of devices in
            the plan, enables data-parallel scaling to this many devices.

    Returns:
        List of generated file paths (one per rank).
    """
    from nnscaler.codegen import ModuleCodeGen, ScheduleCodeGen
    from nnscaler.execplan.planpass.grouping import Grouping

    # Apply grouping if needed (codegen requires ops wrapped in IRSegments)
    needs_grouping = any(
        isinstance(node, IRFwOperation) and not isinstance(node, IRDataOperation)
        for devid in execplan.devices()
        for node in execplan.seq(devid)
    )
    if needs_grouping:
        execplan = Grouping.apply(execplan)

    plan_ndevs = len(execplan.devices())
    scale_ndevs = runtime_ndevs if runtime_ndevs and runtime_ndevs > plan_ndevs else None

    mgener = ModuleCodeGen(execplan, runtime_ndevs=scale_ndevs)
    sgener = ScheduleCodeGen(execplan, runtime_ndevs=scale_ndevs)

    os.makedirs(outdir, exist_ok=True)
    generated: List[str] = []

    for rank in range(num_ranks):
        fname = os.path.join(outdir, filename_pattern.format(rank=rank))
        mgener.gen(rank, outfile=fname, attach=False)
        sgener.gen(device=rank, outfile=fname, attach=True)
        generated.append(fname)
        _logger.info(f'Generated code for rank {rank}: {fname}')

    return generated


def gencode_from_file(
    infile: str,
    num_ranks: int,
    outdir: str = '.',
    filename_pattern: str = 'gencode{rank}.py',
    runtime_ndevs: Optional[int] = None,
) -> List[str]:
    """Load a saved ExecutionPlan and generate code files.

    Convenience wrapper that combines :func:`load_execution_plan` and
    :func:`gencode_from_execution_plan`.

    Args:
        infile: Path to the dill file created by :func:`save_execution_plan`.
        num_ranks: Number of rank files to generate.
        outdir: Directory for output files.
        filename_pattern: Output filename pattern with ``{rank}`` placeholder.
        runtime_ndevs: Optional data-parallel scaling target.

    Returns:
        List of generated file paths.
    """
    execplan = load_execution_plan(infile)
    return gencode_from_execution_plan(
        execplan,
        num_ranks=num_ranks,
        outdir=outdir,
        filename_pattern=filename_pattern,
        runtime_ndevs=runtime_ndevs,
    )
