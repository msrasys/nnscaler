#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""
Operator rescheduling.

The operators (compute ops and communication ops) inside a forward segment are
emitted by the code generator in the exact order they appear in ``IRSegment._nodes``.
That order is *one* valid execution order, but it is not the only one: as long as the
temporal causality (the data dependencies) between operators is respected, the
operators can be reordered freely.

This module collects the data dependencies among the operators of a segment and
builds a directed acyclic dependency graph (:class:`OpDependencyGraph`) such that
**any topological order of the graph is a legal execution order**.  A topological
order is then written back into the segment so that the downstream code generation
emits the operators in the rescheduled order.

The graph encodes three kinds of edges:

- data dependencies (read-after-write / write-after-read / write-after-write),
  derived from the produced / consumed (sub)tensors of every operator;
- communication-serialization edges, that keep the *relative* order of the
  communication operators unchanged.  This is required for correctness in SPMD
  execution: collective communication primitives must be issued in a consistent
  order across all ranks, otherwise different ranks may deadlock.  Preserving the
  relative order of the communication operators on every rank keeps the global
  collective order consistent;
- recompute-group boundaries are respected by building one dependency graph per
  maximal contiguous run of operators that share the same recompute group, and
  concatenating the runs in their original order.  This guarantees that operators
  never move across recompute (gradient-checkpointing) boundaries.
"""

from typing import Callable, Dict, Hashable, List, Optional, Tuple, Union
import logging
import json
import os
from pathlib import Path

import more_itertools

from nnscaler.ir.cten import IRCell, IRObject
from nnscaler.ir.tensor import IRSubTensor
from nnscaler.ir.adapter import IRAdapter, IRWeightReducer
from nnscaler.graph.segment import IRSegment

from nnscaler.execplan import ExecutionPlan
from nnscaler.execplan.execplan import ExeReuseCell
from nnscaler.execplan.planpass.planpass import PlanPass


_logger = logging.getLogger(__name__)

# bump this when the on-disk schedule config layout changes incompatibly
SCHEDULE_CONFIG_VERSION = 1


def _is_comm(node: IRCell) -> bool:
    """Whether a node is a communication operator."""
    return isinstance(node, IRAdapter)


def _group_key(obj: IRObject) -> Optional[Hashable]:
    """Get a coarse key that buckets together objects that *may* overlap.

    Two objects can only overlap if they share the same key, so the key is used to
    avoid comparing every pair of objects.

    - for an :class:`IRSubTensor`, objects overlap only when they share the same
      parent full tensor, hence the parent is used as the key;
    - for any other :class:`IRObject`, ``overlap`` reduces to tensor-id equality,
      hence the tensor id is used as the key.
    """
    if isinstance(obj, IRSubTensor):
        return obj.parent
    if isinstance(obj, IRObject):
        return obj.tid
    return None


class OpDependencyGraph:
    """A directed acyclic graph of operators ordered by their data dependencies.

    The input ``nodes`` must be given in a known-valid execution order (the original
    order of the segment).  Edges are only added in the forward direction (from an
    earlier node to a later node), so the resulting graph is acyclic by construction
    and every topological order of it is a legal execution order.

    Args:
        nodes: operators in their original (known-valid) execution order.
        serialize_segments: when ``True``, edges are added between consecutive
            *non-communication* nodes so that they keep their original relative
            order.  This is required at the cross-segment (execution sequence)
            level, where the forward -> backward dependency, the gradient flow
            between backward segments, and the backward -> reducer dependency are
            all implicit (carried by the runtime's saved-activation / autograd
            machinery) and are therefore *not* visible in the tensor data-flow.
            Only the communication adapters are then free to move within their
            data-dependency window.
        comm_types: node types that are treated as communication.  These are the
            only nodes allowed to move when ``serialize_segments`` is set, and their
            relative order is preserved (communication-serialization) to keep
            collective communication consistent across ranks.
    """

    def __init__(
        self,
        nodes: List[IRCell],
        *,
        serialize_segments: bool = False,
        comm_types: Tuple[type, ...] = (IRAdapter,),
    ):
        self._nodes: List[IRCell] = list(nodes)
        self._order: Dict[IRCell, int] = {node: idx for idx, node in enumerate(self._nodes)}
        # successors[a] contains b iff there is an edge a -> b (a must run before b)
        self._successors: Dict[IRCell, set] = {node: set() for node in self._nodes}
        self._predecessors: Dict[IRCell, set] = {node: set() for node in self._nodes}
        # (src, dst) -> set of edge kinds ('raw' | 'war' | 'waw' | 'comm' | 'order')
        self._edge_kinds: Dict[Tuple[IRCell, IRCell], set] = {}
        self._serialize_segments = serialize_segments
        self._comm_types = comm_types
        self._build()

    def _is_comm(self, node: IRCell) -> bool:
        # unwrap ExeReuseCell (used in pipeline schedules) before the type check
        cell = node.cell if isinstance(node, ExeReuseCell) else node
        return isinstance(cell, self._comm_types)

    def _add_edge(self, src: IRCell, dst: IRCell, kind: str = 'raw') -> None:
        if src is dst:
            return
        # only keep forward edges (earlier node -> later node) w.r.t. the original
        # order, which keeps the graph acyclic.
        if self._order[src] >= self._order[dst]:
            return
        self._successors[src].add(dst)
        self._predecessors[dst].add(src)
        self._edge_kinds.setdefault((src, dst), set()).add(kind)

    def _build(self) -> None:
        self._build_data_edges()
        self._build_comm_edges()
        if self._serialize_segments:
            self._build_anchor_edges()

    def _build_data_edges(self) -> None:
        """Add read-after-write, write-after-read and write-after-write edges."""
        # key -> list of (node, obj) seen so far, in original order
        writers: Dict[Hashable, List[Tuple[IRCell, IRObject]]] = {}
        readers: Dict[Hashable, List[Tuple[IRCell, IRObject]]] = {}

        for node in self._nodes:
            in_objs = node.iobjs()
            out_objs = node.oobjs()

            # read-after-write: a read depends on every earlier overlapping write
            for obj in in_objs:
                key = _group_key(obj)
                for prev_node, prev_obj in writers.get(key, ()):
                    if obj.overlap(prev_obj):
                        self._add_edge(prev_node, node, 'raw')

            # write-after-read (anti dependency) and write-after-write (output
            # dependency): a write depends on every earlier overlapping read/write
            for obj in out_objs:
                key = _group_key(obj)
                for prev_node, prev_obj in readers.get(key, ()):
                    if obj.overlap(prev_obj):
                        self._add_edge(prev_node, node, 'war')
                for prev_node, prev_obj in writers.get(key, ()):
                    if obj.overlap(prev_obj):
                        self._add_edge(prev_node, node, 'waw')

            # record this node's reads / writes for subsequent nodes
            for obj in in_objs:
                readers.setdefault(_group_key(obj), []).append((node, obj))
            for obj in out_objs:
                writers.setdefault(_group_key(obj), []).append((node, obj))

    def _build_comm_edges(self) -> None:
        """Serialize communication operators to preserve their relative order."""
        comm_nodes = [node for node in self._nodes if self._is_comm(node)]
        for prev_node, node in more_itertools.pairwise(comm_nodes):
            self._add_edge(prev_node, node, 'comm')

    def _build_anchor_edges(self) -> None:
        """Keep the relative order of every non-communication node.

        At the cross-segment level the forward -> backward dependency, the gradient
        flow between backward segments, and the backward -> reducer dependency are
        implicit and not captured by the tensor data-flow, so we conservatively
        forbid reordering any non-communication node (segments, reducers, data ops)
        relative to each other.  Only the communication adapters are free to move.
        """
        anchors = [node for node in self._nodes if not self._is_comm(node)]
        for prev_node, node in more_itertools.pairwise(anchors):
            self._add_edge(prev_node, node, 'order')

    def successors(self, node: IRCell) -> Tuple[IRCell, ...]:
        """Get the direct successors of a node (nodes that must run after it)."""
        return tuple(self._successors[node])

    def edges(self) -> List[Tuple[IRCell, IRCell, Tuple[str, ...]]]:
        """Get all edges as ``(src, dst, kinds)`` tuples in a deterministic order."""
        items = sorted(
            self._edge_kinds.items(),
            key=lambda kv: (self._order[kv[0][0]], self._order[kv[0][1]]),
        )
        return [(src, dst, tuple(sorted(kinds))) for (src, dst), kinds in items]

    def topological_sort(
        self,
        priority: Optional[Callable[[IRCell], Hashable]] = None,
    ) -> List[IRCell]:
        """Return a legal execution order (a topological order of the graph).

        Args:
            priority: an optional key function used to break ties between operators
                that are simultaneously ready to be scheduled.  Smaller keys are
                scheduled first.  Defaults to the original order of the operators,
                which keeps the rescheduled order as close to the original as the
                dependencies allow.
        """
        import heapq

        if priority is None:
            priority = lambda node: self._order[node]

        indegree: Dict[IRCell, int] = {
            node: len(self._predecessors[node]) for node in self._nodes
        }
        # the heap stores (priority_key, original_order) so that entries are always
        # comparable even when two nodes share the same priority key, and the node
        # itself never needs to be compared.
        ready: List[Tuple[Hashable, int]] = [
            (priority(node), self._order[node])
            for node in self._nodes
            if indegree[node] == 0
        ]
        heapq.heapify(ready)

        result: List[IRCell] = []
        while ready:
            _, order = heapq.heappop(ready)
            node = self._nodes[order]
            result.append(node)
            for succ in self._successors[node]:
                indegree[succ] -= 1
                if indegree[succ] == 0:
                    heapq.heappush(ready, (priority(succ), self._order[succ]))

        assert len(result) == len(self._nodes), \
            'topological sort failed: the dependency graph is not acyclic'
        return result

    def is_valid_order(self, order: List[IRCell]) -> bool:
        """Check whether ``order`` respects every dependency edge of the graph."""
        if len(order) != len(self._nodes) or set(order) != set(self._nodes):
            return False
        position: Dict[IRCell, int] = {node: idx for idx, node in enumerate(order)}
        for src in self._nodes:
            for dst in self._successors[src]:
                if position[src] >= position[dst]:
                    return False
        return True


# ---------------------------------------------------------------------------
# Config-file driven schedule
# ---------------------------------------------------------------------------
#
# The op order of every forward segment can be dumped to a human-editable config
# file (:func:`dump_schedule`) and later loaded back (:func:`load_schedule_order`)
# to drive the rescheduling.  Operators are matched between the config file and the
# execution plan by their cell id (``cid``), which is deterministic for a given
# model and parallelization configuration.  The recorded order is honoured as a
# *preference*: it is fed as the tie-breaking priority of the topological sort, so
# the resulting schedule always stays a legal (causally-correct) execution order
# even if the config requests an impossible order.


def _node_descriptor(node: IRCell) -> Dict[str, object]:
    """Build a (json-serializable) description of an operator for the config file."""
    return {
        'cid': node.cid,
        'kind': type(node).__name__,
        'name': node.name,
        'signature': getattr(node, 'signature', '') or '',
        'device': list(node.device),
    }


def _iter_forward_segments(execplan: ExecutionPlan):
    """Yield every unique forward segment used by the execution plan."""
    visited = set()
    for devid in execplan.devices():
        for node in execplan.seq(devid):
            if isinstance(node, ExeReuseCell):
                node = node.cell
            if not (isinstance(node, IRSegment) and node.isfw()):
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            yield node


def _is_yaml(path: Union[str, Path]) -> bool:
    return str(path).lower().endswith(('.yaml', '.yml'))


def _read_config(path: Union[str, Path]) -> Dict:
    with open(path, 'r') as f:
        if _is_yaml(path):
            import yaml
            return yaml.safe_load(f)
        return json.load(f)


def _write_config(config: Dict, path: Union[str, Path]) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, 'w') as f:
        if _is_yaml(path):
            import yaml
            yaml.safe_dump(config, f, sort_keys=False)
        else:
            json.dump(config, f, indent=2)


def dump_schedule(execplan: ExecutionPlan, path: Union[str, Path]) -> Dict:
    """Dump the current op order of every forward segment to a config file.

    The produced file lists, for every forward segment, its operators in the order
    they are currently scheduled.  Reorder the entries of a segment's ``order`` list
    to change the schedule, then feed the file back through
    :func:`load_schedule_order` / :meth:`Reschedule.apply`.

    Args:
        execplan: the execution plan to dump.
        path: destination file path. ``.yaml`` / ``.yml`` produces YAML, otherwise
            JSON is used.

    Returns:
        the config dictionary that was written.
    """
    segments_cfg = []
    for segment in _iter_forward_segments(execplan):
        segments_cfg.append({
            'segment_cid': segment.cid,
            'device': list(segment.device),
            'order': [_node_descriptor(child) for child in segment.nodes()],
        })
    config = {
        'version': SCHEDULE_CONFIG_VERSION,
        'comment': (
            'Operator schedule. Reorder the entries of each segment "order" list to '
            'change the execution order. Operators are matched by "cid". The order is '
            'a preference: data dependencies are always enforced, so an illegal order '
            'is automatically corrected to the nearest legal one.'
        ),
        'segments': segments_cfg,
    }
    _write_config(config, path)
    _logger.info(f'op reschedule: dumped schedule of {len(segments_cfg)} segment(s) to {path}')
    return config


def load_schedule_order(config: Union[str, Path, Dict]) -> Dict[int, int]:
    """Load a schedule config and return a mapping from operator cid to its rank.

    The rank is derived from the position of the operator in the (possibly edited)
    ``order`` lists, flattened across segments in file order.  A smaller rank means
    the operator is preferred to be scheduled earlier.

    Args:
        config: a path to a config file, or an already-parsed config dictionary.

    Returns:
        a ``{cid: rank}`` mapping.
    """
    if not isinstance(config, dict):
        config = _read_config(config)
    version = config.get('version')
    if version != SCHEDULE_CONFIG_VERSION:
        _logger.warning(
            f'schedule config version {version} does not match the expected '
            f'version {SCHEDULE_CONFIG_VERSION}; trying to load it anyway'
        )
    order_map: Dict[int, int] = {}
    rank = 0
    for segment in config.get('segments', []):
        for entry in segment.get('order', []):
            cid = int(entry['cid'])
            if cid in order_map:
                _logger.warning(f'duplicated cid {cid} in schedule config, keeping the first one')
                continue
            order_map[cid] = rank
            rank += 1
    return order_map


def config_priority(order_map: Dict[int, int]) -> Callable[[IRCell], Hashable]:
    """Build a topological-sort priority from a ``{cid: rank}`` mapping.

    Operators listed in the mapping are preferred in the recorded order; operators
    that are absent keep their original relative order after the listed ones.
    """
    # operators that are not listed get an "infinite" rank so that they are ordered
    # after the listed ones, while the topological sort falls back to their original
    # order to break the remaining ties.
    sentinel = len(order_map) + 1

    def priority(node: IRCell) -> Hashable:
        return order_map.get(node.cid, sentinel)

    return priority


# ---------------------------------------------------------------------------
# Schedule visualization (Graphviz DOT)
# ---------------------------------------------------------------------------
#
# :func:`visualize_schedule` renders the operators of every forward segment laid
# out linearly in their *current* execution order (the order they will be emitted
# into the generated code), connected by:
#
# - thin grey "program order" connectors, so the linear schedule is easy to read;
# - coloured dependency arrows that encode the causal constraints between ops
#   (read-after-write, write-after-read, write-after-write and the communication
#   serialization edges).
#
# Dump the graph before and after :class:`Reschedule` to compare the two schedules.

# colour / style of each dependency-edge kind
_EDGE_STYLE = {
    'raw': ('#1f77b4', 'read-after-write'),    # blue
    'war': ('#ff7f0e', 'write-after-read'),    # orange
    'waw': ('#d62728', 'write-after-write'),   # red
    'comm': ('#2ca02c', 'comm-serialization'),  # green
    'order': ('#9467bd', 'segment-order'),     # purple
}


def _dot_escape(text: str) -> str:
    return str(text).replace('\\', '\\\\').replace('"', '\\"')


def _node_dot_label(idx: int, node: IRCell) -> str:
    cell = node.cell if isinstance(node, ExeReuseCell) else node
    name = getattr(cell, 'name', '') or type(cell).__name__
    if isinstance(cell, IRSegment):
        name = f"segment({'fw' if cell.isfw() else 'bw'})"
    elif isinstance(cell, IRAdapter):
        name = 'adapter(comm)'
    elif isinstance(cell, IRWeightReducer):
        name = 'reducer'
    # escape each line separately, then join with a graphviz newline (``\n``) so the
    # separator is not turned into a literal backslash-n by the escaping.
    line1 = _dot_escape(f'#{idx} {name}')
    line2 = _dot_escape(f'cid={cell.cid} {type(cell).__name__}')
    return f'{line1}\\n{line2}'


def _node_dot_fill(node: IRCell) -> str:
    cell = node.cell if isinstance(node, ExeReuseCell) else node
    if isinstance(cell, (IRAdapter, IRWeightReducer)):
        return '#d5f5d5'   # light green for communication ops
    if isinstance(cell, IRSegment):
        return '#fff2cc'   # light yellow for segments
    return '#dae8fc'       # light blue for compute ops


def _cluster_to_dot(
    lines: List[str],
    cluster_idx: int,
    label: str,
    nodes: List[IRCell],
    dep: 'OpDependencyGraph',
) -> None:
    """Emit one DOT cluster: nodes in linear order + dependency arrows."""
    node_id = {node: f'c{cluster_idx}_{idx}' for idx, node in enumerate(nodes)}
    lines.append(f'  subgraph cluster_{cluster_idx} {{')
    lines.append(f'    label="{_dot_escape(label)}";')
    lines.append('    color="#999999";')
    for idx, node in enumerate(nodes):
        lines.append(
            f'    {node_id[node]} '
            f'[label="{_node_dot_label(idx, node)}", '
            f'fillcolor="{_node_dot_fill(node)}"];'
        )
    # program-order connectors keep the nodes laid out in the current order
    for idx in range(len(nodes) - 1):
        lines.append(
            f'    {node_id[nodes[idx]]} -> {node_id[nodes[idx + 1]]} '
            f'[color="#cccccc", arrowhead=none, weight=100];'
        )
    lines.append('  }')
    # dependency arrows (do not constrain the layout so the linear order stays)
    for src, dst, kinds in dep.edges():
        kind = kinds[0]
        color, _ = _EDGE_STYLE.get(kind, ('#000000', kind))
        style = 'dashed' if kind in ('comm', 'order') else 'solid'
        lines.append(
            f'  {node_id[src]} -> {node_id[dst]} '
            f'[color="{color}", style={style}, constraint=false];'
        )


def schedule_to_dot(
    execplan: ExecutionPlan,
    name: str = 'schedule',
    scope: str = 'segment',
) -> str:
    """Build a Graphviz DOT string visualizing the current op schedule.

    Args:
        execplan: the execution plan to visualize.
        name: the digraph name.
        scope: ``'segment'`` visualizes the operators *inside* every forward segment;
            ``'sequence'`` visualizes the cross-segment execution sequence of every
            device (segments shown as boxes, with the adapters / reducers between
            them). Dependency / communication / segment-order arrows are drawn.
    """
    if scope not in ('segment', 'sequence'):
        raise ValueError(f"invalid scope {scope!r}, expected 'segment' | 'sequence'")

    lines: List[str] = [
        f'digraph {name} {{',
        '  rankdir=TB;',
        '  graph [fontname="monospace"];',
        '  node [shape=box, style="rounded,filled", fontname="monospace", fontsize=10];',
        '  edge [fontname="monospace", fontsize=8];',
    ]

    if scope == 'segment':
        for seg_idx, segment in enumerate(_iter_forward_segments(execplan)):
            nodes = list(segment.nodes())
            dep = OpDependencyGraph(nodes)
            label = f'segment cid={segment.cid} device={list(segment.device)}'
            _cluster_to_dot(lines, seg_idx, label, nodes, dep)
    else:  # sequence
        for dev_idx, devid in enumerate(execplan.devices()):
            # keep the (distinct) sequence nodes; ExeReuseCell wrappers are unwrapped
            # only for the label/colour, not for identity (the underlying cells are
            # shared across micro-batches in pipeline schedules).
            nodes = list(execplan.seq(devid))
            dep = OpDependencyGraph(
                nodes, serialize_segments=True, comm_types=(IRAdapter,))
            _cluster_to_dot(lines, dev_idx, f'device {devid} execution sequence', nodes, dep)

    lines.append('}')
    return '\n'.join(lines) + '\n'



def visualize_schedule(
    execplan: ExecutionPlan,
    path: Union[str, Path],
    scope: str = 'segment',
) -> str:
    """Write a Graphviz visualization of the current op schedule to ``path``.

    The operators are arranged linearly in their current execution order with
    dependency arrows between them.  Dump it before and after :class:`Reschedule` to
    compare the two schedules.

    A ``.dot`` file is always written.  If ``path`` has an image extension
    (``.png`` / ``.svg`` / ``.pdf``) and the ``graphviz`` package is available, the
    image is rendered as well; otherwise only the ``.dot`` file is produced.

    Args:
        execplan: the execution plan to visualize.
        path: output file path.
        scope: ``'segment'`` (inside each forward segment) or ``'sequence'`` (the
            cross-segment execution sequence of every device).

    Returns:
        the DOT source string.
    """
    dot = schedule_to_dot(execplan, scope=scope)
    path = str(path)
    ext = os.path.splitext(path)[1].lower()

    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)

    image_exts = ('.png', '.svg', '.pdf')
    if ext in image_exts:
        dot_path = path[: -len(ext)] + '.dot'
    else:
        dot_path = path
    with open(dot_path, 'w') as f:
        f.write(dot)

    if ext in image_exts:
        if not _render_dot(dot, dot_path, path, ext[1:]):
            _logger.warning(
                f'could not render schedule image to {path}; wrote DOT source to '
                f'{dot_path} instead. Render it manually with: '
                f'dot -T{ext[1:]} {dot_path} -o {path}'
            )
    _logger.info(f'op reschedule: wrote schedule visualization to {dot_path}')
    return dot


def insert_path_suffix(path: str, suffix: Optional[str]) -> str:
    """Insert ``suffix`` before the file extension, e.g. ``g.dot`` -> ``g.before.dot``."""
    if not suffix:
        return path
    root, ext = os.path.splitext(path)
    return f'{root}.{suffix}{ext}'


def _render_dot(dot: str, dot_path: str, out_path: str, fmt: str) -> bool:
    """Render a DOT source to an image. Returns whether rendering succeeded.

    Tries the ``dot`` command line binary first, then the ``graphviz`` python
    package. Both are optional; if neither is available, returns ``False``.
    """
    import shutil
    import subprocess

    dot_bin = shutil.which('dot')
    if dot_bin:
        try:
            subprocess.run(
                [dot_bin, f'-T{fmt}', dot_path, '-o', out_path],
                check=True, capture_output=True,
            )
            return True
        except Exception as e:  # noqa: BLE001 - rendering is best-effort
            _logger.warning(f'`dot` failed to render {out_path}: {e}')

    try:
        import graphviz
        graphviz.Source(dot).render(outfile=out_path, format=fmt, cleanup=True)
        return True
    except Exception:  # noqa: BLE001 - rendering is best-effort
        return False


class Reschedule(PlanPass):
    """Reschedule the operators inside every forward segment of an execution plan."""

    @staticmethod
    def apply(
        execplan: ExecutionPlan,
        priority: Optional[Callable[[IRCell], Hashable]] = None,
        config: Optional[Union[str, Path, Dict]] = None,
        scope: str = 'segment',
    ) -> ExecutionPlan:
        """Reschedule operators of ``execplan``.

        Args:
            execplan: the execution plan to reschedule (modified in place).
            priority: an optional tie-breaking key function forwarded to
                :meth:`OpDependencyGraph.topological_sort`.
            config: an optional schedule config (a path to a file produced by
                :func:`dump_schedule`, or an already-parsed dict). When given, the
                operator order recorded in the config drives the rescheduling and
                takes precedence over ``priority``.
            scope: what to reschedule:
                - ``'segment'`` (default): reorder operators *inside* every forward
                  segment.
                - ``'sequence'``: reorder the cross-segment execution sequence
                  (segments keep their relative order; adapters / reducers, e.g.
                  asynchronous communication, may move within their data window).
                - ``'both'``: do both.
        """
        if config is not None:
            order_map = load_schedule_order(config)
            priority = config_priority(order_map)

        if scope not in ('segment', 'sequence', 'both'):
            raise ValueError(f"invalid scope {scope!r}, expected 'segment' | 'sequence' | 'both'")

        if scope in ('segment', 'both'):
            visited = set()
            reordered = 0
            for devid in execplan.devices():
                for node in execplan.seq(devid):
                    if isinstance(node, ExeReuseCell):
                        node = node.cell
                    if not (isinstance(node, IRSegment) and node.isfw()):
                        continue
                    if id(node) in visited:
                        continue
                    visited.add(id(node))
                    if Reschedule._reschedule_segment(node, priority):
                        reordered += 1
            _logger.info(f'op reschedule: reordered ops inside {reordered} forward segment(s)')

        if scope in ('sequence', 'both'):
            reordered = Reschedule._reschedule_sequence(execplan, priority)
            _logger.info(f'op reschedule: reordered the execution sequence of {reordered} device(s)')

        return execplan

    @staticmethod
    def _reschedule_sequence(
        execplan: ExecutionPlan,
        priority: Optional[Callable[[IRCell], Hashable]] = None,
    ) -> int:
        """Reschedule the cross-segment execution sequence of every device.

        Segments keep their original relative order (the forward -> backward
        dependency is implicit and not visible in the tensor data-flow); only the
        adapters / reducers between them are free to move within their data window.
        This is what lets an asynchronous communication (e.g. an ``irecv``) be issued
        as early as its data allows, instead of right before the consuming operator.

        A deliberately constructed pipeline schedule is left untouched.

        Returns the number of device sequences that changed.
        """
        # never reorder a deliberately constructed pipeline schedule
        if execplan.graph.sched is not None:
            return 0

        reordered = 0
        for devid in execplan.devices():
            seq = execplan.at(devid)   # the actual per-device list (mutated in place)
            nodes = list(seq)
            if len(nodes) <= 1:
                continue
            graph = OpDependencyGraph(
                nodes,
                serialize_segments=True,
                comm_types=(IRAdapter,),
            )
            new_order = graph.topological_sort(priority)
            changed = len(new_order) != len(nodes) or \
                any(a is not b for a, b in zip(new_order, nodes))
            if changed:
                seq[:] = new_order
                reordered += 1
        return reordered

    @staticmethod
    def _reschedule_segment(
        segment: IRSegment,
        priority: Optional[Callable[[IRCell], Hashable]] = None,
    ) -> bool:
        """Reschedule the direct child operators of a single segment.

        Returns whether the order of the segment changed.
        """
        nodes: List[IRCell] = list(segment.nodes())

        # split into maximal contiguous runs sharing the same recompute group,
        # matching how `ModuleCodeGen.emit_segment` groups the nodes. Operators are
        # only reordered within a run so that recompute (gradient-checkpointing)
        # boundaries are never crossed.
        runs: List[List[IRCell]] = list(
            more_itertools.split_when(nodes, lambda prev, curr: prev.recompute != curr.recompute)
        )

        new_nodes: List[IRCell] = []
        for run in runs:
            if len(run) <= 1:
                new_nodes.extend(run)
                continue
            graph = OpDependencyGraph(run)
            new_nodes.extend(graph.topological_sort(priority))

        # recurse into nested forward segments
        for node in new_nodes:
            if isinstance(node, IRSegment) and node.isfw():
                Reschedule._reschedule_segment(node, priority)

        changed = len(new_nodes) != len(nodes) or \
            any(a is not b for a, b in zip(new_nodes, nodes))
        if changed:
            # write back the rescheduled order and rebuild the producer / consumer
            # bookkeeping so that it stays consistent with the new node order.
            segment._nodes = new_nodes
            segment._reorder_producer_consumer()
        return changed
