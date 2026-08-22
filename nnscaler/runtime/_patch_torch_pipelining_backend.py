# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates

# NOTE: this file is copied from torch.distributed.pipelining._backward.py
# and modified to support nnscaler runtime. The original file can be found at
# https://github.com/pytorch/pytorch/blob/v2.13.0/torch/distributed/pipelining/_backward.py

# We have to copy the code here because we need to modify the behavior of stage_backward_weight
# to support AccumulateGrad hook (reducer depends on it).
# and stage_backward_input and stage_backward_weight are paired functions that are used together,
# so we need to copy both of them.

import collections
from contextlib import contextmanager
from collections.abc import Iterator, Sequence
from typing import Any

import torch
from torch.autograd.function import BackwardCFunction
from torch.autograd.graph import GradientEdge, Node
from torch.nn import Parameter

from torch.distributed.pipelining._debug import map_debug_info

from nnscaler.flags import RuntimeFlag


@contextmanager
def _fbw_phase(
    phase: str,
    deferred_tasks: list | None = None,
):
    previous = RuntimeFlag.fbw_phase
    previous_tasks = RuntimeFlag.fbw_deferred_tasks
    RuntimeFlag.fbw_phase = phase
    RuntimeFlag.fbw_deferred_tasks = deferred_tasks
    try:
        yield
    finally:
        RuntimeFlag.fbw_phase = previous
        RuntimeFlag.fbw_deferred_tasks = previous_tasks


def _get_grad_fn_or_grad_acc(t: torch.Tensor) -> Node | None:
    """
    Get the grad function or grad accumulator for a tensor.

    Accumulate grad nodes are lazily created, so we need to a
    dummy view in order to trigger its creation.
    """
    if t.requires_grad and t.grad_fn is None:
        # if no grad function (leaf tensors) we use view
        viewed_t = t.view_as(t)
        grad_fn = viewed_t.grad_fn
        if grad_fn is not None:
            return grad_fn.next_functions[0][0]
        else:
            raise RuntimeError(
                "Attempted to get grad_fn, but got None."
                "Is this being created in a no-grad context?"
            )
    else:
        return t.grad_fn


def reverse_closure(
    roots: list[Node], target_nodes: set[Node], reverse_edges_dict
) -> tuple[set[Node], set[Node]]:
    """
    This function returns the reverse closure of the given roots,
    i.e. the set of nodes that can be reached from the roots by following the
    reverse edges of the graph. The target_nodes are the nodes that we want to
    include in the closure.
    """
    # Recurse until we reach a target node
    closure: set[Node] = set()
    visited_target_nodes = set()
    q: collections.deque[Node] = collections.deque()
    for node in roots:
        if node is not None and node not in closure:
            closure.add(node)
            q.append(node)
    while q:
        node = q.popleft()
        reverse_edges = reverse_edges_dict[node]
        for fn in reverse_edges:
            if fn in closure or fn is None:
                continue
            if fn in target_nodes:
                visited_target_nodes.add(fn)
                continue
            closure.add(fn)
            q.append(fn)
    return closure, visited_target_nodes


def construct_reverse_graph(roots: list[Node]) -> dict[Node, list[Node]]:
    q: collections.deque[Node] = collections.deque()
    root_seen: set[Node] = set()
    reverse_edges_dict: dict[Node, list[Node]] = collections.defaultdict(list)
    for node in roots:
        if node is not None and node not in root_seen:
            q.append(node)
            root_seen.add(node)
    while q:
        node = q.popleft()
        for fn, _ in node.next_functions:
            if fn is not None:
                if len(reverse_edges_dict[fn]) == 0:
                    q.append(fn)
                reverse_edges_dict[fn].append(node)
    return reverse_edges_dict


def _fully_covered_by_edges(
    node: Node,
    reverse_edges_dict: dict[Node, list[Node]],
    covered_edges: set[tuple[Node, Node]],
    coverage_cache: dict[Node, bool],
) -> bool:
    """Whether every producer path reaches ``node`` through a covered edge.

    Keep this recursive helper at module scope.  A nested recursive function
    captures itself in a closure cell; that reference cycle can retain the
    reverse autograd graph, including all of its saved CUDA tensors, until a
    generation-2 Python garbage collection.
    """
    if node in coverage_cache:
        return coverage_cache[node]
    producers = reverse_edges_dict[node]
    covered = bool(producers) and all(
        (node, producer) in covered_edges
        or _fully_covered_by_edges(
            producer,
            reverse_edges_dict,
            covered_edges,
            coverage_cache,
        )
        for producer in producers
    )
    coverage_cache[node] = covered
    return covered


def _select_cache_frontier(
    node: Node,
    target: Node,
    reverse_edges_dict: dict[Node, list[Node]],
    cached_weight_grads: dict[tuple[Node, Node], list[torch.Tensor]],
    indirect_records_by_edge: dict[
        tuple[Node, Node], list[dict[str, Any]]
    ],
    selected_direct_grads: dict[Node, list[torch.Tensor]],
    selected_indirect_targets: dict[int, set[Node]],
    visited_edges: set[tuple[Node, Node]],
) -> None:
    """Select the nearest non-overlapping cached edge on each producer path."""
    for producer in reverse_edges_dict[node]:
        edge = (node, producer)
        if edge in visited_edges:
            continue
        visited_edges.add(edge)
        if edge in cached_weight_grads:
            selected_direct_grads[target].extend(cached_weight_grads[edge])
            continue
        records = indirect_records_by_edge.get(edge)
        if records:
            for record in records:
                selected_indirect_targets[id(record)].add(target)
            continue
        _select_cache_frontier(
            producer,
            target,
            reverse_edges_dict,
            cached_weight_grads,
            indirect_records_by_edge,
            selected_direct_grads,
            selected_indirect_targets,
            visited_edges,
        )


def _is_parameter_adapter_node(node: Node) -> bool:
    """Whether a backward node only adapts/aggregates a parameter tensor."""
    name = type(node).__name__
    return name.startswith((
        "AliasBackward",
        "AllReduceIdentityBackward",
        "AsStridedBackward",
        "CatBackward",
        "CloneBackward",
        "CopySlices",
        "ExpandBackward",
        "IdentityAllreduceBackward",
        "IdentityBackward",
        "NarrowBackward",
        "PermuteBackward",
        "ReshapeAliasBackward",
        "SelectBackward",
        "SliceBackward",
        "SplitBackward",
        "SplitWithSizesBackward",
        "SqueezeBackward",
        "StackBackward",
        "TBackward",
        "ToCopyBackward",
        "TransposeBackward",
        "UnbindBackward",
        "UnsqueezeBackward",
        "ViewBackward",
    ))


def _reachable_targets(root: Node, targets: set[Node]) -> set[Node]:
    """Return target leaves reachable through parameter-only adapter nodes."""
    reached: set[Node] = set()
    seen: set[Node] = set()
    pending: collections.deque[Node] = collections.deque([root])
    while pending:
        node = pending.popleft()
        if node in seen:
            continue
        seen.add(node)
        if node in targets:
            reached.add(node)
            continue
        # A compiled backward also returns dInput. Following that edge through
        # the whole preceding model can eventually reach many parameters, but
        # it is not a cached dWeight path. Only cross cheap view/alias/reducer
        # adapters that can sit between a compiled dWeight and its flat leaf.
        if not _is_parameter_adapter_node(node):
            continue
        pending.extend(
            next_node
            for next_node, _ in node.next_functions
            if next_node is not None
        )
    return reached


def get_param_groups(
    inputs: list[Node], params: list[Node], reverse_edges_dict
) -> list[dict[str, Any]]:
    """
    Given a list of inputs and a list of parameters, return a list of parameter
    groups, where each group contains the parameters and the intermediates that
    are connected to the parameters.

    The returned list of parameter groups is a list of dictionaries, where each
    dictionary contains the following keys:
    - "params": a set of parameters
    - "intermediates": a set of intermediates

    The returned list of parameter groups is a list of dictionaries,
    """
    # reverse graph that starts with inputs, and goes up to the dOutput or the loss,
    # but omits weights and any subgraphs connecting weights to this closure
    inputs_closure, _ = reverse_closure(inputs, set(), reverse_edges_dict)
    # A parameter can reach more than one split intermediate, and a later
    # shared parameter can bridge intermediate sets that were previously
    # disjoint. Build the connected components explicitly; incrementally
    # merging dictionaries leaves stale aliases behind when such a bridge is
    # encountered and can consequently drop or double-count dWeight paths.
    parent: dict[Node, Node] = {}

    def find(node: Node) -> Node:
        root = node
        while parent[root] is not root:
            root = parent[root]
        while parent[node] is not node:
            next_node = parent[node]
            parent[node] = root
            node = next_node
        return root

    def union(left: Node, right: Node) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root is not right_root:
            parent[right_root] = left_root

    param_intermediates: list[tuple[Node, set[Node]]] = []
    for param in params:
        _, intersected = reverse_closure(
            [param], inputs_closure, reverse_edges_dict
        )
        if not intersected:
            continue
        param_intermediates.append((param, intersected))
        intermediates = iter(intersected)
        first = next(intermediates)
        parent.setdefault(first, first)
        for intermediate in intermediates:
            parent.setdefault(intermediate, intermediate)
            union(first, intermediate)

    groups_by_root: dict[Node, dict[str, set[Node]]] = {}
    for param, intermediates in param_intermediates:
        root = find(next(iter(intermediates)))
        param_group = groups_by_root.setdefault(
            root, {"params": set(), "intermediates": set()}
        )
        param_group["params"].add(param)
        param_group["intermediates"].update(intermediates)

    return list(groups_by_root.values())


def _autograd_grad_for_inputs(
    outputs: Sequence[torch.Tensor],
    inputs: Sequence[Any],
    grad_outputs: Sequence[torch.Tensor | None] | None = None,
    retain_graph: bool = False,
    allow_unused: bool = False,
) -> tuple[torch.Tensor | None, ...]:
    """Compute input gradients, returning ``None`` for non-grad inputs."""
    # Some inputs may not be used or may not require gradients, so we filter them out
    # before calling autograd.grad and place None for those positions in the result.
    grad_indices: list[int] = []
    inputs_requiring_grad: list[torch.Tensor] = []
    for i, inp in enumerate(inputs):
        if isinstance(inp, torch.Tensor) and inp.requires_grad:
            grad_indices.append(i)
            inputs_requiring_grad.append(inp)

    if not inputs_requiring_grad:
        return tuple(None for _ in inputs)

    grads = torch.autograd.grad(
        outputs=outputs,
        inputs=inputs_requiring_grad,
        grad_outputs=grad_outputs,
        retain_graph=retain_graph,
        allow_unused=allow_unused,
    )

    result: list[torch.Tensor | None] = [None] * len(inputs)
    for idx, g in zip(grad_indices, grads, strict=True):
        result[idx] = g
    return tuple(result)


def stage_backward_input_selective(
    stage_outputs_or_loss: list[torch.Tensor],
    output_grads: list[torch.Tensor] | None,
    input_values: list[Any],
) -> tuple[tuple[torch.Tensor | None, ...], list[dict[str, Any]]]:
    """Run dInput while deferring only explicitly registered dWeight work.

    This follows Megatron's delayed-wgrad model: the ordinary backward owns
    small/native parameter gradients, while phase-aware Linear/MoE Functions
    retain their operands and register a direct ``backward_dw``-style task.
    It avoids discovering and retaining the full autograd graph in Python for
    every microbatch.
    """
    valid_outputs: list[torch.Tensor] = []
    valid_output_grads: list[torch.Tensor | None] = []
    for i, stage_output in enumerate(stage_outputs_or_loss):
        if not stage_output.requires_grad and stage_output.grad_fn is None:
            continue
        valid_outputs.append(stage_output)
        valid_output_grads.append(
            torch.ones_like(stage_output) if output_grads is None else output_grads[i]
        )

    for input_value in input_values:
        if isinstance(input_value, torch.Tensor) and input_value.requires_grad:
            input_value.retain_grad()

    deferred_tasks: list = []
    if valid_outputs:
        with _fbw_phase("input", deferred_tasks):
            torch.autograd.backward(
                valid_outputs,
                grad_tensors=valid_output_grads,
            )

    dinputs = tuple(
        input_value.grad
        if isinstance(input_value, torch.Tensor) and input_value.requires_grad
        else None
        for input_value in input_values
    )
    return dinputs, [{
        "params": set(),
        "intermediates": [],
        "grads": [],
        "deferred_tasks": deferred_tasks,
    }]


def stage_backward_input(
    stage_outputs_or_loss: list[torch.Tensor],
    output_grads: list[torch.Tensor] | None,
    input_values: list[Any],
    weights: Iterator[Parameter],
) -> tuple[tuple[torch.Tensor | None, ...], list[dict[str, Any]]]:
    """
    Compute dInput and retain the minimum state needed by delayed dWeight.

    The dInput GraphTask targets stage inputs only. Phase-aware custom
    Functions retain their dWeight operands in deferred tasks, while native
    autograd paths retain graph split points for the W phase. Keeping weights
    out of this GraphTask is important: adding them as targets changes the
    engine traversal order and can perturb dInput numerics.
    """
    weights = tuple(weights)
    valid_outputs: list[torch.Tensor] = []
    valid_output_grads: list[torch.Tensor | None] = []
    for i, stage_output in enumerate(stage_outputs_or_loss):
        if not stage_output.requires_grad and stage_output.grad_fn is None:
            continue
        valid_outputs.append(stage_output)
        valid_output_grads.append(
            torch.ones_like(stage_output) if output_grads is None else output_grads[i]
        )

    stage_output_grad_fns: list[Node] = list(
        filter(None, map(_get_grad_fn_or_grad_acc, valid_outputs))
    )
    stage_input_grad_fns: list[Node] = list(
        filter(
            None,
            (
                _get_grad_fn_or_grad_acc(inp)
                for inp in input_values
                if isinstance(inp, torch.Tensor)
            ),
        )
    )
    weight_grad_pairs = [
        (weight, _get_grad_fn_or_grad_acc(weight)) for weight in weights
    ]
    weight_grad_fns: list[Node] = [
        grad_fn for _, grad_fn in weight_grad_pairs if grad_fn is not None
    ]
    reverse_edges_dict = construct_reverse_graph(stage_output_grad_fns)
    param_groups = get_param_groups(
        stage_input_grad_fns, weight_grad_fns, reverse_edges_dict
    )

    # Opaque Python/AOT nodes can produce a complete dWeight even when only
    # dInput is requested. Retain that returned storage directly; the Tensor
    # reference keeps it alive until W and avoids the previous full-size clone.
    weight_grad_fn_set = set(weight_grad_fns)
    cached_weight_grads: dict[
        tuple[Node, Node], list[torch.Tensor]
    ] = collections.defaultdict(list)
    cached_weight_edges: list[dict[str, Any]] = []
    potential_cached_edges: set[tuple[Node, Node]] = set()
    potentially_fully_cached_params: set[Node] = set()
    opaque_affected_params: set[Node] = set()
    graph_nodes = set(stage_output_grad_fns) | set(reverse_edges_dict)
    graph_nodes.update(
        node
        for reverse_edges in reverse_edges_dict.values()
        for node in reverse_edges
    )

    deferred_tasks: list = []
    handles = []
    try:
        for node in graph_nodes:
            if not isinstance(node, BackwardCFunction):
                continue
            weight_edges = tuple(
                (index, next_node)
                for index, (next_node, _) in enumerate(node.next_functions)
                if next_node in weight_grad_fn_set
            )
            potential_cached_edges.update(
                (grad_acc, node) for _, grad_acc in weight_edges
            )
            indirect_weight_edges = []
            for index, (next_node, input_nr) in enumerate(node.next_functions):
                if next_node is None or next_node in weight_grad_fn_set:
                    continue
                targets = _reachable_targets(next_node, weight_grad_fn_set)
                if not targets:
                    continue
                record = {
                    "producer": node,
                    "next_node": next_node,
                    "edge": GradientEdge(next_node, input_nr),
                    "targets": targets,
                    "grad": None,
                }
                cached_weight_edges.append(record)
                indirect_weight_edges.append((index, record))

            if not weight_edges and not indirect_weight_edges:
                continue

            def get_weight_cache_hook(
                node,
                weight_edges,
                indirect_weight_edges,
            ):
                def hook(grad_inputs, grad_outputs):
                    for index, grad_acc in weight_edges:
                        if index >= len(grad_inputs):
                            continue
                        grad = grad_inputs[index]
                        if grad is not None:
                            opaque_affected_params.add(grad_acc)
                            if grad_acc in potentially_fully_cached_params:
                                cached_weight_grads[(grad_acc, node)].append(
                                    grad.detach()
                                )
                    for index, record in indirect_weight_edges:
                        if index >= len(grad_inputs):
                            continue
                        grad = grad_inputs[index]
                        if grad is not None:
                            opaque_affected_params.update(record["targets"])
                            if record["cache_targets"]:
                                record["grad"] = grad.detach()

                return hook

            handles.append(node.register_hook(
                get_weight_cache_hook(
                    node,
                    weight_edges,
                    indirect_weight_edges,
                )
            ))

        # Decide cache eligibility from graph topology before I starts. A
        # partial opaque contribution is not consumed by the cache path: its
        # parameter group must still run native W for the uncovered branches.
        # Do not retain such a potentially multi-GiB dWeight merely to discard
        # it after the rest of I has already reached its memory peak.
        potential_cached_edges.update(
            (record["next_node"], record["producer"])
            for record in cached_weight_edges
        )
        potential_coverage_cache: dict[Node, bool] = {}

        potentially_fully_cached_params.update(
            grad_acc
            for grad_acc in weight_grad_fn_set
            if _fully_covered_by_edges(
                grad_acc,
                reverse_edges_dict,
                potential_cached_edges,
                potential_coverage_cache,
            )
        )
        for record in cached_weight_edges:
            record["cache_targets"] = (
                record["targets"] & potentially_fully_cached_params
            )

        for param_group in param_groups:
            intermediates = list(param_group["intermediates"])
            param_group["intermediates"] = intermediates
            for index, intermediate in enumerate(intermediates):

                def get_hook(param_group, index):
                    def hook(grad_inputs):
                        if param_group.get("grads") is None:
                            param_group["grads"] = [None] * len(
                                param_group["intermediates"]
                            )
                        # Keep the original storage alive rather than cloning
                        # it. The W GraphTask consumes this exact dInput value.
                        param_group["grads"][index] = tuple(
                            grad.detach() if isinstance(grad, torch.Tensor) else grad
                            for grad in grad_inputs
                        )

                    return hook

                handles.append(intermediate.register_prehook(
                    get_hook(param_group, index)
                ))

        if valid_outputs:
            with _fbw_phase("input", deferred_tasks):
                target_grads = _autograd_grad_for_inputs(
                    valid_outputs,
                    input_values,
                    valid_output_grads,
                    retain_graph=True,
                    allow_unused=True,
                )
                dinputs = target_grads
        else:
            dinputs = tuple(None for _ in input_values)

        for inp, dinput in zip(input_values, dinputs):
            if isinstance(inp, torch.Tensor) and dinput is not None:
                if inp.grad is None:
                    inp.grad = dinput
                else:
                    inp.grad += dinput

        # Every capture hook is single-use for the I GraphTask. Native paths
        # below may run a second GraphTask eagerly; leaving these hooks active
        # would overwrite the saved frontier while it is being consumed.
        for handle in handles:
            handle.remove()
        handles.clear()

        # A callback can fully replace W traversal for an unflattened weight.
        # Generated flat parameters may contain both custom and native slices;
        # those stay in the native graph and are combined with callback output.
        fully_deferred_params: set[Node] = set()
        for task in deferred_tasks:
            for target in getattr(task, "_nnscaler_fbw_targets", ()):
                for weight, grad_acc in weight_grad_pairs:
                    if grad_acc is None:
                        continue
                    # Object identity proves that the callback owns the only
                    # edge to this leaf. A full-size view/alias can still have
                    # native contributions (for example tied embedding/output
                    # weights), so storage/shape equality is not sufficient.
                    same_full_tensor = target is weight
                    if same_full_tensor:
                        fully_deferred_params.add(grad_acc)
        for param_group in param_groups:
            param_group["params"].difference_update(fully_deferred_params)

        cached_producer_edges = set(cached_weight_grads)
        cached_producer_edges.update(
            (record["next_node"], record["producer"])
            for record in cached_weight_edges
            if record["grad"] is not None
        )
        coverage_cache: dict[Node, bool] = {}

        fully_cached_params = {
            grad_acc
            for grad_acc in weight_grad_fn_set
            if _fully_covered_by_edges(
                grad_acc,
                reverse_edges_dict,
                cached_producer_edges,
                coverage_cache,
            )
        }
        # Several opaque nodes can lie on the same weight path. For example,
        # compiled loss backward feeds IdentityBackward, which in turn feeds a
        # MultiRefBackward directly connected to AccumulateGrad. Caching all
        # three hooks would count the same contribution more than once. Walk
        # from each parameter towards the outputs and select the first cached
        # edge on every producer path; this is the closest non-overlapping
        # cache frontier to the parameter.
        indirect_records_by_edge: dict[
            tuple[Node, Node], list[dict[str, Any]]
        ] = collections.defaultdict(list)
        for record in cached_weight_edges:
            if record["grad"] is not None:
                indirect_records_by_edge[
                    (record["next_node"], record["producer"])
                ].append(record)

        selected_direct_grads: dict[
            Node, list[torch.Tensor]
        ] = collections.defaultdict(list)
        selected_indirect_targets: dict[int, set[Node]] = collections.defaultdict(set)

        for grad_acc in fully_cached_params:
            _select_cache_frontier(
                grad_acc,
                grad_acc,
                reverse_edges_dict,
                cached_weight_grads,
                indirect_records_by_edge,
                selected_direct_grads,
                selected_indirect_targets,
                set(),
            )

        # An opaque Python/AOT backward has already paid for these dWeights in
        # I and cannot defer only that part of its computation.  Accumulate the
        # completed results now instead of retaining one full-size dWeight per
        # pending microbatch.  In particular, the output-vocabulary gradient
        # can be several GiB and the zero-bubble schedule may queue multiple I
        # actions before its first W.  A single parameter .grad buffer has the
        # same lifetime as normal training and avoids both the repeated opaque
        # backward and that unbounded queue.
        cached_contributions: dict[Node, list[torch.Tensor]] = (
            collections.defaultdict(list)
        )
        for grad_acc in fully_cached_params:
            cached_contributions[grad_acc].extend(
                selected_direct_grads[grad_acc]
            )

        selected_edge_records = []
        for record in cached_weight_edges:
            targets = (
                record["targets"]
                & selected_indirect_targets[id(record)]
            )
            if record["grad"] is not None and targets:
                selected_edge_records.append({
                    "edge": record["edge"],
                    "grad": record["grad"],
                    "targets": targets,
                })

        if selected_edge_records:
            edge_targets = set().union(*(
                record["targets"] for record in selected_edge_records
            ))
            ordered_targets = tuple(
                grad_acc for grad_acc in weight_grad_fns
                if grad_acc in edge_targets
            )
            with _fbw_phase("native_weight"):
                edge_grads = torch.autograd.grad(
                    tuple(record["edge"] for record in selected_edge_records),
                    tuple(GradientEdge(grad_acc, 0) for grad_acc in ordered_targets),
                    grad_outputs=tuple(
                        record["grad"] for record in selected_edge_records
                    ),
                    retain_graph=True,
                    allow_unused=True,
                )
            for grad_acc, grad in zip(ordered_targets, edge_grads, strict=True):
                if grad is not None:
                    cached_contributions[grad_acc].append(grad)

        weight_by_grad_acc = {
            grad_acc: weight
            for weight, grad_acc in weight_grad_pairs
            if grad_acc is not None
        }
        accumulation_roots: list[torch.Tensor] = []
        accumulation_grads: list[torch.Tensor] = []
        for grad_acc in weight_grad_fns:
            contributions = cached_contributions.get(grad_acc)
            if not contributions:
                continue
            combined_grad = contributions[0]
            for contribution in contributions[1:]:
                combined_grad = combined_grad + contribution.to(
                    combined_grad.dtype
                )
            accumulation_roots.append(weight_by_grad_acc[grad_acc])
            accumulation_grads.append(combined_grad)
        if accumulation_roots:
            with _fbw_phase("native_weight"):
                torch.autograd.backward(
                    accumulation_roots,
                    grad_tensors=accumulation_grads,
                )

        # The cache hooks also observe opaque contributions to partially
        # covered parameters. Those values cannot be used as a complete
        # replacement for W, so an affected parameter group is recomputed
        # below. Drop every cached/frontier reference before that recompute.
        # In particular, a tied output/embedding weight can otherwise keep two
        # full vocabulary dWeights alive while the eager native GraphTask tries
        # to allocate a third one.
        cached_weight_grads.clear()
        for record in cached_weight_edges:
            record["grad"] = None
        cached_contributions.clear()
        selected_direct_grads.clear()
        selected_indirect_targets.clear()
        selected_edge_records.clear()
        accumulation_grads.clear()
        edge_grads = ()
        combined_grad = None
        contribution = None
        grad = None

        for param_group in param_groups:
            param_group["params"].difference_update(fully_cached_params)
            if not param_group["params"]:
                # Every path owned by this group was completed by the opaque
                # I backward above. Drop its captured split-point gradients
                # immediately; otherwise a no-op scheduled W would keep large
                # AOT intermediates alive across all queued microbatches.
                param_group["intermediates"] = []
                param_group["grads"] = []

        # A parameter group touched by an opaque Python/AOT backward cannot
        # defer only its remaining native dWeight work: a second GraphTask
        # either repeats the opaque backward or retains its entire split-point
        # gradient until W. Complete only those affected groups now. Ordinary
        # native groups remain deferred, preserving the I/W contract for paths
        # that autograd can actually split. Phase-aware Linear/MoE Functions
        # still returned no dWeight in ``native_weight`` and keep their large
        # GEMMs in deferred_tasks.
        eager_param_groups = [
            param_group
            for param_group in param_groups
            if param_group["params"] & opaque_affected_params
        ]
        deferred_native_param_groups = [
            param_group
            for param_group in param_groups
            if param_group not in eager_param_groups
        ]
        if eager_param_groups:
            stage_backward_weight(
                iter(weights),
                eager_param_groups,
                retain_graph=bool(deferred_native_param_groups),
            )
        param_groups = deferred_native_param_groups

        param_groups.append({
            "params": set(),
            "intermediates": [],
            "grads": [],
            "deferred_tasks": deferred_tasks,
        })

        # Python nodes are owned by the output graph. Native-only graphs can
        # release that side immediately; Python/AOT graphs remain alive until W.
        if not any(isinstance(node, BackwardCFunction) for node in graph_nodes):
            for tensor in stage_outputs_or_loss:
                if not tensor._is_view():
                    tensor.detach_()

        return dinputs, param_groups
    except Exception as e:
        exc_msg = f"""
        Failed to run stage backward input:
        Stage output: {map_debug_info(stage_outputs_or_loss)}
        Output gradient: {map_debug_info(output_grads)}
        Input: {map_debug_info(input_values)}
        """
        raise RuntimeError(exc_msg) from e
    finally:
        for handle in handles:
            handle.remove()


def stage_backward_weight(
    weights: Iterator[Parameter], param_groups: list[dict[str, Any]], retain_graph=False
) -> tuple[torch.Tensor | None, ...]:
    weights = tuple(weights)
    grad_acc_to_weight = None
    grad_weights = None

    def get_weight_metadata():
        """Build the full parameter/AccumulateGrad map only when required.

        Fine-grained FBW commonly calls this function only to drain custom
        callbacks. Expert wgrad may already have been published at the DeepEP
        dispatch boundary, while Linear callbacks can accumulate directly
        into reducer-owned storage. Neither case needs an alias GraphTask or a
        scan of every stage parameter. Lazily constructing this map avoids
        repeating that Python/autograd work for every segment and microbatch.
        """
        nonlocal grad_acc_to_weight, grad_weights
        if grad_acc_to_weight is None:
            grad_weights = tuple(
                weight for weight in weights if weight.requires_grad
            )
            grad_acc_to_weight = {
                _get_grad_fn_or_grad_acc(weight): weight
                for weight in weights
            }
        return grad_acc_to_weight, grad_weights

    contributions: dict[int, list[torch.Tensor]] = collections.defaultdict(list)
    reducer_accumulated_weights: set[torch.Tensor] = set()

    from nnscaler.runtime.adapter.reducer import (
        accumulate_reducer_grad,
        complete_reducer_grad,
        has_reducer_grad_accumulator,
    )
    from nnscaler.runtime.adapter.nn import (
        mark_deferred_identity_allreduce_grad_ready,
    )
    from nnscaler.runtime.utils import get_grad_dtype

    def record_contribution(
        weight: torch.Tensor,
        grad: torch.Tensor,
        *,
        deferred: bool = False,
    ) -> None:
        if has_reducer_grad_accumulator(weight):
            if not accumulate_reducer_grad(weight, grad):
                raise RuntimeError(
                    "Reducer disappeared while accumulating deferred dWeight"
                )
            reducer_accumulated_weights.add(weight)
            return
        if deferred and isinstance(weight, Parameter) and weight.is_leaf:
            grad_dtype = get_grad_dtype(weight)
            if weight.grad is None:
                weight.grad = (
                    grad if grad.dtype == grad_dtype else grad.to(grad_dtype)
                )
            else:
                weight.grad.add_(grad.to(weight.grad.dtype))
            mark_deferred_identity_allreduce_grad_ready(weight)
            return
        existing = contributions[id(weight)]
        if existing:
            existing[0].add_(grad.to(existing[0].dtype))
        else:
            existing.append(grad)

    # Phase-aware custom Functions retained their actual GEMM operands during
    # I. Compute those dWeights directly, then map aliases/views back to the
    # generated stage parameters without re-entering the model GraphTask.
    # A module-level W action commonly contains several projections (for
    # example Q/K/V/O).  Entering autograd once per projection repeatedly
    # traverses the same generated weight-alias graph and adds a sizeable
    # Python/GraphTask launch cost to every microbatch.  Run the direct GEMM
    # callbacks first, then map all of their results to the stage parameters in
    # one GraphTask.  Autograd preserves the sum when a root is referenced by
    # more than one contribution.
    custom_roots: list[torch.Tensor] = []
    custom_root_grads: list[torch.Tensor] = []
    with _fbw_phase("weight"):
        for state in param_groups:
            for task in state.pop("deferred_tasks", ()):
                task_contributions = task()
                if task_contributions is None:
                    continue
                for target, grad in task_contributions:
                    if grad is None:
                        # A phase-aware custom kernel may have accumulated
                        # directly into the reducer buffer.  Keep the marker
                        # so the reducer lifecycle is completed after all
                        # callbacks and native contributions for this W action.
                        if (
                            isinstance(target, Parameter)
                            and has_reducer_grad_accumulator(target)
                        ):
                            reducer_accumulated_weights.add(target)
                        continue
                    custom_roots.append(target)
                    custom_root_grads.append(grad)
        if custom_roots:
            _, active_grad_weights = get_weight_metadata()
        else:
            active_grad_weights = ()
        if custom_roots and active_grad_weights:
            custom_grads = torch.autograd.grad(
                custom_roots,
                active_grad_weights,
                grad_outputs=custom_root_grads,
                allow_unused=True,
                retain_graph=False,
            )
            for weight, grad in zip(
                active_grad_weights, custom_grads, strict=True
            ):
                if grad is not None:
                    record_contribution(weight, grad, deferred=True)
        else:
            custom_grads = ()
    custom_roots.clear()
    custom_root_grads.clear()
    custom_grads = ()

    active_param_groups = [
        param_group
        for param_group in param_groups
        if param_group.get("params") and any(
            grad is not None
            for grads_tuple in param_group.get("grads", ())
            for grad in grads_tuple
        )
    ]
    if active_param_groups:
        active_grad_acc_to_weight, _ = get_weight_metadata()
    else:
        active_grad_acc_to_weight = {}
    last_active_group = active_param_groups[-1] if active_param_groups else None
    for param_group in param_groups:
        intermediates = param_group.get("intermediates", [])
        valid_edges = []
        valid_grad_outputs: list[torch.Tensor] = []
        for grads_tuple, intermediate in zip(
            param_group.get("grads", ()), intermediates
        ):
            for index, grad in enumerate(grads_tuple):
                if grad is not None:
                    valid_edges.append(GradientEdge(intermediate, index))
                    valid_grad_outputs.append(grad)

        handles = []
        try:
            if len(intermediates) > 1:
                for grads_tuple, intermediate in zip(
                    param_group["grads"], intermediates, strict=True
                ):
                    handles.append(intermediate.register_prehook(
                        lambda grad_outputs, saved=grads_tuple: saved
                    ))

            param_group.pop("intermediates", None)
            target_grad_accs = tuple(param_group.get("params", ()))
            if valid_edges and target_grad_accs:
                target_edges = tuple(
                    GradientEdge(grad_acc, 0) for grad_acc in target_grad_accs
                )
                # Custom Functions only propagate dInput in this traversal;
                # their dWeights already came from deferred tasks above.
                with _fbw_phase("native_weight"):
                    native_grads = torch.autograd.grad(
                        valid_edges,
                        target_edges,
                        grad_outputs=valid_grad_outputs,
                        retain_graph=(
                            retain_graph or param_group is not last_active_group
                        ),
                        allow_unused=True,
                    )
                for grad_acc, grad in zip(
                    target_grad_accs, native_grads, strict=True
                ):
                    if grad is not None:
                        record_contribution(
                            active_grad_acc_to_weight[grad_acc], grad
                        )

            param_group.pop("grads", None)
        finally:
            for handle in handles:
                handle.remove()

    for weight in reducer_accumulated_weights:
        if not complete_reducer_grad(weight):
            raise RuntimeError(
                "Reducer disappeared while completing deferred dWeight"
            )

    accumulation_roots: list[torch.Tensor] = []
    accumulation_grads: list[torch.Tensor] = []
    if contributions:
        _, contribution_grad_weights = get_weight_metadata()
    else:
        contribution_grad_weights = ()
    for weight in contribution_grad_weights:
        weight_contributions = contributions.get(id(weight), ())
        if not weight_contributions:
            continue
        combined_grad = weight_contributions[0]
        for contribution in weight_contributions[1:]:
            combined_grad = combined_grad + contribution.to(combined_grad.dtype)
        accumulation_roots.append(weight)
        accumulation_grads.append(combined_grad)

    if accumulation_roots:
        with _fbw_phase("native_weight"):
            torch.autograd.backward(
                accumulation_roots,
                grad_tensors=accumulation_grads,
                retain_graph=retain_graph,
            )
        for weight in accumulation_roots:
            mark_deferred_identity_allreduce_grad_ready(weight)
    return tuple(weight.grad for weight in weights)


def stage_backward(
    stage_output,
    output_grads,
    input_values,
    outputs_with_grads_idxs: list[int] | None = None,  # deprecated, not used
) -> tuple[torch.Tensor | None, ...]:
    """
    This is a helper function to:
    1. compute the gradients for the stage inputs, and
    2. accumulate gradients for the stage module's parameters.

    Given the input value(s) and the corresponding gradient for the output
    value(s), compute and accumulate gradients for all parameter values (leaves
    in the autograd trace) as well as return a list of the gradients for the
    input values
    """
    if outputs_with_grads_idxs is not None:
        # Deprecated, not used in runtime calls, only exists in compiler
        stage_output = [stage_output[i] for i in outputs_with_grads_idxs]
        output_grads = [output_grads[i] for i in outputs_with_grads_idxs]

    try:
        # stage_output may be a composite datatype like dict. Extract all individual
        # tensor values here
        stage_output_tensors: list[torch.Tensor] = []
        output_grad_tensors: list[torch.Tensor | None] = []

        def extract_tensors_with_grads(
            output_val,
            grad_val,
            # Don't delete me- see [Note: ref cycle]
            extract_tensors_with_grads,
        ):
            if isinstance(output_val, torch.Tensor):
                if not output_val.requires_grad and output_val.grad_fn is None:
                    return
                if not isinstance(grad_val, (torch.Tensor, type(None))):
                    raise AssertionError(
                        f"Expected Tensor or None gradient but got {type(grad_val)}"
                    )
                stage_output_tensors.append(output_val)
                output_grad_tensors.append(grad_val)
            elif isinstance(output_val, (tuple, list)):
                if grad_val is None:
                    return
                if not isinstance(grad_val, (tuple, list)):
                    raise AssertionError(
                        f"grad_value expected to have type {type(output_val)} but got {type(grad_val)}"
                    )
                if not len(output_val) == len(grad_val):
                    raise AssertionError(
                        f"Expected len(output_val) == len(grad_val), got {len(output_val)} != {len(grad_val)}"
                    )
                for ov, gv in zip(output_val, grad_val):
                    extract_tensors_with_grads(
                        ov,
                        gv,
                        extract_tensors_with_grads,
                    )
            elif isinstance(output_val, dict):
                if grad_val is None:
                    return
                if not isinstance(grad_val, dict):
                    raise AssertionError(f"Expected dict, got {type(grad_val)}")
                if not set(output_val.keys()) == set(grad_val.keys()):
                    raise AssertionError(
                        f"Expected keys {set(output_val.keys())}, got {set(grad_val.keys())}"
                    )
                for k in output_val:
                    extract_tensors_with_grads(
                        output_val[k], grad_val[k], extract_tensors_with_grads
                    )
            else:
                # Output is a non-tensor type; just ignore it
                pass

        # Note: ref cycle
        # break a ref cycle that would keep tensors alive until GC runs
        # 1. extract_tensors_with_grads refers to a cell that holds refs to any vars defined in stage_backward
        #    and used in extract_tensors_with_grads
        # 2. extract_tensors_with_grads referred to both stage_output_tensors, output_grad_tensors,
        #    and to itself (extract_tensors_with_grads) since it makes a recursive call
        # 3. stage_output_tensors was kept alive by the above refcycle, and it holds activation tensors, which is bad
        # fix -> explicitly pass in the ref to the fn, so there is no gc cycle anymore
        extract_tensors_with_grads(
            stage_output, output_grads, extract_tensors_with_grads
        )

        torch.autograd.backward(
            stage_output_tensors,
            grad_tensors=output_grad_tensors,  # type: ignore[arg-type]
        )

        # Extract gradients wrt the input values
        grad_inputs: list[torch.Tensor | None] = []
        for val in input_values:
            if isinstance(val, torch.Tensor):
                grad_inputs.append(val.grad)
                # Since gradients that will pass back to previous stages do not require gradient accumulation,
                # by decrementing the gradients' reference count at this point, the memory of gradients will be
                # returned to the allocator as soon as the next micro batch's get_bwd_send_ops comes and current
                # asynchronous send completes.
                # This prevents the gradients from persisting in GPU memory for the entire duration of step_microbatches
                # until clear_runtime_states() is called.
                val.grad = None
            else:
                grad_inputs.append(None)

        # Alternative impl: `torch.autograd.grad`.
        # Note that `torch.autograd.grad` will not accumulate gradients into the
        # model's parameters.
        """
        inputs_with_grad = []
        for val in input_values:
            if isinstance(val, torch.Tensor) and val.requires_grad:
                inputs_with_grad.append(val)

        grad_inputs = torch.autograd.grad(
            stage_output_tensors, inputs_with_grad, output_grad_tensors,  # type: ignore[arg-type]
        )
        """

    except Exception as e:
        exc_msg = f"""
        Failed to run stage backward:
        Stage output: {map_debug_info(stage_output)}
        Output gradient: {map_debug_info(output_grads)}
        Input: {map_debug_info(input_values)}
        """
        raise RuntimeError(exc_msg) from e

    return tuple(grad_inputs)


# TODO: handling requires_grad=False dynamically. Can we analyze this during initial
# IR emission?
def _null_coalesce_accumulate(lhs, rhs):
    """
    Coalesce two values, even if one of them is null, returning the non-null
    value.
    """
    if lhs is None:
        return rhs
    elif rhs is None:
        return lhs
    else:
        return torch.add(lhs, rhs)
