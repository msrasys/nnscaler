#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from collections import defaultdict
from typing import Any
from nnscaler.graph.function.dimops import IRDimops
from nnscaler.ir.cten import IR, IRObject, IRTensor, ValueTrack
from nnscaler.ir.operator import IRFwOperation


class ValueTracker:
    """
    Example:
    >>> vt = ValueTracker()
    >>> vt.track_value(input1)
    >>> vt.track_value(input2)
    >>> ...
    >>> vt.track_nodes([node1])
    >>> vt.track_nodes([node2])
    >>> vt.untrack_node(node2)  # when node2 is folded
    >>> vt.track_nodes([node3])
    >>> ...
    >>> vt.complete_tracking([node1, node3, ...])  # pass all tracked nodes here
    """
    def __init__(self):
        # value_id -> ValueTrack
        # Please note some ValueTracks may be merged together (from annotation)
        # So the key can be different from the id of the ValueTrack
        self._vtm: dict[int, ValueTrack] = {}
        self._equiv_value_ids: dict[int, set[int]] = {}
        # store removed value ids
        # used to delay the removal of value tracks in deps
        self._removed_value_ids: set[int] = set()

    def _add_track_value(self, value: ValueTrack):
        if value.value_id not in self._vtm:
            # always use the updated value track in self._vtm
            self._vtm[value.value_id] = value

        if value.value_id not in self._equiv_value_ids:
            self._equiv_value_ids[value.value_id] = {value.value_id}

    def track_values(self, objs: list[Any]) -> set[int]:
        """
        Track the value tracks of the given objects.
        Args:
            objs (list[Any]): the objects to be tracked
        Returns:
            set[int]: the set of value ids tracked
        """
        value_ids = set()
        for obj in objs:
            value_ids.update(self._track_value(obj))
        return value_ids

    def _track_value(self, value: Any):
        for obj in IR.get_objects(value):
            if isinstance(obj, IRTensor):
                for dt in obj.dim_tracks:
                    self._add_track_value(dt)
                    yield dt.value_id
            else:
                assert isinstance(obj, IRObject)
                self._add_track_value(obj.value_track)
                yield obj.value_track.value_id

    def _update_track_value(self, obj: IRObject):
        if isinstance(obj, IRTensor):
            new_dim_tracks = []
            for dt in obj.dim_tracks:
                new_dim_tracks.append(self._vtm[dt.value_id])
            obj.dim_tracks = new_dim_tracks
        else:
            assert isinstance(obj, IRObject)
            obj.value_track = self._vtm[obj.value_track.value_id]

    def _update_constness(self, obj: IRObject):
        if isinstance(obj, IRTensor):
            for dt in obj.dim_tracks:
                dt.is_constant = dt.is_constant and all(self._vtm[dep].is_constant for dep in dt.deps or [])
        else:
            assert isinstance(obj, IRObject)
            obj.value_track.is_constant = obj.value_track.is_constant and all(self._vtm[dep].is_constant for dep in obj.value_track.deps or [])

    def track_nodes(self, nodes: list[IRFwOperation]):
        """
        Track the value tracks of the input and output objects in the given nodes.
        Here we assume the nodes are topologically sorted.

        Please note we only update the tracks of nodes in arguments.
        For nodes not in arguments, their tracks are not updated.

        Args:
            nodes (list[IRFwOperation]): the nodes to be tracked
        """
        # collect all value tracks from nodes
        if not nodes:
            return

        # collect all involved value ids from nodes
        node_value_ids = set()
        for node in nodes:
            for obj in node.iobjs():
                node_value_ids.update(self._track_value(obj))
            for obj in node.oobjs():
                node_value_ids.update(self._track_value(obj))

        # collect extra value tracks from dimops
        for node in nodes:
            if isinstance(node, IRDimops):
                self._track_dims(node)

        # merge equivalent value tracks together
        done_value_ids = set()
        for value_id in node_value_ids:
            equiv_ids = self._equiv_value_ids[value_id]

            min_value_id = min(equiv_ids)
            if min_value_id in done_value_ids:
                continue
            done_value_ids.add(min_value_id)

            # use the smallest id as the representative
            rep_one = self._vtm[min_value_id]
            for vid in equiv_ids:
                if vid == min_value_id or self._vtm[vid] is rep_one:
                    continue
                # TODO: how we merge dependencies?
                # current we take union (Union may be too strict)
                if rep_one.deps is None:
                    rep_one.deps = self._vtm[vid].deps
                elif self._vtm[vid].deps is not None:
                    # deps can still have duplicates here
                    # because merging of the rest value tracks haven't been done yet
                    # NOTE:
                    # 1. this duplication is temporary,
                    # Duplicated value ids will be removed when we touch the same value track again
                    # in future track_nodes call.
                    # 2. duplication is not harmful for correctness
                    rep_one.deps = list(
                        set(rep_one.deps)
                        .union(self._vtm[vid].deps)
                        .difference(self._removed_value_ids)
                    )
                self._vtm[vid] = rep_one

        self._propagate_tracks(nodes)

    def untrack_node(self, node: IRFwOperation):
        """
        Untrack the value tracks of output objects in the given node.
        This function is used when we fold a node from the graph.

        Args:
            node (IRFwOperation): the node to be untracked
        """
        input_value_ids = set()
        for obj in node.iobjs():
            if isinstance(obj, IRTensor):
                for dt in obj.dim_tracks:
                    input_value_ids.add(dt.value_id)
            else:
                assert isinstance(obj, IRObject)
                input_value_ids.add(obj.value_track.value_id)

        for obj in node.oobjs():
            # we can only remove value tracks that are not used by inputs
            if isinstance(obj, IRTensor):
                for dt in obj.dim_tracks:
                    if dt.value_id not in input_value_ids:
                        self._removed_value_ids.add(dt.value_id)
            else:
                assert isinstance(obj, IRObject)
                if obj.value_track.value_id not in input_value_ids:
                    self._removed_value_ids.add(obj.value_track.value_id)

    def complete_tracking(self, nodes: list[IRFwOperation]):
        """
        Complete the tracking process.
        Should be called after all nodes are tracked.
        """
        # remove all removed value ids for vtm
        # note we don't remove them from equivalence classes
        for removed_id in self._removed_value_ids:
            if self._vtm[removed_id].value_id == removed_id \
                and (new_equiv_cls := self._equiv_value_ids[removed_id].difference(self._removed_value_ids)):
                # change the representative value id of this equivalence class
                # NOTE:
                # In current usage, code should not reach here.
                # As we remove value tracks only for constant irobjects,
                # and all equivalent value tracks should be removed together.
                self._vtm[removed_id].value_id = min(new_equiv_cls)
            self._vtm.pop(removed_id, None)

        # replace dependencies with their representative value tracks
        # which can introduce some duplicates
        # So we use `set` to further dedup dependencies
        for vt in self._vtm.values():
            if vt.deps is not None:
                vt.deps = list(set(
                    self._vtm[d].value_id for d in vt.deps
                    if d not in self._removed_value_ids
                ))

        self._propagate_tracks(nodes)

    def _propagate_tracks(self, nodes: list[IRFwOperation]):
        """
        Update value tracks and constantness information of the input and output objects
        in the given nodes.
        """
        # propagate the merged value tracks back to nodes
        for node in nodes:
            for obj in node.iobjs():
                self._update_track_value(obj)
            for obj in node.oobjs():
                self._update_track_value(obj)

        # propagate the constantness information back to nodes
        for node in nodes:
            for obj in node.iobjs():
                self._update_constness(obj)
            for obj in node.oobjs():
                self._update_constness(obj)

    def _track_dims(self, node: IRDimops):
        """
        Track the dimension values of output tensors according to input tensors.
        This function should be called after shape inference.
        """
        # align the dim_ids of output with inputs
        # not-hidden-dimension means the identifier is all for this dimension
        # for example, in `l (2 h) m`,
        # l and m are not-hidden-dimension identifiers, h is hidden-dimension identifier
        #
        # If the annotation is `l (2 h) m -> l h (m 2 h)`
        # We will get the following relations (nhd->not-hidden-dimension, hd->hidden-dimension):
        # 1. for `l`: `input.dim_tracks[0] is output.dim_tracks[0]`                # both nhd, equality
        # 2. for `m`: `input.dim_tracks[2].value_id in output.dim_tracks[2].deps`  # one is hd, depencency
        # 3. for `h`: `input.dim_tracks[1].value_id in output.dim_tracks[2].deps`  # one is hd, depencency
        #             `input.dim_tracks[1] in output.dim_tracks[1].deps`           # one is hd, depencency

        # TODO: We can handle more complex cases in the future if needed.
        # In current version, we don't handle the case like
        # 1. `(2 h) -> (2 h)`: input.dim_tracks[0] should be equal to output.dim_tracks[0]? (2 can be a runtime number, so we cannot be sure)
        # 2. `(l m) -> (l m)`: input.dim_tracks[0] should be equal to output.dim_tracks[0].

        # ivt => identifier_value_track_map
        hidden_ivt: dict[str, list[ValueTrack]] = defaultdict(list)
        non_hidden_ivt: dict[str, list[ValueTrack]] = defaultdict(list)

        for i, input_tensor in enumerate(node.inputs()):
            if not isinstance(input_tensor, IRTensor) or node.ianno(i).ignore:
                continue

            ianno = node.ianno(i)
            for dim, dim_track in zip(ianno.dims, input_tensor.dim_tracks):
                identifiers = [i for i in dim.identifiers if not str.isdecimal(i)]
                if len(identifiers) == 1 and len(dim.identifiers) == 1:
                    # not hidden dimension
                    non_hidden_ivt[identifiers[0]].append(dim_track)
                else:
                    for iden in identifiers:
                        hidden_ivt[iden].append(dim_track)

        for iden, iden_infos in non_hidden_ivt.items():
            # merge all not-hidden-dimension infos together
            first = iden_infos[0]
            for info in iden_infos[1:]:
                self._add_equiv_value(first.value_id, info.value_id)

        for i, output_tensor in enumerate(node.outputs()):
            if not isinstance(output_tensor, IRTensor) or node.oanno(i).ignore:
                continue

            oanno = node.oanno(i)
            for dim, dim_track in zip(oanno.dims, output_tensor.dim_tracks):
                # find the first identifier that is not a number
                identifiers = [i for i in dim.identifiers if not str.isdecimal(i)]
                if len(identifiers) == 1 and len(dim.identifiers) == 1:
                    ident = identifiers[0]
                    if ident in non_hidden_ivt:
                        first = non_hidden_ivt[ident][0]
                        self._add_equiv_value(first.value_id, dim_track.value_id)
                    else:
                        # this identifier is used together with other identifiers
                        # so it is just a dependency.
                        dim_track.deps = dim_track.deps or []
                        dim_track.deps.extend(v.value_id for v in hidden_ivt[ident])
                        dim_track.deps = list(set(dim_track.deps))  # deduplicate
                else:
                    dim_track.deps = dim_track.deps or []
                    for ident in identifiers:
                        if ident in hidden_ivt:
                            dim_track.deps.extend(v.value_id for v in hidden_ivt[ident])
                        if ident in non_hidden_ivt:
                            first = non_hidden_ivt[ident][0]
                            dim_track.deps.append(first.value_id)

    def _add_equiv_value(self, value_id, other_value_id):
        self._equiv_value_ids[value_id].update(self._equiv_value_ids[other_value_id])
        for vid in self._equiv_value_ids[other_value_id]:
            self._equiv_value_ids[vid] = self._equiv_value_ids[value_id]
