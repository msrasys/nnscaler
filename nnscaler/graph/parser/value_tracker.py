#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from collections import defaultdict
from typing import Any
from nnscaler.graph.function.dimops import IRDimops
from nnscaler.ir.cten import IR, IRObject, IRTensor, ValueTrack
from nnscaler.ir.operator import IRFwOperation


class ValueTracker:
    def __init__(self):
        # value_id -> ValueTrack
        # Please note some ValueTracks may be merged together (from annotation)
        # So the key can be different from the id of the ValueTrack
        self._vtm: dict[int, ValueTrack] = {}
        self._equiv_value_ids: dict[int, set] = {}

    def track_values(self, objs: list[Any]):
        for obj in objs:
            self.track_value(obj)

    def track_value(self, obj: Any):
        for item in IR.get_objects(obj):
            if isinstance(item, IRTensor):
                for dt in item.dim_tracks:
                    self._vtm[dt.value_id] = dt
            elif isinstance(item, IRObject):
                self._vtm[item.value_track.value_id] = item.value_track

    def _update_track_value(self, obj: Any):
        if isinstance(obj, IRTensor):
            new_dim_tracks = []
            for dt in obj.dim_tracks:
                new_dim_tracks.append(self._vtm[dt.value_id])
            obj.dim_tracks = new_dim_tracks
        elif isinstance(obj, IRObject):
            obj.value_track = self._vtm[obj.value_track.value_id]

    def track_nodes(self, nodes: list[IRFwOperation]):
        """
        Track the value tracks of the input and output objects in the given nodes.
        Here we assume the nodes are topologically sorted.
        """
        # collect all value tracks from nodes
        for node in nodes:
            for obj in node.iobjs():
                self.track_value(obj)
            for obj in node.oobjs():
                self.track_value(obj)

        # init equivalence classes
        for vt in self._vtm.values():
            self._equiv_value_ids[vt.value_id] = {vt.value_id}

        # collect extra value tracks from dimops
        for node in nodes:
            if isinstance(node, IRDimops):
                self._track_dims(node)

        # merge equivalent value tracks together
        for value_id, equiv_ids in self._equiv_value_ids.items():
            min_value_id = min(equiv_ids)
            if value_id != min_value_id:
                continue

            # use the smallest id as the representative
            rep_one = self._vtm[min_value_id]
            for vid in equiv_ids:
                if vid == min_value_id:
                    continue
                # TODO: how we merge dependencies?
                # current we take union (Union may be too strict)
                if rep_one.deps is None:
                    rep_one.deps = self._vtm[vid].deps
                elif self._vtm[vid].deps is not None:
                    rep_one.deps = list(set(rep_one.deps).union(set(self._vtm[vid].deps)))
                self._vtm[vid] = rep_one

        # dedup dependencies
        # Here we will replace dependencies with their representative value tracks
        # which can introduce some duplicates
        for vt in self._vtm.values():
            if vt.deps is not None:
                vt.deps = list(set(self._vtm[d].value_id for d in vt.deps))

        # propagate the merged value tracks back to nodes
        for node in nodes:
            for obj in node.iobjs():
                self._update_track_value(obj)
            for obj in node.oobjs():
                self._update_track_value(obj)

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
