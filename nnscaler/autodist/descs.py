#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import copy
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from nnscaler.ir.operator import IRFwOperation


@dataclass
class NodePartitionDesc:
    # list element: ((idx, dim), num), the order matters
    desc: List[Tuple[Tuple[int, int], int]]


@dataclass(frozen=True)
class FixedPartitionDesc:
    """A user-specified, ordered partition plan for matching operators."""

    name: str
    parent_module: str
    desc: Tuple[Tuple[Tuple[int, int], int], ...]

    @staticmethod
    def from_config(content: Dict[str, Any]) -> 'FixedPartitionDesc':
        if not isinstance(content, dict):
            raise ValueError(
                f'fixed partition description must be a dict, got {type(content)}'
            )

        expected_keys = {'name', 'parent_module', 'desc'}
        if set(content) != expected_keys:
            raise ValueError(
                'fixed partition description must contain exactly '
                f'{sorted(expected_keys)}, got {sorted(content)}'
            )

        name = content['name']
        parent_module = content['parent_module']
        raw_desc = content['desc']
        if not isinstance(name, str) or not name:
            raise ValueError(f'fixed partition name must be a non-empty str, got {name!r}')
        if not isinstance(parent_module, str):
            raise ValueError(
                f'fixed partition parent_module must be a str, got {parent_module!r}'
            )
        if not isinstance(raw_desc, list) or not raw_desc:
            raise ValueError('fixed partition desc must be a non-empty list')

        desc = []
        step_keys = {'input', 'dim', 'num'}
        for step in raw_desc:
            if not isinstance(step, dict) or set(step) != step_keys:
                got = sorted(step) if isinstance(step, dict) else type(step)
                raise ValueError(
                    'each fixed partition step must contain exactly '
                    f'{sorted(step_keys)}, got {got}'
                )
            input_idx, dim, num = step['input'], step['dim'], step['num']
            if any(not isinstance(v, int) or isinstance(v, bool)
                   for v in (input_idx, dim, num)):
                raise ValueError(
                    f'fixed partition step values must be ints, got {step}'
                )
            if (input_idx, dim) != (-1, -1) and (input_idx < 0 or dim < 0):
                raise ValueError(
                    'fixed partition input and dim must both be -1 for replication, '
                    f'or both be non-negative, got {(input_idx, dim)}'
                )
            if num < 1:
                raise ValueError(f'fixed partition num must be positive, got {num}')
            desc.append(((input_idx, dim), num))

        return FixedPartitionDesc(name, parent_module, tuple(desc))


def select_fixed_partition_desc(
    name: str,
    module_types: Sequence[str],
    fixed_descs: Iterable[FixedPartitionDesc],
) -> Tuple[Optional[FixedPartitionDesc], Set[FixedPartitionDesc]]:
    """Select the closest eligible fixed rule and return all matched rules."""
    candidates = []
    matched = set()
    module_types = tuple(module_types)
    for fixed_desc in fixed_descs:
        if fixed_desc.name != name:
            continue
        if not fixed_desc.parent_module:
            # An empty parent is a global fallback for this exact signature.
            candidates.append((-1, 0, fixed_desc))
            matched.add(fixed_desc)
            continue
        parent_types = tuple(fixed_desc.parent_module.split('.'))
        matches = [
            start
            for start in range(len(module_types) - len(parent_types) + 1)
            if module_types[start:start + len(parent_types)] == parent_types
        ]
        if matches:
            matched.add(fixed_desc)
            # Prefer a match ending closest to the innermost module, then the
            # longer (more specific) class chain.
            end = matches[-1] + len(parent_types) - 1
            candidates.append((end, len(parent_types), fixed_desc))
    if not candidates:
        return None, matched
    candidates.sort(key=lambda item: (-item[0], -item[1]))
    return candidates[0][2], matched


@dataclass
class MeshDesc:
    # inter node
    row: int
    # intra node
    col: int

    @property
    def ngpus(self):
        return self.row * self.col

    def to_json(self):
        return (self.row, self.col)

    @staticmethod
    def from_json(val):
        return MeshDesc(*val)


@dataclass
class TensorParallelDesc:
    partition_descs: Dict[int, NodePartitionDesc]
    recompute_groups: List[List[int]]
    mesh_desc: MeshDesc
    analysis: Dict[str, Any]

    def to_json(self, cid2node: Optional[Dict[int, IRFwOperation]] = None):
        ret = {}
        descs_list = []
        for k, v in self.partition_descs.items():
            entry = {'cid': k, 'partition': v.desc}
            if cid2node is not None:
                node = cid2node.get(k)
                if node is not None:
                    entry['fqn'] = node.fqn
                    entry['op'] = node.signature
            descs_list.append(entry)
        ret['partition_descs'] = descs_list
        ret['recompute_groups'] = self.recompute_groups
        ret['mesh_desc'] = self.mesh_desc.to_json()
        ret['analysis'] = self.analysis
        return ret

    @staticmethod
    def from_json(ret):
        partition_descs = {}
        for item in ret['partition_descs']:
            if isinstance(item, dict):
                # new format: {"cid": ..., "partition": ..., "fqn": ..., "op": ...}
                partition_descs[item['cid']] = NodePartitionDesc(item['partition'])
            else:
                # old format: [cid, desc]
                k, v = item
                partition_descs[k] = NodePartitionDesc(v)
        return TensorParallelDesc(partition_descs,
                                  copy.deepcopy(ret['recompute_groups']),
                                  MeshDesc.from_json(ret['mesh_desc']),
                                  ret['analysis'])


@dataclass
class SPMDSearchOutput:
    desc: TensorParallelDesc
    memory: float
    all_time: float
    comp_time: float

    def to_json(self, cid2node: Optional[Dict[int, IRFwOperation]] = None):
        return {
            'desc': self.desc.to_json(cid2node),
            'memory': self.memory,
            'all_time': self.all_time,
            'comp_time': self.comp_time,
        }

    @staticmethod
    def from_json(json_val):
        desc = TensorParallelDesc.from_json(json_val['desc'])
        return SPMDSearchOutput(desc, json_val['memory'], json_val['all_time'],
                                json_val['comp_time'])


@dataclass
class PipelineParallelDesc:
    spmd_descs: List[TensorParallelDesc]
    recompute_groups: List[List[int]]
    mesh_desc: MeshDesc

    def to_json(self, cid2node: Optional[Dict[int, IRFwOperation]] = None):
        return {
            'spmd_descs': [desc.to_json(cid2node=cid2node) for desc in self.spmd_descs],
            'recompute_groups': self.recompute_groups,
            'mesh_desc': self.mesh_desc.to_json(),
        }

    @staticmethod
    def from_json(json_val):
        spmd_descs = []
        for spmd_desc_json in json_val['spmd_descs']:
            spmd_descs.append(TensorParallelDesc.from_json(spmd_desc_json))
        recompute_groups = copy.deepcopy(json_val['recompute_groups'])
        mesh_desc = MeshDesc.from_json(json_val['mesh_desc'])
        return PipelineParallelDesc(spmd_descs, recompute_groups, mesh_desc)


@dataclass
class PipelineSearchOutput:
    desc: PipelineParallelDesc
    e2e_time: float
    stage_mems: List[float]
    stage_all_times: List[float]
    stage_comp_times: List[float]

    def to_json(self, cid2node: Optional[Dict[int, IRFwOperation]] = None):
        return {
            'desc': self.desc.to_json(cid2node=cid2node),
            'e2e_time': self.e2e_time,
            'stage_mems': self.stage_mems,
            'stage_all_times': self.stage_all_times,
            'stage_comp_times': self.stage_comp_times,
        }

    @staticmethod
    def from_json(json_val):
        desc = PipelineParallelDesc.from_json(json_val['desc'])
        return PipelineSearchOutput(desc, json_val['e2e_time'],
                                    json_val['stage_mems'],
                                    json_val['stage_all_times'],
                                    json_val['stage_comp_times'])


@dataclass
class PartitionConstraint:

    # the name of the corresponding operator in the model. It equals
    # to the `signature` field in the `IRFwOperation` in cube
    name: str
    # the **closest** father module name of the operator
    parent_module: str
    # a list of allowed partition dimensions of input tensors
    allowed_partition_dims: List[Tuple[int, int]]
    replica_allowed: bool = True

    @staticmethod
    def from_json(content: Dict[str, Any]):
        allowed_partition_dims = [
            tuple(x) for x in content['allowed_partition_dims']
        ]
        return PartitionConstraint(content['name'], content['parent_module'],
                                   allowed_partition_dims,
                                   content['replica_allowed'])

    def to_json(self):
        return {
            'name': self.name,
            'parent_module': self.parent_module,
            'allowed_partition_dims': self.allowed_partition_dims,
            'replica_allowed': self.replica_allowed,
        }

    @staticmethod
    def from_yaml(content: Dict[str, Any]):

        def _parse_dims(dims: str) -> List[int]:
            return tuple([int(x) for x in dims.split(',')])

        allowed_partition_dims = [
            _parse_dims(x) for x in content['allowed_partition_dims']
        ]
        return PartitionConstraint(content['name'], content['parent_module'],
                                   allowed_partition_dims,
                                   content['replica_allowed'])

    def to_yaml(self):

        def to_str(dims: List[int]) -> str:
            return ','.join([str(x) for x in dims])

        allowed_partition_dims = [
            to_str(x) for x in self.allowed_partition_dims
        ]
        return {
            'name': self.name,
            'parent_module': self.parent_module,
            'allowed_partition_dims': allowed_partition_dims,
            'replica_allowed': self.replica_allowed,
        }

    def __hash__(self):
        return hash((self.name, self.parent_module,
                     tuple(self.allowed_partition_dims), self.replica_allowed))
