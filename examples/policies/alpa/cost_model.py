"""
Cost model for intra-op plan search
"""
from typing import List, Callable, Tuple, Dict
import numpy as np

from nnscaler.graph import IRGraph
from nnscaler.ir.cten import IRTensor
from nnscaler.ir.operator import IRFwOperation
from nnscaler.graph.function.anchor import IRGraphAnchor
from nnscaler.graph.function.dimops import IRDimops, TransformRule, DimopSplit


DistSpec = Dict[int, Tuple[Tuple[int, int]]]


class CommCost:
    """
    Get communication cost in milliseconds
    """
    @staticmethod
    def get_bandwidth(ranks: List[int]):
        """
        TODO: support with real runtime information
        """
        if len(ranks) < 8:
            return 150 * 1e9 # 150 GB/s for intra-node (NVLink)
        else:
            return 12.5 * 1e9 # 12.5 GB/s for inter-node (IB)

    @staticmethod
    def allreduce_cost(tensor: IRTensor, num_devices: int) -> float:
        bandwidth = CommCost.get_bandwidth(list(range(num_devices))) 
        return 2 * (num_devices - 1) * tensor.byte_size() / num_devices / bandwidth * 1000

    @staticmethod
    def alltoall_cost(tensor: IRTensor, num_devices: int) -> float:
        # bandwidth in all-to-all is really worse (1GB/s) and should not use
        return 1e6
        bandwidth = CommCost.get_bandwidth(list(range(num_devices)))
        return tensor.byte_size() / num_devices / num_devices * (num_devices - 1) / bandwidth * 1000

    @staticmethod
    def allgather_cost(tensor: IRTensor, num_devices: int) -> float:
        # bandwidth in allgather can only be half due to torch implementation issues
        # return 1e6
        bandwidth = CommCost.get_bandwidth(list(range(num_devices))) / 2.98
        return tensor.byte_size() / num_devices * (num_devices - 1) / bandwidth * 1000

    @staticmethod
    def reducescatter_cost(tensor: IRTensor, num_devices: int) -> float:
        # bandwidth in reduce-scatter can only be half due to torch implementation issues
        # return 1e6
        bandwidth = CommCost.get_bandwidth(list(range(num_devices))) / 2.38
        return tensor.byte_size() / num_devices * (num_devices - 1) / bandwidth * 1000


class CostModel:

    def __init__(self, graph: IRGraph, estimator: Callable):

        self.graph = graph
        self.estimator = estimator

        # node property
        self.comp_cost = {}
        self.mem_cost = {}

        self.edges: Dict[int, List[int]] = {}
        for ftensor in graph.full_tensors():
            if ftensor.is_grad(): continue
            for producer in graph.producers(ftensor):
                if not isinstance(producer, IRFwOperation): continue
                for consumer in graph.consumers(ftensor):
                    if not isinstance(consumer, IRFwOperation): continue
                    self.edges.setdefault(producer.cid, []).append(consumer.cid)
        
        # node.cid -> ((idx, dim),)
        self.partition_algos: Dict[int, Tuple[int, int]] = {}

        fnodes = graph.select(ntype=IRFwOperation)
        fnodes = [n for n in fnodes if not (isinstance(n, IRGraphAnchor) or n.name == 'multiref')]

        for fnode in fnodes:
            latency, memory = self.estimator((fnode,))
            self.comp_cost[fnode.cid] = latency
            self.mem_cost[fnode.cid] = memory
            self.partition_algos[fnode.cid] = self.get_transform_space(fnode)

    def get_transform_space(self, node: IRFwOperation) -> List[Tuple[int, int]]:
        """
        Get the transform space of a node
        
        None indicates replicate
        """
        light_op_names = ('add', 'sub', 'mul', 'layernorm')
        # light_op_names = ()
        if isinstance(node, IRDimops):
            params = [t for t in node.inputs() if isinstance(t, IRTensor) and t.is_attr()]
            # must be partitioned for computation-intensive ops
            if len(params) > 0 and node.name not in light_op_names: # not node.signature.startswith('torch.'):
                return list(node.transform_space())
            # can be partitioned or replicated for computation-light ops
            else:
                return [None] + node.transform_space()
        return [None]

    def get_memory_cost(self, fnode: IRFwOperation) -> int:
        if fnode.cid not in self.mem_cost:
            if not (isinstance(fnode, IRGraphAnchor) or fnode.name == 'multiref'):
                print(f'warning: cannot find memory cost for node {fnode.name}({fnode.cid})')
            return 0
        return self.mem_cost[fnode.cid]

    def get_comp_cost(self, fnode: IRFwOperation, num_devices: int) -> np.ndarray:
        """
        Get computation cost related to different partition strategies
        """
        return np.zeros(len(self.partition_algos[fnode.cid]), dtype=float)
        # cost = []
        # original_cost = self.comp_cost[fnode.cid]
        # for strategy in self.partition_algos[fnode.cid]:
        #     if strategy is None:
        #         cost.append(original_cost)
        #     else:
        #         # computation efficiency simulation
        #         efficiency = 1 - (num_devices-1)*0.1/2
        #         cost.append(original_cost / num_devices / efficiency)
        # return np.array(cost, dtype=float)

    def get_comm_cost(self, fnode: IRFwOperation, num_devices) -> np.ndarray:
        """
        Get communication cost for a node given a strategy

        This only calucates the cases for partitioning on value dimension

        @return cost: np.ndarray: 1-D array of the cost on allreduce
        """
        cost = []
        for strategy in self.partition_algos[fnode.cid]:
            if strategy is None:
                cost.append(0.)
                continue
            s_cost = 0
            idx, dim = strategy
            rule: TransformRule = fnode.algorithms('dim').infer(idx, dim, num_devices)
            for idx, output in enumerate(rule.outputs()):
                if output.isV():
                    s_cost += CommCost.allreduce_cost(fnode.output(idx), num_devices)
            cost.append(s_cost)
        return np.array(cost, dtype=float)
    
    def get_pair_reshard_cost(self, fnode_src: IRFwOperation, fnode_dst: IRFwOperation, 
                              num_devices: int) -> np.ndarray:
        """
        Get cost of resharding between two nodes
        @return cost: np.ndarray: 1-D tensor of (nsrc * ndst,) shape,
            nsrc is the number of partitioned ways of the source node
            ndst is the number of partitioned ways of the destination node
        """
        nsrc = len(self.partition_algos[fnode_src.cid])
        ndst = len(self.partition_algos[fnode_dst.cid])
        cost = np.zeros((nsrc, ndst), dtype=float)

        def comm_cost(tensor: IRTensor, num_devices: int,
                      src_split: DimopSplit, dst_split: DimopSplit, dst_replica: bool):
            # note for data parallel, we don't consider allreduce cost as it
            # will only be performed at the last of iteration.
            if tensor.is_attr(): return 0.0
            if src_split.isV() or src_split.isR():
                # identity-allreduce or identity-identity
                if dst_split.isR():
                    return 0.0 if dst_replica else CommCost.allreduce_cost(tensor, num_devices)
                # split-allgather
                if dst_split.isD():
                    return CommCost.allgather_cost(tensor, num_devices)
            if src_split.isD():
                # allgahter-reducescatter or allgather-split
                if dst_split.isR():
                    return CommCost.allgather_cost(tensor, num_devices) if dst_replica else \
                           CommCost.allgather_cost(tensor, num_devices) + CommCost.reducescatter_cost(tensor, num_devices)
                # all2all-all2all or identity-identity
                if dst_split.isD():
                    return 0.0 if src_split == dst_split else 2 * CommCost.alltoall_cost(tensor, num_devices)
            raise NotImplementedError(f"Unknown split type: {src_split} -> {dst_split}")

        # FIXME: need consider cases that an operator has multiple **same** inputs
        tensors: Dict[IRTensor, Tuple[int, int]] = {}
        for idx, output in enumerate(fnode_src.outputs()):
            tensors[output.parent] = [idx]
        for idx, input in enumerate(fnode_dst.inputs()):
            if not isinstance(input, IRTensor): continue
            tensors.setdefault(input.parent, []).append(idx)
        tensors = {t: tuple(v) for t, v in tensors.items() if len(v) == 2}

        for i, strategy_src in enumerate(self.partition_algos[fnode_src.cid]):

            rule_src = None
            if strategy_src is not None:
                idx, dim = strategy_src
                rule_src = fnode_src.algorithms('dim').infer(idx, dim, num_devices)
            
            for j, strategy_dst in enumerate(self.partition_algos[fnode_dst.cid]):
                rule_dst = None
                if strategy_dst is not None:
                    idx, dim = strategy_dst
                    rule_dst = fnode_dst.algorithms('dim').infer(idx, dim, num_devices)

                for tensor, (idx_src, idx_dst) in tensors.items():
                    cost[i, j] += comm_cost(
                        tensor, num_devices, 
                        rule_src.outputs()[idx_src] if rule_src is not None else DimopSplit(r=True),
                        rule_dst.inputs()[idx_dst] if rule_dst is not None else DimopSplit(r=True),
                        strategy_dst is None
                    )
        return cost

    def get_edges(self, nodes: List[IRFwOperation]) -> Dict[IRFwOperation, Tuple[IRFwOperation]]:
        """
        Get edges of a subgraph
        """
        edges: Dict[IRFwOperation, List[IRFwOperation]] = {}
        cid2nodes: Dict[int, IRFwOperation] = {n.cid : n for n in nodes}
        for node in nodes:
            if node.cid in self.edges:
                edges[node] = [cid2nodes[cid] for cid in self.edges[node.cid] if cid in cid2nodes]
        return edges
