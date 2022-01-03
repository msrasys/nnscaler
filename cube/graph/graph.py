"""
IRGraph:
    a graph that is composed by node (IRFwOperation) and edge (IRTensor).

    Note the device of graph.inputs() can be different of the same input
    tensor of operation node in the graph. In this case, a move operation
    will be inserted at scheduling time.
"""

from typing import Union, Tuple, List, Optional, Dict
import copy

from cube.ir.cten import IRTensor, IRCell
from cube.graph.operator.operator import IRBpOperation, IRFwOperation, IRDataOperation
from cube.graph.adapter.adapter import IRAdapter
from cube.graph.tensor import IRSubTensor

from cube.algorithm.generics import GenericDistAlgo


__all__ = ['IRGraph']


class IRGraph(IRCell):
    """
    PyTorch IR Graph

    The IR Graph only contains forward graph
    """

    def __init__(self, 
                 nodes: List[IRCell],
                 inputs: Optional[List[IRTensor]], 
                 outputs: Optional[List[IRTensor]], 
                 module_name: str):
        
        self._nodes: List[IRCell] = nodes
        self._parameters = list()

        if inputs is None:
            inputs = IRGraph.get_inputs(nodes)
        if outputs is None:
            outputs = IRGraph.get_outputs(nodes)

        super().__init__(
            name=module_name,
            signature=module_name,
            input_length=len(inputs),
            output_length=len(outputs)
        )

        for idx, tensor in enumerate(inputs):
            self.set_input(idx, tensor)
        for idx, tensor in enumerate(outputs):
            self.set_output(idx, tensor)

        # set parameter
        for node in self._nodes:
            for input in node.inputs():
                if isinstance(input, IRTensor) and input.is_param():
                    self._parameters.append(input)
        self.reset_dependency()

    def reset_dependency(self):
        """
        Reset the node dataflow dependency
        """
        for node in self._nodes:
            node.clear_predecessor()
            node.clear_successor()
        # set node predecessors and successors
        for src_idx in range(len(self._nodes)):
            src_node = self._nodes[src_idx]
            for dst_node in self._nodes[src_idx+1:]:
                # we don't consider dependencies among adapter
                if isinstance(src_node, IRAdapter) and isinstance(dst_node, IRAdapter):
                    continue
                for out_idx, out_tensor in enumerate(src_node.outputs()):
                    if not isinstance(out_tensor, IRTensor):
                        continue
                    for in_idx, in_tensor in enumerate(dst_node.inputs()):
                        if not isinstance(in_tensor, IRTensor):
                            continue
                        if out_tensor.overlap(in_tensor):
                            src_node.add_successor(out_idx, dst_node)
                            dst_node.add_predecessor(in_idx, src_node)
        # set mirror as control dependency
        for idx1, node1 in enumerate(self._nodes):
            node2 = node1.mirror
            if isinstance(node2, IRCell) and node2 in self._nodes:
                idx2 = self._nodes.index(node2)
                if idx1 < idx2:
                    node1.add_successor(-1, node2)
                    node2.add_predecessor(-1, node1)

    def parameters(self):
        """
        Return parameter list
        """
        return copy.copy(self._parameters)

    def nodes(self, index: Optional[int] = None):
        """
        Get node at position index
        """
        if isinstance(index, int):
            if index >= len(self._nodes):
                raise RuntimeError(
                    f"Get node out of range ({index} >= {len(self._nodes)})"
                )
            return self._nodes[index]
        elif index is None:
            return copy.copy(self._nodes)
        else:
            raise TypeError("Expected index to be None or int")

    def forward(self, *args) -> Union[IRTensor, Tuple[IRTensor]]:
        """
        forward will divide the graph into Actions according to
        node device assignment

        Currently each forward call will result in a new flow
        even if the input is same

        Returns:
            IRTensors
        """
        from cube.logics.translator import LogicTranslator
        return LogicTranslator.forward(self, *args)

    def __call__(self, *args):
        """
        Register forward action
        """
        return self.forward(*args)

    def subgraph(self, sub_nodes: List[IRCell]):
        """
        Create a subgraph with sub nodes.

        Return:
            IRGraph
        """
        sub_inputs = list()
        sub_outputs = list()
        for node in sub_nodes:
            sub_inputs += node.inputs()
            sub_outputs += node.outputs()
        remain_inputs = list()
        remain_outputs = list()
        for node in self.nodes():
            if node in sub_nodes:
                continue
            remain_inputs += node.inputs()
            remain_outputs += node.outputs()
        inputs = list()
        outputs = list()
        for t in sub_inputs:
            if isinstance(t, IRSubTensor) and t not in sub_outputs:
                if t not in inputs:
                    inputs.append(t)
        for t in sub_outputs:
            if isinstance(t, IRSubTensor):
                # not consumed or used outside this subgraph
                if t not in sub_inputs or t in remain_inputs:
                    if t not in outputs:
                        outputs.append(t)
        subgraph = IRGraph(
            nodes = sub_nodes,
            inputs = inputs,
            outputs = outputs,
            module_name = 'segment'
        )
        return subgraph

    @staticmethod
    def get_inputs(nodes: List[IRCell]):
        """
        Get all the input tensors the is not generated by nodes

        Inputs

        Returns:
            List[IRTensor]
        """
        all_outputs = list()
        for node in nodes:
            all_outputs += node.outputs()
        inputs = list()
        for cell in nodes:
            for input in cell.inputs():
                if isinstance(input, IRTensor):
                    if input not in all_outputs:
                        if input not in inputs:
                            inputs.append(input)
        return inputs

    @staticmethod
    def get_outputs(nodes: List[IRCell]):
        """
        Get all the output tensors the is not used by nodes

        Args:
            This will also consider the successor forward nodes.
            If it is required by other outside forward nodes,
            put in the outputs list

        Returns:
            List[IRTensor]
        """
        all_inputs = list()
        for node in nodes:
            all_inputs += node.inputs()
        outputs = list()
        for node in nodes:
            for idx, output in enumerate(node.outputs()):
                # not consumed tensor
                if isinstance(output, IRSubTensor):
                    if output not in all_inputs:
                        if output not in outputs:
                            outputs.append(output)
                            continue
                # consumed by other nodes
                succs = node.successors(idx)
                fsuccs = [
                    fnode for fnode in succs if isinstance(fnode, IRFwOperation)
                ]
                for fsucc in fsuccs:
                    if fsucc not in nodes:
                        if output not in outputs:
                            outputs.append(output)
        return outputs

    ## Parallel Policy Primitives ##

    def replicate(self, op: IRCell, times=1) -> Optional[List[IRCell]]:
        """
        Replicate a forward or data operation multiple times.

        The backward of the forward operation will automatically be replicated.
        """
        if not (isinstance(op, IRFwOperation) or isinstance(op, IRDataOperation)):
            raise TypeError("Expected op to be forward op or data op")
        if not isinstance(times, int) or times < 1:
            raise TypeError("Expected times to be int and >= 1")

        if op not in self.nodes():
            raise RuntimeError(f"Op {op} not exsits")
    
        ops = [op]
        for _ in range(times - 1):
            ops.append(op.replicate())
        if isinstance(op.mirror, IRBpOperation):
            for rep_op in ops[1:]:
                print(rep_op)
                rep_op.gen_backward()
        idx = self.nodes().index(op)
        # forward
        self._nodes = self._nodes[:idx] + ops + self._nodes[idx+1:]
        # backward
        if isinstance(op.mirror, IRCell):
            bops = [op.mirror for op in ops][::-1]
            midx = self.nodes().index(op.mirror)
            self._nodes = self._nodes[:midx] + bops + self._nodes[midx+1:]
        self.reset_dependency()
        return ops

    def partition(self, op: IRCell, algo: GenericDistAlgo, config: Dict) -> Optional[List[IRCell]]:
        """
        Partition an operator (op) by using
        op partition algorithm (algo) and its configuration (config).
        Note the backward op-partition will be automatically done.

        Args:
            op: cell to be partitioned
            algo: generic distributed algorithm related to the op
            config: dict

        Returns:
            nodes: List[IRCell] if partitioned successfully.
            None if failed
        """
        if not isinstance(algo, GenericDistAlgo):
            raise TypeError("Expected algo to be GenericDistAlgo")
        if op not in self.nodes():
            raise RuntimeError(f"Not Exist: {op}")

        if algo.logic_op != type(op):
            return None
        if not algo.satisfy(config):
            return None
        fnodes = algo.instantiate(op, config)

        #FIXME: we don't allow non-weight input to be splitted in value
        for fnode in fnodes:
            for input in fnode.inputs():
                if isinstance(input, IRSubTensor):
                    if input.valmap.chunk_num != 1 and not input.is_param():
                        raise NotImplementedError(
                            f"Not support feature-map {input} to be splitted in value as input"
                        )

        # remove reference
        finputs = op.inputs()
        op.make_empty()
        if op.mirror is not None:
            op.mirror.make_empty()

        # generate backward
        updated = set()
        for input in finputs:
            if not isinstance(input, IRSubTensor):
                continue
            # go through related consumers and update backward op
            for fnode in input.parent.consumers:
                if isinstance(fnode, IRFwOperation) and fnode._id not in updated:
                    if fnode.mirror is not None:
                        fnode.mirror.update()
                    else:
                        fnode.gen_backward()
                    updated.add(fnode._id)

        # insert nodes
        idx = self._nodes.index(op)
        self._nodes = self._nodes[:idx] + fnodes + self._nodes[idx+1:]
        if op.mirror is not None:
            idx = self._nodes.index(op.mirror)
            bnodes = [node.mirror for node in fnodes][::-1]
            self._nodes = self._nodes[:idx] + bnodes + self._nodes[idx+1:]
        self.reset_dependency()
        return copy.copy(fnodes)

    def merge(self, sub_graph, target_op, op_partition_algorithm):
        raise NotImplementedError

    def identity(self, input_tensor, dst_op):
        raise NotImplementedError

    ## Assign Policy Primitives ##

    def assign(self, op: IRCell, ranks: Union[int, List[int]]):
        """
        Assign an operator (subgraph) to (multiple) rank(s).

        If `ranks` has multiple integer, then the operator will be replicated
        `len(ranks)` times and assigned to given device correspondingly.

        Corresponding backward operators (if have) will also be replicated
        and assigned to the same device with it's forward operator

        Returns:
            True if assigned successfully.
            False if not.
        """
        if op not in self._nodes:
            raise KeyError(f"{op} is not in the graph")
        if isinstance(ranks, int):
            ranks = [ranks]
        if not all([isinstance(rank, int) for rank in ranks]):
            raise TypeError("Expected rank to be int")
        if len(ranks) > 1:
            ops = self.replicate(op, times=len(ranks))
        else:
            ops = [op]
        for op, rank in zip(ops, ranks):
            op.device = rank
            # pytorch requirement: forward + backward happened on same device
            if op.mirror is not None:
                op.mirror.device = rank
        return True

    ## Schedule Policy Primitives ##

    def happen_before(self, node1: IRCell, node2: IRCell, skip=None) -> bool:
        """
        Check node1 -> (happen before) node2

        Returns:
            Boolean
        """
        skip = list() if skip is None else skip
        if node1 in skip:
            return False
        if not isinstance(node1, IRCell) or not isinstance(node2, IRCell):
            raise TypeError("Expected node to be IRCell")
        if node2 in node1.successors():
            return True
        else:
            for succ_node in node1.successors():
                if self.happen_before(succ_node, node2, skip):
                    return True
            return False

    def set_order(self, seq: List[IRCell]):
        """
        Set a topological order for IRGraph, which requires seq:

        1). The set of nodes in seq must be same with this IRGraph
        2). Staisfies topological order

        Returns:
            True if set succesfully, False not.
        """
        for node in seq:
            if node not in self.nodes():
                return False
        if len(seq) != len(self.nodes()):
            return False
        if not IRGraph.check_legal_order(seq, integrity_check=True):
            return False
        self._nodes = seq
        return True

    def partial_set_order(self, seq: List[IRCell], eager=True):
        """
        Set a partial topological order for IRGrah.
        The remaining nodes will be automatically inserted to
        make the full legal sequence.

        In most of the cases, `eager=True` has better performance.

        Args:
            seq: partial scheduling sequence
            eager (default True):
                if True, the remaining nodes are inserted once it is ready
                if Flase, the remaining nodes are inserted only when it is needed.
        
        Returns:
            True if set succesfully, False not.
        """
        seq = copy.copy(seq)
        for node in seq:
            if node not in self.nodes():
                return False
        if not IRGraph.check_legal_order(seq, integrity_check=False):
            return False
        remain: List[IRCell] = [node for node in self.nodes() if node not in seq]
        for node in remain:
            if eager:
                pre_indices = [seq.index(pre) for pre in node.predecessors()]
                index = max(pre_indices) + 1
            else:
                suc_indices = [seq.index[suc] for suc in node.successors()]
                index = min(suc_indices)
            seq.insert(index, node)
        self._nodes = seq
        return True

    @staticmethod
    def check_legal_order(seq: List[IRCell], integrity_check=False):
        """
        Check whether seq satisfies topological order.
        
        Args:
            seq: List of IRCell
            integrity_check:
                If true, performs additional integrity check that requires
                all the SUs in predecessor and successor of a SU should
                appear in the sequence.
        
        Returns:
            Boolean: True for satisfying topo order, otherwise False.
        """
        for index, node in enumerate(seq):
            for pre in node.predecessors():
                if pre in seq:
                    pre_idx = seq.index(pre)
                    if pre_idx >= index:
                        return False
                elif integrity_check:
                    return False
        return True

    def __repr__(self):
        dscp = f"Graph{self._id}-{self.device}(inputs={self.inputs()}, outputs={self.outputs()})"
        return dscp

    def extra_repr(self):
        dscp = f"\n{self.name}:\n{'=' * len(self.name)}\n"
        # inputs
        dscp += f"Inputs: {self.inputs()}\n"
        # nodes
        for node in self._nodes:
            # if isinstance(node, IRBpOperation):
            #     continue
            succ_node_ids = [node._id for node in node.successors()]
            # succ_node_ids = [None] * len(node.outputs())
            # for out_idx in range(len(node.outputs())):
            #     node_list = [snode._id for snode in node.successors(out_idx)]
            #     succ_node_ids[out_idx] = node_list
            dscp += f"\n{node._id}: {node} -> node id {succ_node_ids}\n"
        # outputs
        dscp += f"\nOutputs: {self.outputs()}\n{'=' * len(self.name)}\n"
        return dscp

    def module_repr(self):
        return repr(self)