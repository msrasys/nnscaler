"""
IRGraph:
    a graph that is composed by node (IRFwOperation) and edge (IRTensor).

    Note the device of graph.inputs() can be different of the same input
    tensor of operation node in the graph. In this case, a move operation
    will be inserted at scheduling time.
"""

from typing import Any, Union, Tuple, List, Optional, Dict
import copy

from cube.ir.cten import IRTensor, IRCell
from cube.ir.operator import IRBpOperation, IRFwOperation, IRDataOperation
from cube.ir.adapter import IRAdapter
from cube.ir.tensor import IRFullTensor, IRSubTensor

from cube.algorithm.generics import GenericDistAlgo


class IRSegment(IRCell):
    """
    A distributed sub-graph representing a piece of workload in parent IRGraph
    """

    def __init__(self, nodes: List[IRCell], inputs: List[IRSubTensor], outputs: List[IRSubTensor]):
        super().__init__('segment', '', len(inputs), len(outputs), init_outputs=False)

        self._nodes = nodes
        self._idevice = [t.device for t in inputs]
        self._odevice = [t.device for t in outputs]

        for idx, val in enumerate(inputs):
            self.set_input(idx, val)
        for idx, val in enumerate(outputs):
            self.set_output(idx, val)
        # setup device
        device = set()
        for node in nodes:
            device.update(node.device)
        self.device = list(device)
        # setup whether forward
        fnodes = any(isinstance(n, IRFwOperation) for n in nodes)
        bnodes = any(isinstance(n, IRBpOperation) for n in nodes)
        assert not (fnodes and bnodes), "An IRSegment cannot have both forward nodes and backward nodes"
        self._forward = fnodes

    @property
    def forward(self) -> bool:
        return self._forward

    def nodes(self, idx: Optional[int] = None) -> Union[IRCell, List[IRCell]]:
        if isinstance(idx, int):
            return self._nodes[idx]
        else:
            return copy.copy(self._nodes)

    def dispatch(self, devid: int, for_mirror=True) -> Optional[IRCell]:
        """
        Instantiate from distributed representation to a
        device-specific sub-graph.
        
        The mirror will also be dispatched if it is not None.

        Return the dispatched segment
        """
        if devid not in self.device:
            return None
        if len(self.device) == 1 and self.device == [devid]:
            return self
        itensors = [t for t, device in zip(self.inputs(), self._idevice) if devid in device]
        otensors = [t for t, device in zip(self.outputs(), self._odevice) if devid in device]
        nodes = [n for n in self.nodes() if devid in n.device]
        for idx, adapter in enumerate(nodes):
            if isinstance(adapter, IRAdapter):
                nodes[idx] = adapter.dispatch(devid)
        fseg = IRSegment(nodes, itensors, otensors)
        fseg._id = self._id
        # dispatch for mirror
        if for_mirror and isinstance(self.mirror, IRSegment):
            bseg = self.mirror.dispatch(devid, for_mirror=False)
            IRCell.make_pair(fseg, bseg)
        return fseg

    def __repr__(self):
        name = ('f' if self.forward else 'b') + 'Segment'
        return f'{name}{self._id}-{self.device}(inputs={self.inputs()}, outputs={self.outputs()})'

    def extra_repr(self) -> str:
        dscp = repr(self)
        for node in self.nodes():
            dscp += '\n\t' + repr(node)
        return dscp


class IRGraph(IRCell):
    """
    IR Graph. The hyperGraph for representing distributed
    graph.
    """

    def __init__(self, 
                 nodes: List[IRCell],
                 inputs: Optional[List[IRTensor]], 
                 outputs: Optional[List[IRTensor]], 
                 module_name: str):

        self._nodes: List[IRCell] = list()
        self._parameters = list()
        self._full_tensors: Dict[int, IRFullTensor] = dict()

        self._schedule_strategy = None

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

        # set parameters and full tensors
        for node in nodes:
            for tensor in node.inputs() + node.outputs():
                if isinstance(tensor, IRSubTensor):
                    pid = tensor.parent._id
                    self._full_tensors[pid] = tensor.parent
                    if tensor.is_param():
                        self._parameters.append(input)

        for ftensor in self._full_tensors.values():
            ftensor.clear_producer_consumer()

        # insert node from nodes
        for idx, node in enumerate(nodes):
            self.attach(node, idx)

        self.reset_dependency()

    @property
    def schedule_plan(self) -> Optional[Any]:
        return self._schedule_strategy

    @schedule_plan.setter
    def schedule_plan(self, val: Optional[Any]):
        self._schedule_strategy = val

    def reset_dependency(self):
        """
        Reset the node dataflow dependency

        Note all the predefined control dependencies will be removed.
        """
        for node in self._nodes:
            node.clear_predecessor()
            node.clear_successor()
        # TODO: adapter dependency not set
        for ftensor in self._full_tensors.values():
            for ptensor, producer in zip(ftensor.ptensors, ftensor.producers):
                for ctensor, consumer in zip(ftensor.ctensors, ftensor.consumers):
                    if ptensor.overlap(ctensor):
                        pidx = producer.outputs().index(ptensor)
                        cidx = consumer.inputs().index(ctensor)
                        producer.add_successor(pidx, consumer)
                        consumer.add_predecessor(cidx, producer)
                # set mirror as control dependency
                if producer.mirror and isinstance(producer, IRFwOperation):
                    producer.add_successor(-1, producer)
                    producer.mirror.add_predecessor(-1, producer)

    def parameters(self):
        """
        Return parameter list
        """
        return copy.copy(self._parameters)

    def full_tensors(self):
        """
        Return full tensor list
        """
        return list(self._full_tensors.values())

    def nodes(self, index: Optional[int] = None) -> Union[IRCell, List[IRCell]]:
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

    def segment(self, nodes: List[IRCell]) -> IRSegment:
        """
        Create a segment (sub-graph) with part of the nodes.

        Return:
            IRSegment
        """
        inputs, outputs = [], []
        for node in nodes:
            assert not isinstance(node, IRSegment), 'A segment cannot be in other segments'
            # update inputs
            itensors = [t for t in node.inputs() if isinstance(t, IRSubTensor)]
            for itensor in itensors:
                producers = [p for p in itensor.parent.producers if set(p.device).issubset(set(node.device))]
                # no producer means a weight or cross device-group
                if len(producers) == 0 or any(p not in nodes for p in producers):
                    # FIXME: itensor should also consider device difference
                    if itensor not in inputs:
                        inputs.append(itensor)
            # update outputs
            otensors = [t for t in node.outputs() if isinstance(t, IRSubTensor)]
            for otensor in otensors:
                if otensor in self.outputs():
                    outputs.append(otensor)
                consumers = [c for c in otensor.parent.consumers if set(c.device).issubset(set(node.device))]
                # no consumer usually means the loss or cross device-group
                if len(consumers) == 0 or any(c not in nodes for c in consumers):
                    # FIXME: otensor should also consider device difference
                    if otensor not in outputs:
                        outputs.append(otensor)
        segment = IRSegment(nodes, inputs, outputs)
        return segment

    def group(self, nodes: List[IRCell]) -> IRSegment:
        """
        Group consecutive nodes into IRSegment.

        Currently this interface will break the dependency,
        it can only be used after user policy
        """
        allnodes = self.nodes()
        indices = [allnodes.index(n) for n in nodes]
        minidx, maxidx = min(indices), max(indices)
        assert maxidx - minidx + 1 == len(nodes), "nodes are not consecutive"
        segment = self.segment(nodes)
        self._nodes = allnodes[:minidx] + [segment] + allnodes[maxidx+1:]
        # FIXME: set segment dependnecy
        return segment

    def detach(self, node: IRCell, reset_dependency=False) -> int:
        """
        Detach (remove) a node from current graph.

        All the used input and output tensors inside the node
        are removed from consumed and produced tensor list.

        Return:
            index (int): index of the detached node in the graph
        """
        if node not in self.nodes():
            raise KeyError(f"node {node} is not in graph.")
        index = self._nodes.index(node)
        self._nodes.pop(index)
        if isinstance(node, IRAdapter):
            return index
        # update consumer
        itensors: List[IRSubTensor] = []
        for itensor in node.inputs():
            if isinstance(itensor, IRSubTensor) and itensor not in itensors:
                itensors.append(itensor)
        for itensor in itensors:
            itensor.parent.rm_consumer(node)
        # update producer
        otensors: List[IRSubTensor] = []
        for otensor in node.outputs():
            if isinstance(otensor, IRSubTensor) and otensor not in otensors:
                otensors.append(otensor)
        for otensor in otensors:
            otensor.parent.rm_producer(node)
        if reset_dependency:
            self.reset_dependency()
        return index

    def attach(self, node: IRCell, index, reset_dependency=False):
        """
        Attach (insert) a node into current graph at node index.

        All the used input and output tensors inside the node are 
        recorded in consumed and produced tensor list. Adapter node
        will not record the consumer and producer.
        """
        if node in self.nodes():
            raise KeyError(f"node {node} is already in graph.")
        self._nodes.insert(index, node)
        if isinstance(node, IRAdapter):
            return
        # update consumer
        itensors = []
        for itensor in node.inputs():
            if isinstance(itensor, IRSubTensor) and itensor not in itensors:
                itensors.append(itensor)
        for itensor in itensors:
            idx = 0
            for consumer in itensor.parent.consumers:
                if self.nodes().index(consumer) < index:
                    idx += 1
                else:
                    break
            itensor.parent.add_consumer(node, itensor, idx)
        # update producer
        otensors = []
        for otensor in node.outputs():
            if isinstance(otensor, IRSubTensor) and otensor not in otensors:
                otensors.append(otensor)
        for otensor in otensors:
            idx = 0
            for producer in otensor.parent.producers:
                if self.nodes().index(producer) < index:
                    idx += 1
                else:
                    break
            otensor.parent.add_producer(node, otensor, idx)
        if reset_dependency:
            self.reset_dependency()
        return

    def flatten(self) -> List[IRCell]:
        """
        Flattent the graph by expanding nodes
        """
        nodes = []
        for node in self.nodes():
            if isinstance(node, IRSegment):
                nodes += node.nodes()
            else:
                nodes.append(node)
        return nodes

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

    def replicate(self, op: IRCell, times=1, reset_dependency=True) -> Optional[List[IRCell]]:
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
    
        fnodes = [op.replicate() for _ in range(times - 1)]
        # insert forward
        fidx = self.nodes().index(op)
        for idx, fnode in enumerate(fnodes):
            self.attach(fnode, fidx + idx + 1) 
        # insert backward
        if isinstance(op.mirror, IRBpOperation):
            for fnode in fnodes:
                fnode.gen_backward()
            bnodes = [fnode.mirror for fnode in fnodes][::-1]
            bidx = self.nodes().index(op.mirror)
            for idx, bnode in enumerate(bnodes):
                self.attach(bnode, bidx + idx)
        if reset_dependency:
            self.reset_dependency()
        return [op] + fnodes

    def partition(self, op: IRCell, algo: GenericDistAlgo, **config) -> Optional[List[IRCell]]:
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
        if not (isinstance(op, IRFwOperation) or isinstance(op, IRDataOperation)):
            raise ValueError("Only allow op to be forward op or data op.")

        if algo.node != op:
            return None
        if not algo.satisfy(**config):
            return None
        fnodes = algo.instantiate(**config)

        #FIXME: we don't allow non-weight input to be splitted in value
        for fnode in fnodes:
            for input in fnode.inputs():
                if isinstance(input, IRSubTensor):
                    if input.valmap.chunk_num != 1 and not input.is_param():
                        raise NotImplementedError(
                            f"Not support feature-map {input} to be splitted in value as input"
                        )
        # update forward
        findex = self.detach(op)
        for idx, fnode in enumerate(fnodes):
            self.attach(fnode, findex + idx)
        # update backward
        if isinstance(op.mirror, IRBpOperation):
            bindex = self.detach(op.mirror)
            bnodes = [fnode.gen_backward() for fnode in fnodes][::-1]
            for idx, bnode in enumerate(bnodes):
                self.attach(bnode, bindex + idx)
        # update gradient
        updated = set()
        for input in op.inputs():
            if not isinstance(input, IRSubTensor):
                continue
            for fnode in input.parent.consumers:
                bnode = fnode.mirror
                if isinstance(bnode, IRBpOperation) and fnode._id not in updated:
                    idx = self.detach(bnode)
                    bnode.update()
                    self.attach(bnode, idx)
                updated.add(fnode._id)
        # update device
        for fnode in fnodes:
            fnode.device = op.device
            if isinstance(fnode.mirror, IRCell):
                fnode.mirror.device = op.device
        self.reset_dependency()
        return fnodes

    def merge(self, nodes: List[IRCell], target_node: IRCell):
        """
        Merge consecutive nodes in the graph to the target_node.
        Note corresponding mirror nodes (if have) will also be merged.

        We don't check computation equivalence between nodes and target_node.

        Merge requires nodes are consecutive in the graph sequence.
        """
        if not isinstance(target_node, IRCell):
            raise TypeError("Expected target node to be IRCell")
        if target_node in self.nodes():
            raise ValueError("Target node is already in the graph")
        for node in nodes:
            if node not in self.nodes():
                raise KeyError(f"node {node} is not in the graph")
        indices = [self.nodes().index(node) for node in nodes]
        # consecutive
        if max(indices) - min(indices) != len(indices) - 1:
            return False
        index = min(indices)
        # update forward
        for node in nodes:
            self.detach(node)
        self.attach(target_node, index)
        # update backward
        if all([isinstance(node.mirror, IRCell) for node in nodes]):
            bidx = len(self.nodes())
            for node in nodes:
                idx = self.detach(node.mirror)
                bidx = min(idx, bidx)
            if target_node.mirror is None:
                if not isinstance(target_node, IRFwOperation):
                    raise RuntimeError("target node is not FwOp and doens't have mirror node")
                target_node.gen_backward()
            self.attach(target_node.mirror, bidx)
        elif all([isinstance(node.mirror, None) for node in nodes]):
            pass
        else:
            raise ValueError("nodes should have nothing-or-all mirror nodes")
        # update weights
        updated = set()
        for node in nodes + [target_node]:
            for input in node.inputs():
                if not isinstance(input, IRSubTensor):
                    continue
                for fnode in input.parent.consumers:
                    bnode = fnode.mirror
                    if isinstance(bnode, IRBpOperation) and fnode._id not in updated:
                        idx = self.detach(bnode)
                        bnode.update()
                        self.attach(bnode, idx)
                    updated.add(fnode._id)
        return True

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

    def add_schedule(self, nodes: List[IRCell]) -> bool:
        """
        Add node happen before dependencies according to nodes list order
        """
        if not all([isinstance(node, IRCell) for node in nodes]):
            raise TypeError("Expected List[IRCell")
        for idx in range(len(nodes) - 1):
            prev = nodes[idx]
            post = nodes[idx + 1]
            if self.happen_before(post, prev):
                return False
        for idx in range(len(nodes) - 1):
            prev = nodes[idx]
            post = nodes[idx + 1]
            prev.add_successor(output_index=-1, cell=post)
            post.add_predecessor(input_index=-1, cell=prev)
        return True

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
                if len(pre_indices) == 0:
                    index = 0
                else:
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
        #TODO: check no new operators are created (including replicate)
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
            succ_node_ids = [node._id for node in node.successors()]
            # succ_node_ids = [None] * len(node.outputs())
            # for out_idx in range(len(node.outputs())):
            #     node_list = [snode._id for snode in node.successors(out_idx)]
            #     succ_node_ids[out_idx] = node_list
            dscp += f"\n{node._id}: {node} -> node id {succ_node_ids}"
        # outputs
        dscp += f"\nOutputs: {self.outputs()}\n{'=' * len(self.name)}\n"
        return dscp

    def module_repr(self):
        return repr(self)


