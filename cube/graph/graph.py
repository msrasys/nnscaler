"""
IRGraph:
    a graph that is composed by node (IRFwOperation) and edge (IRTensor).

    Note the device of graph.inputs() can be different of the same input
    tensor of operation node in the graph. In this case, a move operation
    will be inserted at scheduling time.
"""

from contextlib import contextmanager
from typing import Union, Tuple, List, Optional, Dict, Set

from cube.ir.cten import IRTensor, IRCell
from cube.ir.unique import IDGenerator
from cube.ir.operator import IRBpOperation, IRFwOperation, IRDataOperation
from cube.ir.adapter import IRAdapter
from cube.ir.tensor import IRFullTensor, IRSubTensor

from cube.graph.function.function import Identity, MultiRef
from cube.graph.segment import IRSegment

from cube.algorithm.generics import GenericDistAlgo


class GraphIndex:

    def __init__(self, gidx: int, sidx: Optional[int]):
        # inner-graph index
        assert isinstance(gidx, int)
        self.gidx = gidx
        # inner-segment index
        assert sidx is None or isinstance(sidx, int)
        self.sidx: Optional[int] = sidx

    def __hash__(self) -> int:
        return hash((self.gidx, self.sidx))

    def __eq__(self, other: object) -> bool:
        assert isinstance(other, GraphIndex), "Cannot compare with non-GraphIndex object"
        return self.gidx == other.gidx and self.sidx == other.gidx
    
    def __lt__(self, other: object) -> bool:
        assert isinstance(other, GraphIndex), "Cannot compare with non-GraphIndex object"
        if self.gidx < other.gidx:
            return True
        if self.gidx > other.gidx:
            return False
        if isinstance(self.sidx, int) and isinstance(other.sidx, int):
            return self.sidx < other.sidx
        if self.sidx is None and isinstance(other.sidx, int):
            return True
        return False
    
    def __le__(self, other: object) -> bool:
        return self < other or self == other

    def __gt__(self, other: object) -> bool:
        return not self <= other

    def __ge__(self, other: object) -> bool:
        return not self < other

    def __sub__(self, offset: int):
        assert isinstance(offset, int)
        if self.sidx is None:
            return GraphIndex(self.gidx - offset, self.sidx)
        else:
            return GraphIndex(self.gidx, self.sidx - offset)

    def __add__(self, offset: int):
        assert isinstance(offset, int)
        if self.sidx is None:
            return GraphIndex(self.gidx + offset, self.sidx)
        else:
            return GraphIndex(self.gidx, self.sidx + offset)

    def tuple(self) -> Tuple[int, Optional[int]]:
        return (self.gidx, self.sidx)


class IRGraph(IRSegment):
    """
    IRGraph.

    IRGraph is used for reprensting a distributed training iteration.
    """

    def __init__(self, 
                 nodes: List[IRCell],
                 inputs: Optional[List[IRTensor]], 
                 outputs: Optional[List[IRTensor]], 
                 module_name: str):

        if inputs is None:
            inputs = IRGraph.get_inputs(nodes)
        if outputs is None:
            outputs = IRGraph.get_outputs(nodes)
        super().__init__([], inputs, outputs, module_name)

        # atrribute tensors
        self._attributes: Set[IRSubTensor] = set()

        self._sched = None  # the schedule strategy

        # set parameters / buffers
        for node in nodes:
            tensors = node.inputs() + node.outputs()
            tensors = [t for t in tensors if isinstance(t, IRSubTensor)]
            for t in tensors:
                t.parent.clear_producer_consumer()
                if t.is_attr():
                    self._attributes.add(t)

        # insert node from nodes
        for idx, node in enumerate(nodes):
            self.insert(node, idx)

        self.reset_dependency()

    @property
    def train(self) -> bool:
        """!
        Train flag.

        @return train bool: True if backward is required, otherwise False (inference only).
        """
        return self._have_forward and self._have_backward

    def reset_dependency(self):
        """
        Reset the node dataflow dependency

        Note all the predefined control dependencies will be removed.
        """
        for node in self._nodes:
            node.clear_predecessor()
            node.clear_successor()
        # TODO: adapter dependency not set
        for ftensor in self._full_tensors:
            for ptensor, producer in zip(ftensor.ptensors, ftensor.producers):
                for ctensor, consumer in zip(ftensor.ctensors, ftensor.consumers):
                    if ptensor.overlap(ctensor):
                        pidx = producer.outputs().index(ptensor)
                        cidx = consumer.inputs().index(ctensor)
                        producer.add_successor(pidx, consumer)
                        consumer.add_predecessor(cidx, producer)
                # set mirror as control dependency
                if producer.mirror and isinstance(producer, IRFwOperation):
                    producer.add_successor(-1, producer.mirror)
                    producer.mirror.add_predecessor(-1, producer)

    def attributes(self) -> Tuple[IRSubTensor]:
        """
        Return parameter list
        """
        return tuple(self._attributes)

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

    # ====================== Graph Accessment =========================
    
    def flatten(self) -> List[IRCell]:
        """
        Get all the single nodes by opening the segment.

        @return List[]
        """
        nodes = []
        for node in self._nodes:
            if isinstance(node, IRSegment):
                nodes += node._nodes
            else:
                nodes.append(node)
        return nodes

    def index(self, node: IRCell) -> GraphIndex:
        """
        Get node index in the graph.

        @param node IRCell: the queried node
        
        @return index Tuple[int, Optional[int]]: (GraphIndex, SegmentIndex)
            
        """
        if node in self._nodes:
            return GraphIndex(self._nodes.index(node), None)
        for idx, check_node in enumerate(self._nodes):
            if isinstance(check_node, IRSegment):
                if check_node.exist(node):
                    return GraphIndex(idx, check_node.index(node))
        raise KeyError(f"The queried node: {node} not in the graph.")

    def flatten_index(self, node: IRCell) -> int:
        """
        Get node index of all the flatten nodes
        
        @param node IRCell: the queried node, cannot be IRSegment

        @return index int: the index.
        """
        idx = 0
        for check_node in self._nodes:
            if isinstance(check_node, IRSegment):
                if node in check_node._nodes:
                    return idx + check_node.index(node)
                else:
                    idx += len(check_node.nnodes)
            if check_node == node:
                return idx
        raise KeyError(f"Node {node} not exist in graph")

    def node(self, index: Union[int, GraphIndex]) -> IRCell:
        """
        Get node given the index

        @param index Tuple[Optional[int], int]: the queired index of
            (SegmentIndex, Index)
        
        @return node IRCell: the quried node.
        """
        index = GraphIndex(index, None) if isinstance(index, int) else index
        assert isinstance(index, GraphIndex)
        node = self._nodes[index.gidx]
        if index.sidx is not None:
            assert isinstance(node, IRSegment), "Expected IRSegment"
            node = node.index(index.sidx)
        return node

    # ========================= Graph Manipulation ========================

    def remove(self, node: IRCell) -> GraphIndex:
        """
        Detach (remove) a node from current graph.
        TODO: dataflow dependency update.

        * Producer/consumer relationship:

          All the used input and output tensors inside the node
          are removed from consumed and produced tensor list.

        @param node IRCell: the removed node.

        @return index Tuple[int, Optional[int]]: index of the detached node in the graph
        """
        index = self.index(node)

        # remove node
        if index.sidx is None:
            self._nodes.pop(index.gidx)
        else:
            segment = self._nodes[index.gidx]
            assert isinstance(segment, IRSegment), "Internal Error: Removing at a wrong index"
            segment.remove(node)

        # update consumer and producer for non-adapter nodes
        rm_nodes = node.nodes() if isinstance(node, IRSegment) else [node]
        for node in rm_nodes:
            # adapter doesn't need to consider producer and consumer
            if isinstance(node, IRAdapter):
                continue
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
                ftensor = otensor.parent
                if len(ftensor.producers) == 0 and len(ftensor.consumers) == 0:
                    del self._full_tensors[otensor.parent]
        return index

    def insert(self, node: IRCell, index: Union[int, GraphIndex]):
        """
        Insert a node into current graph at node index.
        TODO: dataflow dependency update.

        * Producer/consumer relationship:

          For the node except IRAdapter, all its input and output tensors 
          will be recorded in consumed and produced tensor list. 
        
          IRAdapter node will not record the consumer and producer.

        @param node IRCell: the inserted node
        @param index Union[int, Tuple[int, Optional[int]]]: the inserted index
        """
        index = GraphIndex(index, None) if isinstance(index, int) else index
        assert isinstance(index, GraphIndex)

        # update producer and consumer
        in_nodes = node.nodes() if isinstance(node, IRSegment) else [node]
        for node in in_nodes:
            if isinstance(node, IRAdapter): continue
            # update consumer
            itensors: List[IRSubTensor] = []
            for itensor in node.inputs():
                if isinstance(itensor, IRSubTensor) and itensor not in itensors:
                    itensors.append(itensor)
            for itensor in itensors:
                self._full_tensors.add(itensor.parent)
                idx = 0
                for consumer in itensor.parent.consumers:
                    if self.index(consumer) < index:
                        idx += 1
                    else:
                        break
                itensor.parent.add_consumer(node, itensor, idx)
            # update producer
            otensors: List[IRSubTensor] = []
            for otensor in node.outputs():
                if isinstance(otensor, IRSubTensor) and otensor not in otensors:
                    otensors.append(otensor)
            for otensor in otensors:
                self._full_tensors.add(otensor.parent)
                idx = 0
                for producer in otensor.parent.producers:
                    if self.index(producer) < index:
                        idx += 1
                    else:
                        break
                otensor.parent.add_producer(node, otensor, idx)
        
        # insert node
        if index.sidx is None:
            self._nodes.insert(index.gidx, node)
        else:
            segment = self._nodes[index.gidx]
            assert isinstance(segment, IRSegment), "Expected to be a segment"
            segment.insert(node, index.sidx)

        return

    def exist(self, node: IRCell) -> bool:
        """
        Check if the node is in the graph

        @param node IRCell: the queried node
        @return existence bool: True if exist otherwise False
        """
        if node in self._nodes:
            return True
        for segment in self._nodes:
            if not isinstance(segment, IRSegment):
                continue
            if segment.exist(node):
                return True
        return False

    @contextmanager
    def update(self, node):
        """
        Update a node.
        TODO: update operator dependency

        e.g.,
            with graph.modify(node) as node:
                node.set_input(0, tensor)
        
        @param node IRCell: the node that must in the graph
        @return node IRCell: the modify node
        """
        index = self.remove(node)
        yield node
        self.insert(node, index)

    def replace(self, node: IRCell, new_nodes: List[IRCell]) -> int:
        """
        Replace one node by multiple nodes

        TODO: update dataflow dependency

        @param node IRCell: the replaced node
        @param new_nodes List[IRCell]: the nodes to be inserted.

        @return index int: the replaced node index
        """
        index = self.remove(node)
        for new_node in new_nodes[::-1]:
            self.insert(new_node, index)
        return index

    def group(self, fnodes: List[IRCell]) -> IRSegment:
        """!
        Group consecutive forward nodes into IRSegment.
        TODO: update operator dependency
        
        The corresponding backward nodes will also be grouped.

        @param nodes List[IRCell]: the consecutive node subset of this graph
        
        @return segment IRSegment: the grouped segment
        """
        assert any(not isinstance(node, (IRBpOperation, IRSegment, IRDataOperation)) for node in fnodes), \
            "grouped nodes cannot be backward operation, segment or data operation"
        
        findices = [self.index(fnode) for fnode in fnodes]
        
        # get backward nodes
        bnodes = [fnode.mirror for fnode in fnodes[::-1] if fnode.mirror is not None]
        bindices = [self.index(bnode) for bnode in bnodes]

        assert all(idx.sidx is None for idx in findices), \
            "Grouping operators that are already in segment is not allowed"
        assert all(idx.sidx is None for idx in bindices), \
            "Internal Error: backward operators found in segments"
        findices = tuple(idx.gidx for idx in findices)
        bindices = tuple(idx.gidx for idx in bindices)

        minfidx, maxfidx = min(findices), max(findices)
        assert maxfidx - minfidx + 1 == len(fnodes), \
            "Forward nodes are not consecutive"
            
        if len(bnodes) > 0:
            minbidx, maxbidx = min(bindices), max(bindices)
            assert maxbidx - minbidx + 1 == len(bnodes), \
                f"Internal Error: backward nodes are not consecutive. maxbidx: {maxbidx}, minbidx: {minbidx}"

        fsegment = self.segment(fnodes)
        bsegment = self.segment(bnodes) if len(bnodes) > 0 else None
        IRCell.make_pair(fsegment, bsegment)

        # replace backward
        if len(bnodes) > 0:
            self._nodes = self._nodes[:minbidx] + [bsegment] + self._nodes[maxbidx+1:]
        # replace forward
        self._nodes = self._nodes[:minfidx] + [fsegment] + self._nodes[maxfidx+1:]

        return fsegment

    # ========================== Graph Creation ========================

    def segment(self, nodes: List[IRCell]) -> IRSegment:
        """!
        Create a segment with part of the nodes.

        @param nodes List[IRCell]: the subset nodes of this graph

        @return segment IRSegment: the grouped segment. 
        """
        inputs, outputs = [], []
        itdevs, otdevs = dict(), dict()
        for node in nodes:
            assert not isinstance(node, IRSegment), 'A segment cannot be in other segments'
            # update inputs
            itensors = [t for t in node.inputs() if isinstance(t, IRSubTensor)]
            for itensor in itensors:
                producers = [p for p in itensor.parent.producers if set(p.device).issubset(set(node.device))]
                # no producer means a weight or cross device-group
                if len(producers) == 0 or any(p not in nodes for p in producers):
                    if itensor not in itdevs:
                        itdevs[itensor] = []
                    devs = set(itensor.device)
                    if devs not in itdevs[itensor]:
                        inputs.append(itensor)
                        itdevs[itensor].append(devs)
            # update outputs
            otensors = [t for t in node.outputs() if isinstance(t, IRSubTensor)]
            for otensor in otensors:
                consumers = [c for c in otensor.parent.consumers if set(c.device).issubset(set(node.device))]
                # no consumer usually means the loss or cross device-group
                if otensor in self.outputs() or len(consumers) == 0 or any(c not in nodes for c in consumers):
                    devs = set(otensor.device)
                    if otensor not in otdevs:
                        otdevs[otensor] = []
                    if devs not in otdevs[otensor]:
                        outputs.append(otensor)
                        otdevs[otensor].append(devs)
        segment = IRSegment(nodes, inputs, outputs)
        return segment

    @staticmethod
    def from_logic_graph(nodes: List[IRCell],
                         inputs: List[IRFullTensor], outputs: List[IRFullTensor],
                         module_name: str):
        """
        Generate IRGraph from logical graph (IRFullTensor)

        Multiref will be inserted:

        e.g., original graph:
            ```
            t = producer(xx)
            ...
            xx = consumer1(t)
            ...
            xx = consumer2(t)
            ...
            xx = consumer3(t)
            ...
            ```
        will be changed into:
            ```
            t = producer(xx)
            ...
            t1, t2 = multiref(t)
            xx = consumer1(t1)
            ...
            t3, t4 = multiref(t2)
            xx = consumer2(t3)
            ...
            xx = consumer3(t4)
            ...
            ```
        """
        # handle multi-consumed tensor
        consumers: Dict[IRFullTensor, List[IRCell]] = dict()
        producers: Dict[IRFullTensor, IRCell] = dict()
        for node in nodes:
            ftensors = set()
            for ftensor in node.inputs():
                # remove redundant tensors within an operator
                if isinstance(ftensor, IRFullTensor) and ftensor.tid not in ftensors:
                    ftensors.add(ftensor.tid)
                    if ftensor not in consumers:
                        consumers[ftensor] = []
                    consumers[ftensor].append(node)
            for ftensor in node.outputs():
                if isinstance(ftensor, IRFullTensor):
                    producers[ftensor] = node
        for ftensor, cnodes in consumers.items():
            if len(cnodes) == 1 or ftensor.is_attr(): continue
            reftensor = ftensor
            ctensor = ftensor
            while len(cnodes) > 0:
                consumer = cnodes.pop(0)
                if len(cnodes) > 0:
                    itensors = [ftensor.like() for _ in range(2)]
                    multiref = MultiRef(None, [reftensor, 2])
                    for idx, itensor in enumerate(itensors):
                        multiref.set_output(idx, itensor)
                    multiref.infer_shape()
                    # insert multiref right before the consumor
                    idx = nodes.index(consumer)
                    nodes.insert(idx, multiref)
                    ctensor, reftensor = itensors
                else:
                    # the last consumer doesn't need multiref
                    ctensor = reftensor
                # update consumer
                while ftensor in consumer.inputs():
                    idx = consumer.inputs().index(ftensor)
                    consumer.set_input(idx, ctensor)

        # instantiate graph inputs / outputs
        for idx, tensor in enumerate(inputs):
            if isinstance(tensor, IRFullTensor):
                tensor = tensor.tosub()
            inputs[idx] = tensor
        for idx, tensor in enumerate(outputs):
            if isinstance(tensor, IRFullTensor):
                tensor = tensor.tosub()
            outputs[idx] = tensor

        # instantiate to subtensor
        for node in nodes:
            for idx, ftensor in enumerate(node.inputs()):
                ftensors = set()
                if isinstance(ftensor, IRFullTensor):
                    subtensor = ftensor.tosub()
                    node.set_input(idx, subtensor)
            for idx, ftensor in enumerate(node.outputs()):
                if isinstance(ftensor, IRFullTensor):
                    subtensor = ftensor.tosub()
                    node.set_output(idx, subtensor)
        graph = IRGraph(nodes, inputs, outputs, module_name)
        return graph

    ##### Transformation Primitives #####

    def replicate(self, node: Union[IRFwOperation, IRDataOperation], times=1) -> List[IRCell]:
        """
        Partition Primitive:
            - replicate: replicate a forward or data operation multiple times.
        
        Each input and output will be replicated with no gradient accumulation.

        The backward of the forward operation will automatically be replicated.

        @param: node: Union[IRFwOperation, IRDataOperation]
        """
        if not isinstance(node, (IRFwOperation, IRDataOperation)):
            raise TypeError("Expected op to be forward op or data op")
        if not isinstance(times, int) or times < 1:
            raise TypeError("Expected times to be int and >= 1")

        if node not in self.nodes():
            raise RuntimeError(f"Op {node} not exsits")

        fnodes = [node.replicate() for _ in range(times)]

        # insert forward
        self.replace(node, fnodes)
        for fnode in fnodes:
            if isinstance(node, IRFwOperation):
                fnode.recompute = node.recompute
            if isinstance(node.comment, str):
                fnode.comment = node.comment
            fnode.device = node.device
        
        # insert backward
        if isinstance(node.mirror, IRBpOperation):
            bnode: IRBpOperation = node.mirror
            for fnode in fnodes:
                fnode.gen_backward()
            bnodes = [fnode.mirror for fnode in fnodes[::-1]]
            self.replace(bnode, bnodes)
            for bnode in bnodes:
                bnode.device = node.device
        return fnodes

    def partition(self, node: Union[IRFwOperation, IRDataOperation],
                  algo: GenericDistAlgo, **config) -> Optional[List[IRCell]]:
        """
        Partition Primitive:
            - partition: partition a forward or data operation using algorithms.
        
        The comment in the node will be inherited to partitioned nodes.
        The backward of the forward operation will be automatically partitioned.

        Requirement to partition algorithm:
            if backward is required, the algorithm can only transform tensors in:
                replicate: results in gradient accumulation
                split dimensionL no gradient accumulation
                split value (outputs only): no gradient accumulation

        Difference of partition and replicate primitive:
          Both primitive may replicate the tensors, but `replicate` will not do gradient
          accumulation while `partition` will always require gradient accumulation on
          replicated tensors.
        
        @param node Union[IRFwOperation, IRDataOperation]: the node to partition
        @param algo GenericDistAlgo: the partition algorithm related to the node
        @param config Dict[str, Any]: the algorithm configuration, e.g., partition number

        @return Optional[IRCell]: partitioned sub-nodes or None (fail to partition)
        """
        assert isinstance(algo, GenericDistAlgo) and node == algo.node, \
            "The partition algorithm is not initialized for this node"
        assert isinstance(node, (IRFwOperation, IRDataOperation)), \
            f"Only allow op to be forward op or data op, but got: {node}"

        # get partitioned sub-nodes
        fnodes = algo.instantiate(**config)
        if fnodes is None:
            return None

        # update forward
        self.replace(node, fnodes)
        for fnode in fnodes:
            if isinstance(node, IRFwOperation):
                fnode.recompute = node.recompute
            if isinstance(node.comment, str):
                fnode.comment = node.comment
            fnode.device = node.device
        # update backward
        if isinstance(node.mirror, IRBpOperation):
            bnodes = [fnode.gen_backward() for fnode in fnodes[::-1]]
            self.replace(node.mirror, bnodes)
            for bnode in bnodes:
                bnode.device = node.device
        # update gradient
        updated = set()
        for itensor in [t for t in node.inputs() if isinstance(t, IRSubTensor)]:
            for fnode in itensor.parent.consumers:
                bnode: IRBpOperation = fnode.mirror
                if isinstance(bnode, IRBpOperation) and fnode.cid not in updated:
                    with self.update(bnode):
                        bnode.update()
                updated.add(fnode.cid)
        return fnodes

    ## Spatial Primitives ##

    def assign(self, node: Union[IRFwOperation, IRDataOperation], device: int) -> bool:
        """
        Assign an operator (subgraph) to (multiple) rank(s).

        Corresponding backward operators (if have) will also be 
        assigned to the same device.

        @param node Union[IRFwOperation, IRBpOperation, IRSegment]: operator
        @param device int: assigned device id

        @return sucess bool: always true
        """
        assert self.exist(node), f"{node} is not in the graph"
        if isinstance(node, IRSegment):
            assert node.forward, "Only forward segment is allowed to assign devices"
            for subnode in node.nodes():
                subnode.device = device
                if subnode.mirror is not None:
                    subnode.mirror.device = device
        else:
            assert isinstance(node, (IRFwOperation, IRDataOperation)), \
                "Only forward operators and dataloader operators are allowed to assign devices"
            node.device = device
            if node.mirror is not None:
                node.mirror.device = device
        return True

    ## Schedule Policy Primitives ##

    def happen_before(self, node1: IRCell, node2: IRCell, skip=None) -> bool:
        """
        Check node1 -> (happen before) node2

        Returns:
            Boolean
        """
        raise NotImplementedError("dependency is not supported yet")
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

    def depends(self, pre_node: IRCell, post_node: IRCell) -> bool:
        """!
        Check whether pre_node has dataflow dependency on post_node:
            pre_node -> post_node

        @param pre_node: the happen before node
        @param post_node: the happen after node

        @return ret bool: True if post_node depends on pre_node on dataflow, otherwise False.
        """
        itensors = [t for t in post_node.inputs() if isinstance(t, IRSubTensor)]
        for otensor in pre_node.outputs():
            if not isinstance(otensor, IRSubTensor): continue
            for itensor in itensors:
                if otensor.overlap(itensor):
                    return True
        return False

    def schedule(self, node1: IRCell, action: str, node2: IRCell) -> bool:
        """!
        Schedule node1 and node2 based on the action

        The node2 will keep unchanged in the sequence and schedule will perform
        on node1.

        @param node1 IRCell
        @param node2 IRCell
        @param action str:
            'after': fixed node2 and schedule node1 after node2 in the sequence.
            'before': fixed node2 and schedule node1 before node2 in the sequence.
        
        @return success bool: True if the scheduling success otherwise False.
        """
        idx1 = self._nodes.index(node1)
        idx2 = self._nodes.index(node2)
        # node2 -> node1
        if action == 'after':
            if idx2 < idx1:
                return True
            for idx in range(idx1+1, idx2+1):
                if self.depends(node1, self._nodes[idx]):
                    return False
            self.remove(node1)
            self.insert(node1, idx2)
            return True
        # node1 -> node2
        if action  == 'before':
            if idx1 < idx2:
                return True
            for idx in range(idx2, idx1):
                if self.depends(self._nodes[idx], node1):
                    return False
            self.remove(node1)
            self.insert(node1, idx2)
            return True
        raise KeyError(f"Unknown scheduling action {action}")

    @property
    def sched(self):
        """!
        Return schedule plan for the execution.
        """
        return self._sched

    @sched.setter
    def sched(self, strategy):
        """!
        Set schedule plan for the execution.

        @param strategy IRScheduleStrategy: the schedule strategy instance
        """
        self._sched = strategy

    @staticmethod
    def legal_schedule(seq: List[IRCell], integrity_check=False):
        """
        Check whether seq satisfies topological order.

        @note: this functionality is not enabled due to predecessor and succesor
        functionality.
        
        @param seq List[IRCell]: the nodes in scheudled order
        @param integrity_check bool:
                If true, performs additional integrity check that requires
                all the nodes in predecessor and successor of a node should
                appear in the sequence.
        
        @return valid bool: True for satisfying topo order, otherwise False.
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

    # ================= staging primitives ==================

    def staging(self, nodes: Tuple[IRFwOperation]):
        """!
        Group forward / dataloader operators into sequential stages.
        The corresponding backward operators will also be grouped into stages
        Cross-stage dataflow will be limited to neighbor stages.
        This should be called before any operator partition.

        The transformation and temporal scheduling can only be applied within each stage.
        For example, after staging, user cannot schedule a (transformed) node 
        from one stage to another stage.

        The stage is a concept that is only about logical separation of nodes, 
        it doesn't have additional constraints for device assignment.

        Changes will be made:

        1). Identity creation:
            If a non-attribute tensor is produced / consumed not in
            neighbor stages, 
                e.g., 
                    stage 1: t1 = producer()
                    stage 2: ...
                    stage 3: xx = consume(t1)
                    stage 4: ...
                    stage 5: xx = consume(t1)
            then Identity nodes will be created for every device in stage2:
                    stage 1: t1 = producer()
                    stage 2: t2 = identity(t1)
                    stage 3: xx = consume(t2)
                    stage 4: t3 = identity(t2)
                    stage 5: xx = consume(t3)
    
        2). REMOVED: Multiref Modification:
            If a non-attribute tensor has multiref node to different devmeshes,
                e.g., 
                    stage 1: t1, t2 = multiref(t) 
                    stage 2: xx = consume(t1)
                    stage 3: ...
                    stage 4: xx = consume(t2)
            then the multiref will be transfered into identity operator:
                    stage 1: t1 = multiref(t)
                    stage 2: xx = consume(t1)
                             t2 = identity(t1)
                    stage 3: t3 = identity(t2)
                    stage 4: xx = consume(t3)

        @param starts  Tuple[int]: the start index of each stage
        @return None
        """
        assert all(isinstance(node, IRFwOperation) for node in nodes), \
            f"Find node is not IRFwOperation or IRDataOperation: {node}"
        assert all(node in self._nodes for node in nodes), \
            f"Exist node is not in graph nodes"
        starts = tuple(self._nodes.index(node) for node in nodes)
        assert len(starts) > 0
        starts = (0,) + starts if starts[0] != 0 else starts

        last_fidx = 0
        for idx, node in enumerate(self._nodes):
            if not isinstance(node, IRBpOperation):
                last_fidx = idx
        
        fstages: List[List[IRCell]] = []
        bstages: List[List[IRCell]] = []
        for sid in range(len(starts)):
            begin = starts[sid]
            end = starts[sid+1] if sid != len(starts) - 1 else last_fidx + 1
            while isinstance(self.node(begin), IRDataOperation):
                begin += 1
            while isinstance(self.node(end), IRDataOperation):
                end -= 1
            if begin == end: continue
            assert begin < end
            fnodes = self._nodes[begin:end]
            bnodes = [fnode.mirror for fnode in fnodes[::-1] if fnode.mirror is not None]
            fstages.append(fnodes)
            bstages = [bnodes] + bstages

        def get_sid(fnode: IRCell) -> Optional[int]:
            for idx, fnodes in enumerate(fstages):
                if fnode in fnodes:
                    return idx
            return None

        def insert_identity(tensor: IRSubTensor, sid: int) -> IRFwOperation:
            identity = Identity('', [tensor])
            identity.infer_shape()
            identity.set_output(0, identity.output(0).tosub())
            # insert forward
            self.insert(identity, self.index(fstages[sid][0]))
            fstages[sid].insert(0, identity)
            
            # insert backward
            if self.train:
                bnode = identity.gen_backward()
                self.insert(bnode, self.index(bstages[sid][-1]) + 1)
                bstages[sid].append(bnode)
            return identity

        # create identity op for cross-stage dataflow
        # the gradient flow of neighbor stages is automatically guaranteed
        for ftensor in self.full_tensors():
            if ftensor.is_grad() or ftensor.is_attr(): continue
            assert len(ftensor.producers) <= 1, \
                "The staging interface should be called before any operator partition."
            if len(ftensor.consumers) == 0: continue
            producer, ptensor = ftensor.producers[0], ftensor.ptensors[0]
            psid = get_sid(producer)
            # outside of stages, not consider
            if psid is None: continue 
            out = ptensor
            curr_sid = psid
            for ctensor, consumer in zip(ftensor.ctensors, ftensor.consumers):
                assert ctensor == ptensor, "The staging interface should be called before any operator partition." 
                csid = get_sid(consumer)
                if curr_sid == csid: continue
                for sid in range(curr_sid + 1, csid):
                    identity = insert_identity(out, sid)
                    out = identity.output(0)
                # update consumer and its backward
                with self.update(consumer) as consumer:
                    tidx = consumer.inputs().index(ptensor)
                    consumer.set_input(tidx, out)
                if self.train:
                    with self.update(consumer.mirror) as bnode:
                        bnode.update()
                curr_sid = csid
        
        # grouping into segment
        for sid in range(len(fstages)):
            self.group(fstages[sid])


    # ================= Other optimizations ==================

    def recompute(self, nodes: Union[List[IRFwOperation], IRSegment]) -> bool:
        """!
        Recompute a set of nodes. The forward nodes will be assigned with a unique
        recompute group id. A forward not can not be recomputed in different recompute groups.

        @param nodes List[IRFwOperation]: nodes for a recompute group

        @return success boolean: always success
        """
        assert all(isinstance(nodes, IRFwOperation)) or isinstance(nodes, IRSegment), \
            "Require forward nodes or a single segment"

        recompute_group_id: int = IDGenerator().gen_cell_id()

        if isinstance(nodes, IRSegment):
            assert nodes.forward, "Can only apply recompute on segment node"
            for fnode in nodes.node():
                fnode.recompute = recompute_group_id
        else:
            indices = [self.index(node) for node in nodes]
            if all(idx[1] is None for idx in indices):
                assert all(idx[0] == indices[0][0] for idx in indices), \
                    f"Cross-stage recompute is not allowed yet."
            elif all(idx[1] is not None for idx in indices):
                assert all(idx[0] == indices[0][0] for idx in indices), \
                    f"Cross-stage recompute is not allowed yet."
            else:
                assert False, f"Cross-stage recompute is not allowed yet."
            for fnode in nodes:
                fnode.recompute = recompute_group_id
    
        return True

    def __repr__(self) -> str:
        dscp = f"Graph{self._id}-{self.device}(inputs={self.inputs()}, outputs={self.outputs()})"
        return dscp

    def extra_repr(self) -> str:
        dscp = f"\n{self.name}:\n{'=' * len(self.name)}\n"
        # inputs
        dscp += f"Inputs: {self.inputs()}\n"
        for node in self._nodes:
            dscp += f"\n{node}"
            if isinstance(node, IRSegment):
                for subnode in node.nodes():
                    dscp += f"\n\t{subnode}"
        # outputs
        dscp += f"\nOutputs: {self.outputs()}\n{'=' * len(self.name)}\n"
        return dscp
