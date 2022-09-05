"""
IRGraph:
    a graph that is composed by node (IRFwOperation) and edge (IRTensor).

    Note the device of graph.inputs() can be different of the same input
    tensor of operation node in the graph. In this case, a move operation
    will be inserted at scheduling time.
"""

from typing import Union, Tuple, List, Optional, Dict

from cube.ir.cten import IRTensor, IRCell
from cube.ir.unique import IDGenerator
from cube.ir.operator import IRBpOperation, IRFwOperation, IRDataOperation
from cube.ir.tensor import IRFullTensor, IRSubTensor
from cube.ir.dtype import IRDType, DTypeInferRule

from cube.graph.function.function import Identity, MultiRef
from cube.graph.segment import IRSegment

from cube.algorithm.generics import GenericDistAlgo


class IRGraph(IRSegment):
    """
    IRGraph.

    IRGraph is used for reprensting a distributed training iteration.
    """

    def __init__(self, nodes: List[IRCell], inputs: List[IRTensor], outputs: List[IRTensor], 
                 module_name: str):

        super().__init__(nodes, inputs, outputs, module_name)

        self._sched = None  # the schedule strategy


    @property
    def train(self) -> bool:
        """!
        Train flag.

        @return train bool: True if backward is required, otherwise False (inference only).
        """
        return self._have_forward and self._have_backward

    # ================ Deep Learning Interfalce ======================

    def __call__(self, *args):
        """
        Register forward action
        """
        return self.forward(*args)
    
    def forward(self, *args: Tuple[IRSubTensor]) -> Union[IRTensor, Tuple[IRTensor]]:
        """
        forward will divide the graph into Actions according to
        node device assignment

        Currently each forward call will result in a new flow
        even if the input is same

        @param args Tuple[Any]

        @return outputs Union[IRSubTensor, Tuple[IRSubTensor]]
        """
        # align graph with input tensors
        itensors: Tuple[IRSubTensor, ...] = self.inputs()
        assert len(args) == len(itensors)
        for idx, (itensor, arg) in enumerate(zip(itensors, args)):
            self.set_input(idx, arg)
            for producer in self.producers(itensor.parent):
                with self.update(producer):
                    while itensor in producer.outputs():
                        oidx = producer.outputs().index(itensor)
                        producer.set_output(oidx, arg)
            for consumer in self.consumers(itensor.parent):
                with self.update(consumer):
                    while itensor in consumer.inputs():
                        iidx = consumer.inputs().index(itensor)
                        consumer.set_input(iidx, arg)
            while itensor in self.outputs():
                oidx = self.outputs().index(itensor)
                self.set_output(oidx, arg)
            while itensor in self.inputs():
                iidx = self.inputs().index(itensor)
                self.set_input(iidx, arg)
        
        # dtype inference
        for node in self._nodes:
            itensors = [t for t in node.inputs() if isinstance(t, IRSubTensor)]
            # setup gradient
            for itensor in itensors:
                if itensor.parent.grad is not None:
                    itensor.parent.dtype = itensor.dtype
            if len(itensors) == 0: continue
            odtype = DTypeInferRule.infer(node, [t.dtype for t in itensors])
            assert odtype != IRDType.unknown, f"{node} : {[t.dtype for t in itensors]}"
            otensors = [t for t in node.outputs() if isinstance(t, IRSubTensor)]
            for tensor in otensors:
                tensor.dtype = odtype
                # setup graidient
                if tensor.parent.grad is not None:
                    tensor.parent.grad.dtype = odtype

        from cube.program import Program
        Program().add_nodes(self.nodes())

        # return
        if len(self.outputs()) == 1:
            return self.output(0)
        else:
            return self.outputs()

    def backward(self, loss: IRSubTensor):
        """
        Backward the graph from the entry tensor of loss.

        @param loss IRSubTensor: the loss tensor, must be in the output
            of current graph. The loss shape should be (1,)

        @return self IRGraph: None
        """
        assert loss in self.outputs() and tuple(loss.shape) == (1,), \
            f"backward should be in graph outputs and the loss is of shape [1,] (got {loss.shape})"
        from cube.program import Program
        loss.parent.grad = 1.0
        for fnode in self.nodes()[::-1]:
            assert not isinstance(fnode, IRSegment), "Internal Error: Segment should not appear for now"
            if isinstance(fnode, IRFwOperation):
                bnode: IRBpOperation = self.create_bwop(fnode)
                Program().add_node(bnode)
        # set program graph mirror to self
        Program().mirror_as_self()
        return self


    # ========================= Graph Manipulation ========================

    def group(self, fnodes: List[IRCell]) -> IRSegment:
        """!
        Group consecutive forward nodes into IRSegment.
        TODO: update operator dependency
        
        The corresponding backward nodes will also be grouped.

        @param nodes List[IRCell]: the consecutive node subset of this graph
        
        @return segment IRSegment: the grouped segment
        """
        assert any(not isinstance(node, (IRBpOperation, IRDataOperation)) for node in fnodes), \
            "grouped nodes cannot be backward operation, segment or data operation"
        
        fgraphs = [self.segment(fnode) for fnode in fnodes]
        assert len(set(fgraphs)) == 1, "Cross-segment grouping is not allowed yet."
        
        # get backward nodes
        bnodes = [fnode.mirror for fnode in fnodes[::-1] if fnode.mirror is not None]
        
        fgraph: IRSegment = fgraphs[0]
        bgraph: IRSegment = fgraph.mirror

        findices: Tuple[int] = tuple(fgraph.index(fnode)[0] for fnode in fnodes)
        bindices: Tuple[int] = tuple(bgraph.index(bnode)[0] for bnode in bnodes)

        minfidx, maxfidx = min(findices), max(findices)
        assert maxfidx - minfidx + 1 == len(fnodes), \
            "Forward nodes are not consecutive"

        if len(bnodes) > 0:
            minbidx, maxbidx = min(bindices), max(bindices)
            assert maxbidx - minbidx + 1 == len(bnodes), \
                f"Internal Error: backward nodes are not consecutive. maxbidx: {maxbidx}, minbidx: {minbidx}"

        fsegment = fgraph.create_segment(fnodes)
        bsegment = bgraph.create_segment(bnodes) if len(bnodes) > 0 else None
        IRCell.make_pair(fsegment, bsegment)

        # replace forward
        for fnode in fnodes:
            fidx = fgraph.remove(fnode)
        fgraph.insert(fsegment, fidx)

        # replace backward
        if len(bnodes) > 0:
            for bnode in bnodes:
                bidx = bgraph.remove(bnode)
            bgraph.insert(bsegment, bidx)
            # setup gradient
            self.update_bwop(bsegment)

        return fsegment

    # ========================== Graph Creation ========================

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

        fsegment: IRSegment = self.segment(node)
        # replicate
        fnodes = [node.replicate() for _ in range(times)]
        # insert forward
        for fnode in fnodes:
            if isinstance(node, IRFwOperation):
                fnode.recompute = node.recompute
            if isinstance(node.comment, str):
                fnode.comment = node.comment
            fnode.device = node.device
        fsegment.replace(node, fnodes)
        # insert backward
        bsegment: IRSegment = fsegment.mirror
        if isinstance(node.mirror, IRBpOperation):
            bnodes = tuple(self.create_bwop(fnode) for fnode in fnodes[::-1])
            for bnode in bnodes:
                bnode.device = node.device
            bsegment.replace(node.mirror, bnodes)
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
        
        fsegment: IRSegment = self.segment(node)
        # get partitioned sub-nodes
        fnodes = algo.instantiate(**config)
        assert fnodes is not None, f"Fail to partition node: {node} use algothim and config: {config}"
        # update forward
        for fnode in fnodes:
            if isinstance(node, IRFwOperation):
                fnode.recompute = node.recompute
            if isinstance(node.comment, str):
                fnode.comment = node.comment
            fnode.device = node.device
        fsegment.replace(node, fnodes)
        # update backward
        bsegment: IRSegment = fsegment.mirror
        if isinstance(node.mirror, IRBpOperation):
            bnodes = tuple(self.create_bwop(fnode) for fnode in fnodes[::-1])
            bsegment.replace(node.mirror, bnodes)
            for bnode in bnodes:
                bnode.device = node.device
        # update gradient
        updated = set()
        for itensor in [t for t in node.inputs() if isinstance(t, IRSubTensor)]:
            for fnode in fsegment.consumers(itensor.parent):
                bnode: IRBpOperation = fnode.mirror
                if isinstance(bnode, IRBpOperation) and fnode.cid not in updated:
                    self.update_bwop(bnode)
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
            fidx = self.index(fstages[sid][0])
            if tensor.requires_grad:
                self.finsert(identity, fidx)
                bstages[sid].append(identity.mirror)
            else:
                self.insert(identity, fidx)
            fstages[sid].insert(0, identity)
            return identity

        # create identity op for cross-stage dataflow
        # the gradient flow of neighbor stages is automatically guaranteed
        for ftensor in self.full_tensors():
            if ftensor.is_grad() or ftensor.is_attr(): continue
            assert len(self.producers(ftensor)) <= 1, \
                "The staging interface should be called before any operator partition."
            if len(self.consumers(ftensor)) == 0: continue
            producer, ptensor = self.producers(ftensor)[0], self.ptensors(ftensor)[0]
            psid = get_sid(producer)
            # outside of stages, not consider
            if psid is None: continue 
            out = ptensor
            curr_sid = psid
            for ctensor, consumer in zip(self.ctensors(ftensor), self.consumers(ftensor)):
                assert ctensor == ptensor, "The staging interface should be called before any operator partition." 
                csid = get_sid(consumer)
                if curr_sid == csid: continue
                for sid in range(curr_sid + 1, csid):
                    identity = insert_identity(out, sid)
                    out = identity.output(0)
                # update consumer
                with self.update(consumer) as consumer:
                    tidx = consumer.inputs().index(ptensor)
                    consumer.set_input(tidx, out)
                curr_sid = csid
            # update all its backward operators
            self.update_ftensor_bw(ftensor.grad)
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
