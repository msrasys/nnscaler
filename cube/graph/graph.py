"""
IRGraph:
    a graph that is composed by node (IRFwOperation) and edge (IRTensor).

    Note the device of graph.inputs() can be different of the same input
    tensor of operation node in the graph. In this case, a move operation
    will be inserted at scheduling time.
"""

from typing import Sequence, Set, Union, Tuple, List, Optional, Dict, Any
import warnings
import copy
import pickle
import dill

from cube.ir.cten import IRTensor, IRCell, IRObject
from cube.ir.unique import IDGenerator
from cube.ir.operator import IRBpOperation, IRFwOperation, IRDataOperation
from cube.ir.tensor import IRFullTensor, IRSubTensor, IndexMap, ValueMap
from cube.ir.dtype import IRDType, DTypeInferRule

from cube.graph.function.function import Identity, MultiRef
from cube.graph.function.anchor import IRGraphAnchor
from cube.graph.function.pyfunc import IRPyFunc
from cube.graph.function.dimops import IRDimops
from cube.graph.segment import IRSegment

from cube.algorithm.generics import GenericDistAlgo


FOp = Union[IRFwOperation, IRDataOperation]


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
        return any(not n.isfw() for n in reversed(self._nodes))

    # ================ Deep Learning Interfalce ======================

    def __call__(self, *args):
        """
        Register forward action
        """
        return self.forward(*args)
    
    def forward(self, *args: Tuple[Any]) -> Union[IRTensor, Tuple[IRTensor]]:
        """
        forward will divide the graph into Actions according to
        node device assignment

        Currently each forward call will result in a new flow
        even if the input is same

        @param args Tuple[Any]

        @return outputs Union[IRSubTensor, Tuple[IRSubTensor]]
        """
        # align graph with input tensors
        itensors: Tuple[IRObject, ...] = self.inputs()
        if len(args) != len(itensors):
            print(f'ERROR(skipping) len(args) != len(itensors): {len(args)} != {len(itensors)}')
            if len(args) > len(itensors):
                args = args[:len(itensors)]
                print(f'WARNING: args shrinked into {args}')
            else:
                raise RuntimeError('len(args) < len(itensors)')

        arg_objs = IRGraph.get_objects_from_complex(args)
        graph_objs = IRGraph.get_objects_from_complex(self.inputs())
        assert len(arg_objs) == len(graph_objs), f"input object number not match: {len(arg_objs)} != {len(graph_objs)}"

        for idx, (itensor, arg) in enumerate(zip(itensors, args)):
            self.set_input(idx, arg)

        for arg, itensor in zip(arg_objs, graph_objs):
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
            # reset output
            for oidx, output in enumerate(self.outputs()):
                output = IRGraph.modify_objects_of_complex(
                    self.output(oidx), lambda t: t if t != itensor else arg)
                self.set_output(oidx, output)

        # dtype inference
        for node in self._nodes:
            # reset input
            itensors: List[IRTensor] = [t for t in node.inputs() if isinstance(t, IRSubTensor)]
            for itensor in itensors:
                itensor.parent.dtype = itensor.dtype
            # infer output dtype with default dtype promotion rules
            if len(itensors) == 0: continue
            default_dtype = DTypeInferRule.infer(node, [t.dtype for t in itensors])
            # set output tensors if it has unkown tensor dtype
            otensors = [t for t in node.outputs() if isinstance(t, IRSubTensor)]
            for otensor in otensors:
                if otensor.dtype == IRDType.unknown:
                    otensor.parent.dtype = default_dtype

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
        # set mirror as self
        self._mirror = self
        # set loss gradient
        loss.parent.to_loss()

        # infer gradient
        for ftensor in self.full_tensors():
            self.infer_grad(ftensor)
        # create backward node
        for fnode in self.nodes()[::-1]:
            assert not isinstance(fnode, IRSegment), "Internal Error: Segment should not appear for now"
            if not isinstance(fnode, IRFwOperation): continue
            tensors = [t for t in fnode.inputs() + fnode.outputs() if isinstance(t, IRSubTensor)]
            grads = [t.grad for t in tensors]
            # no backward op generated for fnode
            if all(grad is None for grad in grads): continue
            # create backward op and insert to graph
            bwop = self.create_bwop(fnode)
            self.insert(bwop, self.nnodes)

        return self


    # ========================= Graph Manipulation ========================

    def group(self, fnodes: List[IRCell]) -> IRSegment:
        """!
        Group consecutive forward nodes into IRSegment.
        Note the fnodes should not apply any transformation.
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

        # remove fnodes and insert fsegment
        fsegment: IRSegment = fgraph.create_segment(fnodes)
        for fnode in fnodes:
            fidx = fgraph.remove(fnode)
        fgraph.insert(fsegment, fidx)

        # reset fsegment gradient
        for itensor in fsegment.inputs():
            if isinstance(itensor, IRTensor):
                fgraph.infer_grad(itensor.parent)

        # update backward
        if len(bnodes) > 0:
            # remove backward nodes
            for bnode in bnodes:
                bidx = bgraph.remove(bnode)
            # create new backward node
            bnodes = [fsegment.create_bwop(fnode) for fnode in fnodes[::-1]]
            # create and insert backward segment
            bsegment = fgraph.create_bwop(fsegment)
            bgraph.insert(bsegment, bidx)

    # ========================== Graph Creation ========================

    @staticmethod
    def from_logic_graph(nodes: List[IRCell],
                         inputs: List[Any], outputs: List[Any],
                         module_name: str):
        """
        Generate IRGraph from logical graph (IRFullTensor)

        @param nodes: nodes of the graph
        @param inputs List[Any]: graph inputs
        @param outputs List[Any]: graph outputs
        @param module_name str: graph name

        @return graph IRGraph
        """
        modifier = lambda t: t.tosub() if isinstance(t, IRFullTensor) else t
        # input / output
        inputs = [IRGraph.modify_objects_of_complex(t, modifier) for t in inputs]
        outputs = [IRGraph.modify_objects_of_complex(t, modifier) for t in outputs]
        # nodes
        for node in nodes:
            for idx, ftensor in enumerate(node.inputs()):
                if isinstance(ftensor, IRObject):
                    subtensor = ftensor.tosub() if isinstance(ftensor, IRFullTensor) else ftensor
                    node.set_input(idx, subtensor)
            for idx, ftensor in enumerate(node.outputs()):
                if isinstance(ftensor, IRObject):
                    subtensor = ftensor.tosub() if isinstance(ftensor, IRFullTensor) else ftensor
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

        @param node Union[IRFwOperation, IRDataOperation]

        @return ops List[IRCell]: the replicated operators
        """
        if not isinstance(node, (IRFwOperation, IRDataOperation)):
            raise TypeError("Expected op to be forward op or data op")
        if not isinstance(times, int) or times < 1:
            raise TypeError("Expected times to be int and >= 1")
        if node.name == 'multiref':
            warnings.warn(
                'Detected partition a multiref node. This will be skipped as system will automatically handle it.')
            return [node]
        if isinstance(node, IRPyFunc):
            warnings.warn(
                'Detected partition a python runtime function. This will be skipped as system will automatically handle it')
            return [node]

        fsegment: IRSegment = self.segment(node)
        # replicate
        fnodes = [node.replicate() for _ in range(times)]
        # set gradient
        for fnode in fnodes:
            for rtensor, itensor in zip(fnode.inputs(), node.inputs()):
                if isinstance(rtensor, IRSubTensor):
                    rtensor.grad = copy.copy(itensor.grad)
            for rtensor, itensor in zip(fnode.outputs(), node.outputs()):
                if isinstance(rtensor, IRSubTensor):
                    rtensor.grad = copy.copy(itensor.grad)
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
        if isinstance(node.mirror, IRCell):
            bnodes = [node.mirror.replicate() for _ in range(times)]
            for bnode, fnode in zip(bnodes, fnodes[::-1]):
                IRCell.make_pair(fnode, bnode)
                bnode.device = fnode.device
            bsegment.replace(node.mirror, bnodes)
        return fnodes

    def partition(self, node: Union[IRFwOperation, IRDataOperation],
                  algo: GenericDistAlgo, **config) -> List[IRCell]:
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

        @return ops List[IRCell]: partitioned sub-nodes
        """
        assert isinstance(algo, GenericDistAlgo) and node == algo.node, \
            f"The partition algorithm ({algo}) is not initialized for this node"
        assert isinstance(node, (IRFwOperation, IRDataOperation)), \
            f"Only allow op to be forward op or data op, but got: {node}"
        if node.name == 'multiref':
            warnings.warn(
                'Detected partition a multiref node. This will be skipped as system will automatically handle it.')
            return [node]
        if isinstance(node, IRPyFunc):
            warnings.warn(
                'Detected partition a python runtime function. This will be skipped as system will automatically handle it')
            return [node]

        # get partitioned sub-nodes
        fnodes = algo.instantiate(**config)
        assert fnodes is not None, f"Fail to partition node: {node} use algorithm and config: {config}"

        # set gradient
        valmaps: Dict[IRFullTensor, ValueMap] = dict()
        for t in node.inputs():
            if isinstance(t, IRSubTensor) and t.requires_grad:
                valmaps[t.parent] = ValueMap(t.grad.valmap)
        # set up consumers
        ctensors: Dict[IRFullTensor, List[IRSubTensor]] = dict()
        consumers: Dict[IRFullTensor, List[IRCell]] = dict()
        for fnode in fnodes:
            for itensor in fnode.inputs():
                if not isinstance(itensor, IRSubTensor): continue
                if not itensor.requires_grad: continue
                if itensor.parent not in ctensors:
                    ctensors[itensor.parent] = []
                    consumers[itensor.parent] = []
                ctensors[itensor.parent].append(itensor)
                consumers[itensor.parent].append(fnode)
        # set up gradient
        for fnode in fnodes:
            for itensor in fnode.inputs():
                if not isinstance(itensor, IRSubTensor): continue
                if not itensor.requires_grad: continue
                ftensor = itensor.parent
                # the [::-1] only makes the valuemap to grow with execution order
                cs = [c for c, t in zip(consumers[ftensor], ctensors[ftensor]) if t == itensor][::-1]
                valmap = valmaps[itensor.parent].map((cs.index(fnode), len(cs)))
                grad = ftensor.grad.select(itensor.indmap, valmap)
                itensor.grad = grad
            for otensor in fnode.outputs():
                if not isinstance(otensor, IRSubTensor): continue
                if not otensor.requires_grad:
                    grad = None
                else:
                    grad = otensor.parent.grad.select(otensor.indmap, (0,1))
                otensor.grad = grad

        # insert forward node
        fsegment: IRSegment = self.segment(node)
        for fnode in fnodes:
            if isinstance(node, IRFwOperation):
                fnode.recompute = node.recompute
            if isinstance(node.comment, str):
                fnode.comment = node.comment
            fnode.device = node.device
        fsegment.replace(node, fnodes)

        # insert backward node
        if isinstance(node.mirror, IRCell):
            bnodes = [fsegment.create_bwop(fnode) for fnode in fnodes[::-1]]
            assert isinstance(node.mirror, IRBpOperation)
            assert len(bnodes) == len(fnodes)
            for bnode in bnodes:
                bnode.device = node.device
            bsegment: IRSegment = fsegment.mirror
            bsegment.replace(node.mirror, bnodes)

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
            assert node.isfw(), "Only forward segment is allowed to assign devices"
            for subnode in node.nodes():
                self.assign(subnode, device)
        else:
            assert isinstance(node, (IRFwOperation, IRDataOperation)), \
                "Only forward operators and dataloader operators are allowed to assign devices"
            node.device = device
            if node.mirror is not None:
                node.mirror.device = device
        return True

    def reside(self, tensor: IRSubTensor, devices: Union[int, List[int]]):
        """
        Allocate an attribute tensor to devices.
        """
        assert tensor.is_attr(), f"Only support to set devices for graph attribute tensors"
        raise NotImplementedError("Not supported yet")

    ## Schedule Policy Primitives ##

    def sequential(self, nodes: Sequence[Union[FOp, Set[FOp]]]):
        """
        Scheduling Primitive: sequentially execute a list of nodes,
        or a list of concurrent nodes.

        Note there should be no dependency from a later node (set) to a previous node (set).

        Note in current implementation we don't check correctness

        Currently only support node (set) from a same device.

        @param nodes Sequence[Set[FOp]]: a sequence of operators or
            a sequence of concurrent operators. Note there should be no 
        """
        assert len(nodes) > 0
        concurrent_groups = [[node] if isinstance(node, IRCell) else node for node in nodes]
        segment: IRSegment = self.segment(concurrent_groups[0][0])
        idx = segment.index(nodes[0])
        for group in concurrent_groups[1:]:
            for node in group:
                assert segment.exist(node, flatten=False), "All nodes should in a same segment"
                # TODO: should check every node to see if they can be gathered based on that node
                segment.reorder(node, idx)

    def concurrent(self, nodes: Set[Union[FOp, Sequence[FOp]]]):
        """
        Scheduling Primitive: concurrently execut a list of nodes,
        or a list of sequential nodes.

        Note there should be no dependency from a node (set) to another node (set).

        Currently only suuport node (set) from different devices.

        @param nodes Set[Sequence[Fop]]: a set of operators or
            a set of sequential operators.
        """
        assert len(nodes) > 0
        seq_groups = [[node] if isinstance(node, IRCell) else node for node in nodes]
        segment: IRSegment = self.segment(seq_groups[0][0])
        idx = segment.index(nodes[0])
        for group in seq_groups[1:]:
            for node in group:
                assert segment.exist(node, flatten=False), "All nodes should in a same segment"
                # TODO: should check every node to see if they can be gathered based on that node
                segment.reorder(node, idx)

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

    def predef_sched(self, strategy):
        """!
        Set schedule plan for the execution.

        @param strategy IRScheduleStrategy: the schedule strategy instance
        """
        self._sched = strategy

    def _bind_schedule(self, schedplan):
        """
        Set schedule plan for the execution

        @param schedplan SchedulePlan
        """
        assert self._sched is None, "The graph is already binded with one schedule plan."
        self._sched = schedplan

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
        Group forward operators into sequential stages.
        The corresponding backward operators (if have) will also be grouped into stages
        Cross-stage dataflow will be limited to neighbor stages.
        This should be called before any operator partition.

        The transformation and temporal scheduling can only be applied within each stage.
        For example, after staging, user cannot schedule a (transformed) node 
        from one stage to another stage.

        Changes will be made:

        1). Identity creation:
            If a non-attribute tensor is produced / consumed not in
            neighbor stages, 
                e.g., 
                    stage 1: t1 = producer()
                    stage 2: ...
                    stage 3: xx = consume(t1)
                             xx = consume(t1)
                    stage 4: ...
                    stage 5: xx = consume(t1)
            then Identity nodes will be created for every device in stage2:
                    stage 1: t1 = producer()
                    stage 2: t2 = identity(t1)
                    stage 3: t3 = identity(t2)
                             xx = consume(t3)
                             xx = consume(t3)
                    stage 4: t4 = identity(t3)
                    stage 5: t5 = identity(t4)
                             xx = consume(t5)

        @param nodes Tuple[IRFwOperations]: the start forward node of each stage.
        @return None
        """
        assert all(isinstance(node, IRFwOperation) for node in nodes), \
            f"Find node is not IRFwOperation or IRDataOperation: {node}"
        assert all(node in self._nodes for node in nodes), \
            f"Exist node is not in graph nodes"
        starts = list(self._nodes.index(node) for node in nodes)
        assert len(starts) > 0

        # multiref (created by graph.auto_multiref) will be moved to the next stage (if possible) for optimization
        for sid in range(len(starts)):
            while starts[sid] > 0:
                node = self.node(starts[sid]-1)
                if node.name == 'multiref' or isinstance(node, IRGraphAnchor):
                    starts[sid] -= 1
                    continue
                break

        # adjust the start of the first stage to involve beginning operators
        for idx in range(starts[0]):
            node = self.node(idx)
            if isinstance(node, IRDataOperation):
                continue
            assert isinstance(node, IRFwOperation), \
                f"Expected nodes previous from the first stage are all IRFwOperation, but got {type(node)}"
            if node.name == 'multiref' or isinstance(node, IRPyFunc):
                pass
            else:
                warnings.warn(f'Detect a node: {node} that is previous from the first stage. Will be included inside the first stage')
            starts[0] = idx
            break

        last_fidx = 0
        for idx, node in enumerate(self._nodes):
            if not isinstance(node, IRBpOperation):
                last_fidx = idx
        
        fstages: List[List[IRCell]] = []
        bstages: List[List[IRCell]] = []
        for sid in range(len(starts)):
            begin = starts[sid]
            end = starts[sid+1] if sid != len(starts) - 1 else last_fidx + 1
            if begin >= end:
                warnings.warn(f"Detected stage {sid} doesn't have operators: [begin({begin}): end({end})). Skipped")
                continue
            fnodes = self._nodes[begin:end]
            assert all(isinstance(node, IRFwOperation) for node in fnodes), \
                f"find at least one nodes are not of IRFwOperation in the stage {sid}. They should be moved to the front"
            bnodes = [fnode.mirror for fnode in fnodes[::-1] if fnode.mirror is not None]
            fstages.append(fnodes)
            bstages = [bnodes] + bstages

        def get_sid(fnode: IRCell) -> Optional[int]:
            for idx, fnodes in enumerate(fstages):
                if fnode in fnodes:
                    return idx
            return None

        def insert_identity(tensor: IRSubTensor, sid: int) -> IRFwOperation:
            fwop = Identity(tensor)
            fwop.infer_shape()
            fwop.set_output(0, fwop.output(0).tosub())
            if tensor.requires_grad:
                fwop.output(0).parent.requires_grad = True
                # set input grad
                igrad = tensor.parent.grad.select(tensor.indmap, tensor.valmap)
                fwop.input(0).grad = igrad
                # set output grad
                otensor = fwop.output(0).parent
                ograd = otensor.grad.select(tensor.indmap, (0,1))
                fwop.output(0).grad = ograd
                # insert identity
                fidx = self.index(fstages[sid][0])
                self.finsert(fwop, fidx)
            else:
                fidx = self.index(fstages[sid][0])
                self.insert(fwop, fidx)
            # update stage op group
            fstages[sid].insert(0, fwop)
            if isinstance(fwop.mirror, IRCell):
                bstages[sid].append(fwop.mirror)
            return fwop

        # create identity op for cross-stage dataflow
        for ftensor in self.full_tensors():
            if ftensor.is_grad() or ftensor.is_attr(): continue
            if len(self.consumers(ftensor)) == 0: continue

            assert len(self.producers(ftensor)) <= 1, \
                "The staging interface should be called before any operator partition."
            ctensors = self.ctensors(ftensor)
            if len(self.ctensors(ftensor)) > 0:
                assert all(ctensor == ctensors[0] for ctensor in ctensors), (
                    "The staging interface should be called before any operator partition."
                )

            producer, ptensor = self.producers(ftensor)[0], self.ptensors(ftensor)[0]
            psid = get_sid(producer)
            # outside of stages, not consider
            if psid is None: continue 

            # group consumers into stages
            consumers = self.consumers(ftensor)
            csids = [get_sid(consumer) for consumer in consumers]
            buckets = [[] for _ in range(len(fstages))]
            for idx, csid in enumerate(csids):
                buckets[csid].append(consumers[idx])

            # go through each stage to generate identity operators
            out = ptensor
            end_sid = max(csids) + 1
            for sid in range(psid + 1, end_sid):
                # insert identity
                op = insert_identity(out, sid)
                out = op.output(0)
                # calculate gradient
                curr_valmap = ValueMap((0, 1))
                nconsumers = len(buckets[sid])
                fgrad = ftensor.grad
                for cidx, consumer in enumerate(buckets[sid]):
                    if fgrad is None:
                        grad = None
                    elif isinstance(fgrad, float):
                        assert fgrad == 1.0, "Detect a backward tensor, but gradient can only be 1.0"
                        grad = fgrad
                    else:
                        valmap = curr_valmap.map((0, 2)) if cidx != nconsumers - 1 else curr_valmap
                        grad = fgrad.select(ptensor.indmap, valmap)
                        curr_valmap = curr_valmap.map((1, 2)) if cidx != nconsumers - 1 else curr_valmap
                    # update forward consumer
                    idx = consumer.inputs().index(ptensor)
                    ptensor = consumer.input(idx)
                    with self.update(consumer) as consumer:
                        consumer.set_input(idx, out)
                        consumer.input(idx).grad = grad
                    # update backward
                    if isinstance(consumer.mirror, IRCell):
                        with self.update(consumer.mirror) as bconsumer:
                            idx = bconsumer.outputs().index(ptensor.grad)
                            bconsumer.set_output(idx,grad )

        # grouping into segment
        for sid in range(len(fstages)):
            self.group(fstages[sid])


    # ================= Other optimizations ==================

    def recompute(self, nodes: Union[IRSegment, List[IRFwOperation]]) -> bool:
        """!
        Recompute a set of nodes. The forward nodes will be assigned with a unique
        recompute group id. A forward not can not be recomputed in different recompute groups.

        @param nodes List[IRFwOperation]: nodes for a recompute group

        @return success boolean: always success
        """
        assert all(isinstance(node, IRFwOperation) for node in nodes) or isinstance(nodes, IRSegment), \
            "Require forward nodes or a single segment"

        if isinstance(nodes, IRSegment):
            assert nodes.isfw() and (not nodes.isbw()), "Only forward IRSegment can recompute"
            return self.recompute(nodes.nodes())
        
        else:
            segments = [self.segment(node) for node in nodes]
            assert all(segment == segments[0] for segment in segments), \
                "Cross-segment recompute is not allowed yet"
            recompute_group_id: int = IDGenerator().gen_cell_id()
            for fnode in nodes:
                if isinstance(fnode, IRGraphAnchor):
                    continue
                # pytorch limitation
                if all(not t.requires_grad for t in fnode.inputs() if isinstance(t, IRSubTensor) and (not t.is_attr())):
                    print(f"skipping recompute node: {fnode}\n\tbecause all its input tensors doesn't require grad.")
                    continue
                fnode.recompute = recompute_group_id
    
        return True

    # =================== Helpers ====================

    def auto_multiref(self):
        """
        Automatically partition and schedule multiref node.
        This requires to call after all transformation and
        scheduling.

        The policy is to partition and assign multiref 
        in the same way of its input producer
        """
        for node in self.nodes(flatten=True):
            if node.name == 'multiref':
                if len(node.device) != 0: continue
                segment: IRSegment = self.segment(node)
                ftensor = node.input(0).parent
                ptensors = segment.ptensors(ftensor)

                multirefs = []

                # use downstream consumers
                devtensors: Dict[int, List[IRSubTensor]] = dict()
                for tensor in node.outputs():
                    for ctensor in segment.ctensors(tensor.parent):
                        for devid in ctensor.device:
                            if devid not in devtensors:
                                devtensors[devid] = []
                            devtensors[devid].append(ctensor)
                devids = list(devtensors.keys())
                ctensors = [ts[0] for ts in devtensors.values()]
                for devid, ctensor in zip(devids, ctensors):
                    itensor = node.input(0).parent.select(ctensor.indmap, ctensor.valmap)
                    otensors = []
                    for otensor in node.outputs():
                        otensors.append(otensor.parent.select(ctensor.indmap, ctensor.valmap))
                    multiref = MultiRef(itensor, len(otensors))
                    for idx, otensor in enumerate(otensors):
                        multiref.set_output(idx, otensor)
                    multiref.device = devid
                    multirefs.append(multiref)
                
                # if no downstream consumers, use upstream producers
                if len(multirefs) == 0:
                    for ptensor in ptensors:
                        assert len(ptensor.device) > 0, \
                            "Auto Multiref requires its producer nodes assigned to devices"
                        for devid in ptensor.device:
                            outputs = []
                            for output in node.outputs():
                                outputs.append(output.parent.select(ptensor.indmap, ptensor.valmap))
                            multiref = MultiRef(ptensor, len(outputs))
                            for idx, otensor in enumerate(outputs):
                                multiref.set_output(idx, otensor)
                            multiref.device = devid
                            multirefs.append(multiref)
                
                # replace into graph
                fidx = self.remove(node)
                if node.mirror is not None:
                    self.remove(node.mirror)
                for multiref in multirefs[::-1]:
                    if node.mirror is not None:
                        self.finsert(multiref, fidx)
                    else:
                        self.insert(multiref, fidx)

    def dump(self, filename: str) -> None:
        """
        Dump the graph into pickled format

        @param filename str
        """
        # FIXME: dump doesn't support customized op
        class PicklingContextSave:
            def __enter__(self):
                IRObject.__getstate__ = IRObject.getstate_for_dump
            def __exit__(self, exc_type, exc_value, traceback):
                IRObject.__getstate__ = lambda self: self.__dict__.copy()

        with PicklingContextSave():
            with open(filename, 'wb') as f:
                save = (IDGenerator().get_states(), self)
                dill.dump(save, f)

    @staticmethod
    def load(filename: str):
        """
        Load the graph from pickled file.
        Note IDGenerator will also be reset to match with graph status

        @param filename str

        @return graph IRGraph
        """
        with open(filename, 'rb') as f:
            id_state, graph = dill.load(f)
        
        # recover IRGenerator
        IDGenerator().load_states(id_state)
        # recover cell
        def reset_node(segment: IRSegment):
            # input
            for t in segment.inputs():
                if isinstance(t, IRObject):
                    t.cell = segment
            # nodes
            for node in segment.nodes():
                for t in node.inputs() + node.outputs():
                    if isinstance(t, IRObject): 
                        t.cell = node
                # recursively recover segments
                if isinstance(node, IRSegment):
                    reset_node(node)
            # output
            for t in IRSegment.get_objects_from_complex(segment.outputs()):
                t.cell = segment
        
        reset_node(graph)
        return graph
