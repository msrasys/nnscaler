from contextlib import contextmanager
from typing import Dict, Union, List, Optional, Set, Tuple, Any, Callable
import numpy as np

from cube.ir.tensor import IRFullTensor, IRSubTensor, ValueMap
from cube.ir.cten import IRTensor, IRCell, IRObject
from cube.ir.operator import IRFwOperation, IRBpOperation
from cube.ir.adapter import IRAdapter

from cube.graph.function.function import MultiRef
from cube.graph.function.pyfunc import IRPyFunc


class CellPosition:

    def __init__(self, indices: Tuple[int]):
        assert all(isinstance(idx, int) for idx in indices) and len(indices) > 0
        self.indices = tuple(indices)

    def __hash__(self) -> int:
        return hash(self.indices)

    def __eq__(self, other: object) -> bool:
        assert isinstance(other, CellPosition), "Cannot compare with non-GraphIndex object"
        return self.indices == other.indices
    
    def __lt__(self, other: object) -> bool:
        assert isinstance(other, CellPosition), "Cannot compare with non-GraphIndex object"
        if len(self.indices) < len(other.indices):
            return True
        if len(self.indices) > len(other.indices):
            return False
        for lidx, ridx in zip(self.indices, other.indices):
            if lidx >= ridx:
                return False
        return True

    def __le__(self, other: object) -> bool:
        return self < other or self == other

    def __gt__(self, other: object) -> bool:
        return not self <= other

    def __ge__(self, other: object) -> bool:
        return not self < other

    def __sub__(self, offset: int):
        assert isinstance(offset, int)
        indices = list(self.indices)
        indices[-1] -= offset
        return CellPosition(indices)

    def __add__(self, offset: int):
        assert isinstance(offset, int)
        indices = list(self.indices)
        indices[-1] += offset
        return CellPosition(indices)

    def __getitem__(self, idx: int) -> int:
        return self.indices[idx]

    def __len__(self) -> int:
        return len(self.indices)

    def __repr__(self) -> str:
        return repr(self.indices)


class IRSegment(IRCell):
    """
    A distributed sub-graph representing a piece of workload in parent IRGraph

    Input/output can be complex data type of Dict, List, Tuple on IRObjects

    Once the segment is generated, its input and output will be fixed.
    Inserting and removing nodes that could change input/output are not allowed.
    """

    def __init__(self, nodes: List[IRCell], inputs: List[IRTensor], outputs: List[IRSubTensor], name='segment'):
        super().__init__(name, '', len(inputs), len(outputs), init_outputs=False)

        self._nodes: List[IRCell] = []

        for idx, val in enumerate(inputs):
            self.set_input(idx, val)
        for idx, val in enumerate(outputs):
            self.set_output(idx, val)

        # full-tensor / sub-tensor mapping
        self._ftensors: Set[IRFullTensor] = set()
        self._producers: Dict[IRFullTensor, List[IRCell]] = dict()
        self._consumers: Dict[IRFullTensor, List[IRCell]] = dict()
        self._ptensors: Dict[IRFullTensor, List[IRSubTensor]] = dict()
        self._ctensors: Dict[IRFullTensor, List[IRSubTensor]] = dict()

        # attributes
        self._attributes: Set[IRFullTensor] = set()

        for node in nodes:
            self.insert(node, self.nnodes)

        self._dispatch_cached: Dict[int, IRSegment] = {}

        # self.reset_dependency()

    def isfw(self) -> bool:
        return all(n.isfw() for n in self._nodes)
        # return self._have_forward

    def full_tensors(self) -> Tuple[IRFullTensor]:
        """
        Get all full tensors of this graph.
        Note the full tensor inside the node will not be returned.

        @return ftensors List[IRFullTensor]
        """
        return tuple(t for t in self._ftensors if isinstance(t, IRFullTensor))

    def attributes(self) -> Tuple[IRFullTensor]:
        """
        Get al full tensor attributes of this graph
        Note the full tensor inside the node will not be returned.

        @return ftensors List[IRFullTensor]
        """
        return tuple(self._attributes)

    def reset_dependency(self):
        """
        Reset the node dataflow dependency
        
        Note all the predefined control dependencies will be removed.
        TODO: adapter dependency is not set
        """
        for node in self._nodes:
            node.clear_predecessor()
            node.clear_successor()
        # TODO: adapter dependency not set
        for ftensor in self._ftensors:
            for ptensor, producer in zip(self.ptensors(ftensor), self.producers(ftensor)):
                for ctensor, consumer in zip(self.ctensors(ftensor), self.consumers(ftensor)):
                    if ptensor.overlap(ctensor):
                        pidx = producer.outputs().index(ptensor)
                        cidx = consumer.inputs().index(ctensor)
                        producer.add_successor(pidx, consumer)
                        consumer.add_predecessor(cidx, producer)
                # set mirror as control dependency
                if producer.mirror is not None and isinstance(producer, IRFwOperation):
                    producer.add_successor(-1, producer.mirror)
                    producer.mirror.add_predecessor(-1, producer)
        # sub segments
        for segment in self._nodes:
            if isinstance(segment, IRSegment):
                segment.reset_dependency()

    # ========================= Basic Graph access =======================

    @property
    def device(self) -> List[int]:
        devices = set()
        for node in self._nodes:
            devices.update(node.device)
        devices = list(devices)
        devices.sort()
        return devices

    @property
    def nnodes(self) -> int:
        """
        Get total node number

        @return number int: the number of nodes
        """
        return len(self._nodes)

    def nodes(self, flatten = False) -> Tuple[IRCell]:
        """
        Get all the nodes.

        @param flatten bool: Flat the segment to get all the nested cells

        @return nodes List[IRCell]: all the nodes
        """
        if not flatten:
            return tuple(self._nodes)
        nodes = []
        for node in self._nodes:
            if not isinstance(node, IRSegment):
                nodes.append(node)
            else:
                nodes += list(node.nodes(flatten))
        return tuple(nodes)

    def node(self, index: Union[int, CellPosition]) -> IRCell:
        """
        Get node at position index

        @param index Union[int, CellPosition]: the node index

        @return node IRCell: the node.
        """
        pos = CellPosition((index,)) if isinstance(index, int) else index
        assert isinstance(pos, CellPosition), "Expect index to be int or CellPosition"
        node = self
        for idx in pos.indices:
            assert isinstance(node, IRSegment), "idx applies on a non-segment node"
            node = node._nodes[idx]
        return node

    def index(self, node: IRCell) -> CellPosition:
        """
        Get node index. The dispatched node (e.g., IRAdapter, IRSegment) 
        will return the index to its un-dispatched node

        @param node IRCell: the queried node

        @return index int: the index
        """
        assert isinstance(node, IRCell)
        if node in self._nodes:
            return CellPosition((self._nodes.index(node),))
        for idx, segment in enumerate(self._nodes):
            if isinstance(segment, IRSegment):
                if segment.exist(node):
                    index = segment.index(node)
                    return CellPosition((idx,) + index.indices)
        raise KeyError(f"The queried node: {node} not in the graph")

    def segment(self, node: IRCell) -> IRCell:
        """
        Get the lowest segment that constains the node

        @param node IRCell: the queried node

        @return segment IRSegment
        """
        assert isinstance(node, IRCell), f"Expected IRCell, but got {node}"
        index = self.index(node)
        if len(index) == 1:
            return self
        else:
            return self.node(CellPosition(index.indices[:-1]))

    def producers(self, ftensor: IRFullTensor) -> Tuple[IRCell]:
        """
        Get producers of ftensor in execution order in this graph

        @param ftensor IRFullTensor: the queried full tensor.

        @return subtensors Tuple[IRSubTensor]: the producers.
        """
        assert ftensor in self._producers, f"{ftensor} is not in the graph"
        return tuple(self._producers[ftensor])

    def consumers(self, ftensor: IRFullTensor) -> Tuple[IRCell]:
        """
        Get consumers of ftensor in execution order in this graph

        @param ftensor IRFullTensor: the queried full tensor.

        @return subtensors Tuple[IRCell]: theconsumers
        """
        assert ftensor in self._consumers, f"{ftensor} is not in the graph"
        return tuple(self._consumers[ftensor])

    def ptensors(self, ftensor: IRFullTensor) -> Tuple[IRSubTensor]:
        """
        Get consumed sub-tensors of ftensor in execution order in this graph

        @param ftensor IRFullTensor: the queried full tensor.

        @return subtensors Tuple[IRSubTensor]: the consumed subtensors.
        """
        assert ftensor in self._ptensors, f"{ftensor} is not in the graph"
        return tuple(self._ptensors[ftensor])

    def ctensors(self, ftensor: IRFullTensor) -> Tuple[IRSubTensor]:
        """
        Get consumed sub-tensors of ftensor in execution order in this graph

        @param ftensor IRFullTensor: the queried full tensor.

        @return subtensors Tuple[IRSubTensor]: the consumed subtensors.
        """
        assert ftensor in self._ctensors, f"{ftensor} is not in the graph"
        return tuple(self._ctensors[ftensor])

    def infer_grad(self, ftensor: IRFullTensor) -> None:
        """
        Set gradient on sub-tensors of a fulltensor

        Note this can only be called when no operator transformation is
        applied for this graph.

        @param ftensor IRFullTensor: the full tensor.

        @return None: gradient are set to producer/consumer tensor's .grad
        """
        fgrad = ftensor.grad
        # set for producer
        assert len(self.producers(ftensor)) <= 1, (
            f"grad can only be set when no transformation is applied but got:\n"
            f"{self.debug_tensor_map_str(ftensor)}"
        )
        for ptensor, producer in zip(self.ptensors(ftensor), self.producers(ftensor)):
            # filter out non-autograd operators of IRPyFunc
            if isinstance(producer, IRPyFunc): continue
            idx = producer.outputs().index(ptensor)
            if fgrad is None:
                grad = None
            else:
                grad = fgrad.select(ptensor.indmap, (0, 1))
            producer.output(idx).grad = grad
        # set for consumers
        ctensors = self.ctensors(ftensor)
        if len(ctensors) > 0:
            assert all(ctensor == ctensors[0] for ctensor in ctensors), (
                f"grad can only be set when no transformation is applied but got:\n"
                f"{self.debug_tensor_map_str(ftensor)}"
            )
        curr_valmap = ValueMap((0, 1))

        # filter out non-autograd operators of IRPyFunc
        consumers, ctensors = [], []
        for ctensor, consumer in zip(self.ctensors(ftensor), self.consumers(ftensor)):
            if isinstance(consumer, IRPyFunc): continue
            consumers.append(consumer)
            ctensors.append(ctensor)

        nconsumers = len(consumers)
        for cidx, (ctensor, consumer) in enumerate(zip(ctensors, consumers)):
            idx = consumer.inputs().index(ctensor)
            if fgrad is None:
                grad = None
            else:
                valmap = curr_valmap.map((0, 2)) if cidx != nconsumers - 1 else curr_valmap
                grad = fgrad.select(ctensor.indmap, valmap)
                curr_valmap = curr_valmap.map((1, 2)) if cidx != nconsumers - 1 else curr_valmap
            consumer.input(idx).grad = grad

    def debug_tensor_map_str(self, ftensor: Optional[IRFullTensor] = None) -> str:
        dscp : str = ''
        ftensors = [ftensor] if ftensor is not None else self._ftensors
        for ftensor in ftensors:
            dscp += f'====\nFull Tensor: {ftensor}\n'
            dscp += f'Producers:\n'
            for producer in self._producers[ftensor]:
                dscp += f'\t{producer}\n'
            dscp += f'Consumers:\n'
            for consumer in self._consumers[ftensor]:
                dscp += f'\t{consumer}\n'
        return dscp

    def create_bwop(self, fwop: IRFwOperation) -> Union[IRBpOperation, IRBpOperation]:
        """
        Create dummy backward operator for given forward operator.
        This assumes input/output tensors of fwop have been set by correct gradient tensors.

        @param fwop IRFwOperation: forward operation

        @return bwop IRBpOperation: the created backward operation
        """
        assert isinstance(fwop, (IRFwOperation, IRSegment)), "Expected IRFwOperation"
        fins = [t for t in fwop.inputs() if isinstance(t, IRSubTensor)]
        fous = [t for t in fwop.outputs() if isinstance(t, IRSubTensor)]
        igrads = [t.grad if t.requires_grad else None for t in fins]
        ograds = [t.grad if t.requires_grad else None for t in fous]
        if isinstance(fwop, IRFwOperation):
            bwop = IRBpOperation(ograds, igrads)
        else:
            bnodes = [fnode.mirror for fnode in fwop.nodes() if fnode.mirror is not None][::-1]
            bwop = IRSegment(bnodes, ograds, igrads)
        IRCell.make_pair(fwop, bwop)
        return bwop

    # ====================== Basic Graph manipulations ======================

    def _add_ftensor(self, ftensor: IRObject):
        """
        Add a full tensor in segment if the segment doesn't have the tensor.
        """
        assert isinstance(ftensor, IRObject)
        if ftensor not in self._ftensors:
            self._ftensors.add(ftensor)
            self._producers[ftensor] = []
            self._consumers[ftensor] = []
            self._ptensors[ftensor] = []
            self._ctensors[ftensor] = []
        if ftensor.is_attr():
            self._attributes.add(ftensor)
    
    def _remove_ftensor(self, ftensor: IRObject):
        """
        Remove a full tensor in segment
        """
        assert isinstance(ftensor, IRObject)
        if ftensor in self._ftensors:
            self._ftensors.remove(ftensor)
            del self._producers[ftensor]
            del self._consumers[ftensor]
            del self._ptensors[ftensor]
            del self._ctensors[ftensor]
        if ftensor.is_attr() and ftensor in self._attributes:
            self._attributes.remove(ftensor)

    def _reorder_producer_consumer(self):
        """
        Re-order producers and consumers for each full tensor to match
        with the ordering of nodes.

        Note sub-segment will also be reordered.
        """
        # clear up
        self._ftensors, self._attributes = set(), set()
        self._producers, self._ptensors = dict(), dict()
        self._consumers, self._ctensors = dict(), dict()
        # set producer and consumer
        for node in self._nodes:
            if isinstance(node, IRAdapter): continue
            itensors = set(t for t in node.inputs() if isinstance(t, IRObject))
            for itensor in itensors:
                ftensor = itensor.parent
                self._add_ftensor(ftensor)
                self._consumers[ftensor].append(node)
                self._ctensors[ftensor].append(itensor)
            otensors = set(t for t in node.outputs() if isinstance(t, IRObject))
            for otensor in otensors:
                ftensor = otensor.parent
                self._add_ftensor(ftensor)
                self._producers[ftensor].append(node)
                self._ptensors[ftensor].append(otensor)
            if isinstance(node, IRSegment):
                node._reorder_producer_consumer()

    def insert(self, node: IRCell, index: Union[int, CellPosition]):
        """
        Insert a node at index.

        TODO: dataflow dependency update
        TODO: input / output check

        @param node IRCell: the inserted node
        @param index int: the index

        """
        pos = CellPosition((index,)) if isinstance(index, int) else index
        assert isinstance(pos, CellPosition), "Expect index to be int or CellPosition"

        if len(pos) == 1:
            index = pos[0]
            # insert node
            self._nodes.insert(index, node)
            # update producer and consumer
            if isinstance(node, IRAdapter): return
            # consumer
            itensors = set(t for t in node.inputs() if isinstance(t, IRObject))
            for itensor in itensors:
                ftensor = itensor.parent
                self._add_ftensor(ftensor)
                self._consumers[ftensor].append(node)
                self._ctensors[ftensor].append(itensor)
            # producer
            otensors = set(t for t in node.outputs() if isinstance(t, IRObject))
            for otensor in otensors:
                ftensor = otensor.parent
                self._add_ftensor(ftensor)
                self._producers[ftensor].append(node)
                self._ptensors[ftensor].append(otensor)
        else:
            segment = self._nodes[pos[0]]
            assert isinstance(segment, IRSegment), "Expected IRSegment"
            pos = CellPosition(pos.indices[1:])
            segment.insert(node, pos)

    def remove(self, node: IRCell, _pos: Union[int, CellPosition] = None) -> CellPosition:
        """
        Remove a node at index

        # TODO: check input and output

        @param node IRCell: the removed node
        @param _pos Optional[Union[int, CellPosition]: help to save cost if provide node position.
        
        @return index CellPosition: the removed index
        """
        pos = self.index(node) if _pos is None else _pos
        assert self.node(pos) == node, \
            f"posititon doesn't not match with node:\n\t{node}\nGot:\n\t{self.node(pos)}"

        if len(pos.indices) == 1:
            index = pos[0]
            # remove
            self._nodes.pop(index)
            # update producer and consumer
            if isinstance(node, IRAdapter): return pos
            # consumer
            itensors = set(t for t in node.inputs() if isinstance(t, IRObject))
            for itensor in itensors:
                ftensor = itensor.parent
                idx = self._consumers[ftensor].index(node)
                self._consumers[ftensor].pop(idx)
                self._ctensors[ftensor].pop(idx)
                if len(self._consumers[ftensor]) == 0 and len(self._producers[ftensor]) == 0:
                    self._remove_ftensor(ftensor)
            # producer
            otensors = set(t for t in node.outputs() if isinstance(t, IRObject))
            for otensor in otensors:
                ftensor = otensor.parent
                idx = self._producers[ftensor].index(node)
                self._producers[ftensor].pop(idx)
                self._ptensors[ftensor].pop(idx)
                if len(self._consumers[ftensor]) == 0 and len(self._producers[ftensor]) == 0:
                    self._remove_ftensor(ftensor)
        else:
            segment = self._nodes[pos[0]]
            assert isinstance(segment, IRSegment)
            segment.remove(node, _pos=CellPosition(pos.indices[1:]))

        return pos

    def replace(self, node: IRCell, new_nodes: List[IRCell]) -> int:
        """
        Replace one node by multiple nodes

        # TODO: check input and output

        @param node IRCell: the replaced node
        @param new_nodes List[IRCell]: the nodes to be inserted.

        @return index int: the replaced node index
        """
        idx = self.remove(node)
        for new_node in new_nodes[::-1]:
            self.insert(new_node, idx)
        return idx

    def reorder(self, node: IRCell, index: int):
        """
        Reorder an existing node to the index.

        @param node IRCell: the node in this segment, not considering inner segments.
        @param index int: the index is under the view of nodes ordering before this call.

        @return None
        """
        prev_index = self._nodes.index(node)
        self.remove(node, prev_index)
        index = index if prev_index >= index else index - 1
        self.insert(index, node)

    @contextmanager
    def update(self, node):
        """
        Update a node. Note the related change in backward operator 
        will not be automatically updated.
    
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

    def exist(self, node: IRCell, flatten: bool = True) -> bool:
        """
        Check if the node is in this graph

        @param node IRCell: the queried node

        @return exsit bool: True if exist otherwise False
        """
        if node in self._nodes:
            return True
        if flatten:
            for segment in self._nodes:
                if not isinstance(segment, IRSegment): continue
                if segment.exist(node, flatten):
                    return True
        return False

    def select(self, name: Optional[str] = None, ntype: Optional[IRCell] = None, flatten: bool = True) -> List[IRCell]:
        """
        Select all the nodes (including nodes in sub-segment) that
        satisfy the condition.

        @param name Optional[str]: the node name
        @param ntype Optional[Type]: the node type
        @param flatten bool: whether to flatten the segment to nodes. (Default True)

        @return nodes List[IRCell]: the nodes that have the name.
        """
        nodes = []
        for node in self.nodes(flatten=flatten):
            if name is not None:
                if node.name != name:
                    continue
            if ntype is not None:
                if not isinstance(node, ntype):
                    continue
            nodes.append(node)
        return nodes

    def finsert(self, fwop: IRFwOperation, index: Union[int, CellPosition]) -> IRFwOperation:
        """
        Insert a forward node and create its backward.
        The created backward operator will be happen right before
        the backward of fwop's previous forward node

        This requires the segment has its backward segment
        This assumes inputs/outputs tensors of fwop have been set with correct gradient

        @param fwop IRFwOperation: forward node
        @param index Union[int, CellPosition]: inserted position

        @return node IRFwOperation: the node itself
        """
        assert isinstance(fwop, IRFwOperation), "Only allow insert an IRFwOperation"
        pos = CellPosition((index,)) if isinstance(index, int) else index
        assert isinstance(pos, CellPosition), "Expect index to be int or CellPosition"
    
        index = pos.indices[-1]
        fsegment = self if len(pos) == 1 else self.node(CellPosition(pos.indices[1:]))
        fsegment.insert(fwop, index)
        # create backward
        bwop = fsegment.create_bwop(fwop)
        bwop.device = fwop.device
        # insert backward
        assert fsegment.mirror is not None, "Missing backward segment"
        bsegment: IRSegment = fsegment.mirror
        bidx = CellPosition((bsegment.nnodes,))
        for idx in range(index - 1, -1, -1):
            prev_fnode = fsegment.node(idx)
            if prev_fnode.mirror is not None:
                bidx = bsegment.index(prev_fnode.mirror)
                break
        bsegment.insert(bwop, bidx)
        return fwop

    # ===================== Advance Graph manipulations ==================

    def multiref(self, ftensor: IRFullTensor, node_groups: List[List[IRFwOperation]]) -> IRFwOperation:
        """
        Add multiref to separate forward nodes that consume a same tensor into different tensor alias.
        This should be called before any graph transformation.

        Operators in a group can only be partitioned by a same tensor split strategy.
        The created multiref operator will be partitioned automatically when generating
        tensor adapters.

        @param tensor IRSubTensor: tensor.
        @param node_groups List[List[IRFwOperation]]:
            operators that take the tensor as input.
        
        @return multiref IRFwOperation: the inserted multiref operator.
        """
        assert ftensor in self._ftensors, f"tensor: {ftensor} not in this graph."
        # check no transformation
        if len(self.consumers(ftensor)) <= 1: return
        assert not ftensor.is_grad(), f"graph.multiref can only be applied on a non-gradient full tensor."
        assert len(set(self.ctensors(ftensor))) == 1, \
            f"Detected happened graph transformation. This interfacee should be called before graph transformation."
        # check completeness
        consumers = set()
        for nodes in node_groups:
            consumers.update(nodes)
        assert consumers == set(self.consumers(ftensor)), f"some consumer(s) are not in node_groups"
        # create new full tensors
        tensor = self.ctensors(ftensor)[0]
        ftensors: List[IRSubTensor] = [ftensor.like() for _ in node_groups]
        otensors: List[IRSubTensor] = [ft.select(tensor.indmap, tensor.valmap) for ft in ftensors]
        # create multiref
        multiref = MultiRef(tensor, len(node_groups))
        for idx, otensor in enumerate(otensors):
            multiref.set_output(idx, otensor)
        # setup gradient
        if tensor.requires_grad:
            multiref.input(0).grad = tensor.parent.grad.select(tensor.indmap, (0, 1))
            for idx, output in enumerate(multiref.outputs()):
                output.grad = ftensors[idx].grad.select(tensor.indmap, (0,1))
        # insert multiref
        if len(self.producers(ftensor)) == 0:
            fidx = min(self.index(consumer) for consumer in self.consumers(ftensor))
        else:
            fidx = max(self.index(prod) for prod in self.producers(ftensor)) + 1
        if ftensor.requires_grad:
            self.finsert(multiref, fidx)
        else:
            self.insert(multiref, fidx)
        # update forward / backward consumer
        for otensor, nodes in zip(otensors, node_groups):
            for idx, node in enumerate(nodes):
                fidx = node.inputs().index(tensor)
                grad = node.input(fidx).grad
                with self.update(node):
                    node.set_input(fidx, otensor)
                if tensor.requires_grad:
                    node.input(fidx).grad = otensor.parent.grad.select(otensor.indmap, (idx, len(nodes)))
                    with self.mirror.update(node.mirror) as bnode:
                        bidx = bnode.outputs().index(grad)
                        bnode.set_output(bidx, node.input(bidx).grad)
        return multiref

    def single_consume(self, one_for_all: bool = True):
        """
        Transform graph to make each non-attribute tensor has up to
        one consumer. Multiref nodes will be inserted. The API is useful 
        for cases like inference, where different consumers are partitioned
        with different tensor dimensions.

        This should be called before any graph transformation.

        e.g., original graph:

            t = producer(xx)
            ...
            xx = consumer1(t)
            ...
            xx = consumer2(t)
            ...
            xx = consumer3(t)
            ...

        If one_for_all is True, will be:

            t = producer(xx)
            t1, t2, t3 = multiref(t)
            ...
            xx = consumer1(t1)
            ...
            xx = consumer2(t2)
            ...
            xx = consumer3(t3)

        Otherwise:

            t = producer(xx)
            ...
            t1, t2 = multiref(t)
            xx = consumer1(t1)
            ...
            t3, t4 = multiref(t2)
            xx = consumer2(t3)
            ...
            xx = consumer3(t4)


        @param one_for_all bool: If True,
        one single multiref node will be created for each fulltensor. Otherwise,
        if a fulltensor has K consumers, then K-1 multiref nodes will be created.

        @return None
        """
        consumers: Dict[IRFullTensor, List[IRCell]] = dict()
        producers: Dict[IRFullTensor, IRCell] = dict()
        if not one_for_all:
            for node in self.nodes():
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
                        multiref = MultiRef(reftensor, 2)
                        for idx, itensor in enumerate(itensors):
                            multiref.set_output(idx, itensor)
                        multiref.infer_shape()
                        # insert multiref right before the consumor
                        idx = self.index(consumer)
                        # require backward
                        if any(itensor.requires_grad for itensor in node.inputs()):
                            self.finsert(multiref, idx)
                        else:
                            self.insert(multiref, idx)
                        ctensor, reftensor = itensors
                    else:
                        # the last consumer doesn't need multiref
                        ctensor = reftensor
                    # update consumer
                    while ftensor in consumer.inputs():
                        idx = consumer.inputs().index(ftensor)
                        consumer.set_input(idx, ctensor)
        else:
            for node in self.nodes():
                ftensors = set()
                for ftensor in node.inputs():
                    # remove redundant tensors within an operator
                    if isinstance(ftensor, IRFullTensor) and ftensor._id not in ftensors:
                        ftensors.add(ftensor._id)
                        if ftensor not in consumers:
                            consumers[ftensor] = []
                        consumers[ftensor].append(node)
                for ftensor in node.outputs():
                    if isinstance(ftensor, IRFullTensor):
                        producers[ftensor] = node
            for ftensor, cnodes in consumers.items():
                if len(cnodes) == 1 or ftensor.is_attr(): continue
                itensors = [ftensor.like() for _ in range(len(cnodes))]
                for itensor, consumer in zip(itensors, cnodes):
                    while ftensor in consumer.inputs():
                        idx = consumer.inputs().index(ftensor)
                        consumer.set_input(idx, itensor)
                # create and insert multiref operation
                multiref = MultiRef(ftensor, len(cnodes))
                for idx, itensor in enumerate(itensors):
                    multiref.set_output(idx, itensor)
                multiref.infer_shape()
                idx = self.index(producers[ftensor]) + 1 if ftensor in producers else 0
                # idx = nodes.index(cnodes[0])
                if any(itensor.requires_grad for itensor in node.inputs()):
                    self.finsert(multiref, idx)
                else:
                    self.insert(multiref, idx)

    # ====================== Graph Generations ============================
    
    @staticmethod
    def get_inputs(nodes: List[IRCell]):
        """
        Get all the input tensors that are required by nodes.

        @param nodes List[IRCell]: the nodes
        
        @return inputs List[IRTensor]: the input tensors
        """
        all_outputs = list()
        for node in nodes:
            all_outputs.extend(node.outputs())
        inputs = list()
        for node in nodes:
            for input in node.inputs():
                if isinstance(input, IRTensor):
                    if input not in all_outputs:
                        if input not in inputs:
                            inputs.append(input)
        return inputs

    @staticmethod
    def get_outputs(nodes: List[IRCell]):
        """
        Get tensors that are produced but not consumed by nodes

        As long as the tensor is consumed in by the nodes, it will
        not be in the output. A tensor will not appear as output if it
        is double-consumed both outside and inside the nodes.

        @param nodes List[IRCell]: the nodes

        @return outputs List[IRTensor]: the output tensors
        """
        all_inputs = list()
        for node in nodes:
            all_inputs.extend(node.inputs())
        outputs = list()
        for node in nodes:
            for output in node.outputs():
                # not consumed tensor
                if isinstance(output, IRTensor):
                    if output not in all_inputs:
                        if output not in outputs:
                            outputs.append(output)
                            continue
        return outputs

    def create_segment(self, nodes: List[IRCell]) -> IRCell:
        """!
        Create a segment with part of the nodes. 
        This only return the created segment wihout modifying the graph.

        @param nodes List[IRCell]: the subset nodes of this graph

        @return segment IRSegment: the grouped segment. 
        """
        segment = self
        segment_inputs = IRSegment.get_objects_from_complex(segment.inputs())
        segment_outputs = IRSegment.get_objects_from_complex(segment.outputs())

        # segments: List[IRSegment] = [self.segment(node) for node in nodes]
        # assert len(set(segments)) == 1, "Cross segment hierarchy grouping is not allowed"
        # segment = segments[0]

        inputs, outputs = set(), set()

        # go through adapters
        adapter_ins: Dict[IRSubTensor, Set[int]] = dict()
        adapter_ous: Dict[IRSubTensor, Set[int]] = dict()
        for adapter in nodes:
            if not isinstance(adapter, IRAdapter):
                continue
            for itensor in adapter.inputs():
                if not isinstance(itensor, IRSubTensor): continue
                if itensor not in adapter_ins:
                    adapter_ins[itensor] = set()
                adapter_ins[itensor].update(itensor.device)
                # producers can from out side node
                producers = []
                for ptensor, prod in zip(segment.ptensors(itensor.parent), segment.producers(itensor.parent)):
                    if ptensor == itensor and set(itensor.device).issubset(set(prod.device)):
                        producers.append(prod)
                if not any(p in nodes for p in producers):
                    inputs.add(itensor)
            for otensor in adapter.outputs():
                if not isinstance(otensor, IRSubTensor): continue
                if otensor not in adapter_ous:
                    adapter_ous[otensor] = set()
                adapter_ous[otensor].update(otensor.device)
                consumers = []
                for ctensor, cons in zip(segment.ctensors(otensor.parent), segment.consumers(otensor.parent)):
                    if ctensor == otensor and set(otensor.device).issubset(set(cons.device)):
                        consumers.append(cons)
                if not any(c in nodes for c in consumers):
                    outputs.add(otensor)

        # go through non-adapter nodes
        for node in nodes:
            if isinstance(node, IRAdapter):
                assert node.differentiable, \
                    "Non-differentiable IRAdapter is not allowed to be grouped"
                continue
            # update inputs
            itensors = [t for t in node.inputs() if isinstance(t, IRObject)]
            for itensor in itensors:
                ftensor = itensor.parent
                if itensor.is_attr(): continue
                # from inside adapters
                if itensor in adapter_ous:
                    if len(node.device) > 0 and set(itensor.device).issubset(adapter_ous[itensor]):
                        continue
                # from segment inputs
                if any(t.overlap(itensor) for t in segment_inputs if isinstance(t, IRObject)):
                    inputs.add(itensor)
                    continue
                # from outside producers
                producers, ptensors = segment.producers(ftensor), segment.ptensors(ftensor)
                producers = [p for p, t in zip(producers, ptensors) if t == itensor]
                if len(itensor.device) > 0:
                    producers = [p for p in producers if set(itensor.device).issubset(set(p.device))]
                # from graph inputs or outside adapter (no producer)
                if len(producers) == 0 or any(p not in nodes for p in producers):
                    inputs.add(itensor)
                    continue
            # update outputs
            otensors = [t for t in node.outputs() if isinstance(t, IRObject)]
            for otensor in otensors:
                ftensor = otensor.parent
                if otensor.is_attr(): continue
                # from inside adapters
                if otensor in adapter_ins:
                    if len(node.device) > 0 and set(otensor.device).issubset(adapter_ins[otensor]):
                        continue
                # from segment outputs
                if any(t.overlap(otensor) for t in segment_outputs if isinstance(t, IRObject)):
                    outputs.add(otensor)
                    continue
                # loss doesn't have consumers
                if len(segment.consumers(ftensor)) == 0:
                    if isinstance(ftensor, IRFullTensor) and ftensor.is_loss():
                        outputs.add(otensor)
                    continue
                # for outside consumers
                consumers, ctensors = segment.consumers(ftensor), segment.ctensors(ftensor)
                consumers = [c for c, t in zip(consumers, ctensors) if t == otensor]
                if len(otensor.device) > 0:
                    consumers = [c for c in consumers if set(otensor.device).issubset(set(c.device))]
                # for adapter (no consumer)
                if len(consumers) == 0 or any(c not in nodes for c in consumers):
                    outputs.add(otensor)
                    continue
        
        def order(tensors: Set[IRObject]) -> Tuple[IRObject]:
            """Reorder by logical tensor id. Temporally necessary for pipeline scheduling"""
            tensors = list(tensors)
            tids = np.array([t.parent.tid for t in tensors])
            indices = np.argsort(tids)
            return tuple(tensors[idx] for idx in indices)

        segment = IRSegment(nodes, order(inputs), order(outputs))
        return segment

    def dispatch(self, devid: int, _gen_mirror: bool = True) -> Optional[IRCell]:
        """
        Instantiate the segement to a specific device.

        @param devid int: the target device

        @return segment IRSegment: the instantiated segment
        """
        if devid not in self.device:
            return None
        if len(self.device) == 1 and self.device == [devid]:
            return self
        if devid in self._dispatch_cached:
            return self._dispatch_cached[devid]
        # inputs, outputs, nodes = [], [], []
        inputs, outputs, nodes = self.inputs(), self.outputs(), []
        for node in self._nodes:
            if devid in node.device:
                nodes.append(node.dispatch(devid))
                # for itensor in node.inputs():
                #     if itensor in self._inputs and itensor not in inputs:
                #         inputs.append(itensor)
                # for otensor in node.outputs():
                #     if otensor in self._outputs and otensor not in outputs:
                #         outputs.append(otensor)

        def order(tensors: Set[IRObject]) -> Tuple[IRObject]:
            """Reorder by logical tensor id. Temporally necessary for pipeline scheduling"""
            tensors = list(tensors)
            tids = np.array([t.parent.tid for t in tensors])
            indices = np.argsort(tids)
            return tuple(tensors[idx] for idx in indices)
        
        if self.isfw():
            inputs, outputs = order(inputs), order(outputs)

        segment = IRSegment(nodes, inputs, outputs, self.name)
        segment._id = self.cid
        if _gen_mirror and self.mirror is not None:
            msegment = self.mirror.dispatch(devid, _gen_mirror=False)
            IRCell.make_pair(segment, msegment)
        self._dispatch_cached[devid] = segment
        return segment


    # ========================== Graph Visualize ================================

    def __repr__(self):
        fw = 'f' if self.isfw() else 'b'
        inputs = tuple(t for t in self.inputs() if isinstance(t, IRObject) and not t.is_attr())
        if self.isfw():
            dscp = f"{fw}Graph{self.cid}-{self.device}(inputs={inputs}, outputs={self.outputs()})"
        else:
            dscp = f"{fw}Graph{self.cid}-{self.device}(fGraph{self.mirror.cid}, inputs={inputs}, outputs={self.outputs()})"
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

    @staticmethod
    def get_objects_from_complex(val: Any, _objects: List[IRObject] = None) -> List[IRObject]:
        """
        Get objects from val of complex data type
        Support complex of types: List, Tuple, Dict, torch.Tensor, object
        
        @param val Any

        @return _objects List[IRObject]: all IRObject
        """
        _objects = [] if _objects is None else _objects
        if isinstance(val, (tuple, list)):
            for item in val:
                IRSegment.get_objects_from_complex(item, _objects)
        if isinstance(val, dict):
            for key, value in val.items():
                IRSegment.get_objects_from_complex(key, _objects)
                IRSegment.get_objects_from_complex(value, _objects)
        if isinstance(val, IRObject):
            _objects.append(val)
        return _objects

    @staticmethod
    def modify_objects_of_complex(val: Any, modifier: Callable) -> Any:
        """
        Get objects from val of complex data type
        Support complex of types: List, Tuple, Dict, torch.Tensor, object
        
        @param val Any
        @param modifier Callable: modify IRObject to another one

        @return new_val List[IRObject]: all IRObject
        """
        rcall = IRSegment.modify_objects_of_complex
        if isinstance(val, tuple):
            return tuple(rcall(item, modifier) for item in val)
        if isinstance(val, list):
            return list(rcall(item, modifier) for item in val)
        if isinstance(val, dict):
            return {rcall(key, modifier):rcall(value, modifier) for key, value in val.items()}
        if isinstance(val, IRObject):
            return modifier(val)
        return val
