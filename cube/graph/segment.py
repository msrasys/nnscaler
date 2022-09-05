from contextlib import contextmanager
from typing import Dict, Union, List, Optional, Set, Tuple

from cube.ir.tensor import IRFullTensor, IRSubTensor
from cube.ir.cten import IRTensor, IRCell
from cube.ir.operator import IRFwOperation, IRBpOperation
from cube.ir.adapter import IRAdapter


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

    Once the segment is generated, its input and output will be fixed.
    Inserting and removing nodes that could change input/output are not allowed.
    """

    def __init__(self, nodes: List[IRCell], inputs: List[IRTensor], outputs: List[IRSubTensor], name='segment'):
        super().__init__(name, '', len(inputs), len(outputs), init_outputs=False)

        self._nodes: List[IRCell] = []
        self._idevice = [t.device for t in inputs]
        self._odevice = [t.device for t in outputs]

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

        # self.reset_dependency()

        # FIXME: update when manipulating
        self._have_forward = any(isinstance(n, IRFwOperation) for n in nodes)
        self._have_backward = any(isinstance(n, IRBpOperation) for n in nodes)

    def isfw(self) -> bool:
        return self._have_forward

    def isbw(self) -> bool:
        return self._have_backward

    def full_tensors(self) -> Tuple[IRFullTensor]:
        """
        Get all full tensors of this graph.
        Note the full tensor inside the node will not be returned.

        @return ftensors List[IRFullTensor]
        """
        return tuple(self._ftensors)

    def attributes(self) -> Tuple[IRFullTensor]:
        """
        Get al full tensor attributes of this graph
        Note the full tensor inside the node will not be returned.

        @return ftensors List[IRFullTensor]
        """
        return Tuple(self._attributes)

    def reset_dependency(self):
        """
        Reset the node dataflow dependency

        FIXME

        Note all the predefined control dependencies will be removed.
        """
        for node in self._nodes:
            node.clear_predecessor()
            node.clear_successor()
        # TODO: adapter dependency not set
        for ftensor in self._ftensors:
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
        Get node index.

        @param node IRCell: the queried node

        @return index int: the index
        """
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
        assert isinstance(node, IRCell)
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

    def grad(self, tensor: IRSubTensor) -> IRSubTensor:
        """
        Get gradient of the tensor.

        @param tensor IRSubTensor: IRSubTensor: the queried tensor

        @return gradient IRSubTensor: the gradient
        """
        segment: IRSegment = self.segment(tensor.cell)
        assert isinstance(tensor, IRSubTensor), "Only tensor has gradient"
        fgrad = tensor.parent.grad
        # None means no gradient requirement, flaot means its the loss
        if fgrad is None or isinstance(fgrad, float):
            tensor.grad = fgrad
            return fgrad
        ftensor = tensor.parent
        # this tensor is consumed
        if tensor in tensor.cell.inputs():
            consumers = []
            for ctensor, consumer in zip(segment.ctensors(ftensor), segment.consumers(ftensor)):
                assert not (ctensor != tensor and ctensor.overlap(tensor)), "parital overlap is not supported for gradient"
                if ctensor == tensor and consumer not in consumers:
                    consumers.append(consumer)
            valmap = (consumers.index(tensor.cell), len(consumers))
            grad = ftensor.grad.select(
                indmap = tensor.indmap,
                valmap = valmap
            )
        # this tensor is produced
        elif tensor in tensor.cell.outputs():
            grad = ftensor.grad.select(
                indmap = tensor.indmap,
                valmap = (0, 1),
            )
        tensor.grad = grad
        return grad

    def debug_print_tensor_map(self, ftensor: Optional[IRFullTensor] = None):
        ftensors = [ftensor] if ftensor is not None else self._ftensors
        for ftensor in ftensors:
            print(f'Full Tensor: {ftensor}')
            print(f'Producers:')
            for producer in self._producers[ftensor]:
                print(f'\t{producer}')
            print(f'Consumers:')
            for producer in self._consumers[ftensor]:
                print(f'\t{producer}')

    def create_bwop(self, fwop: IRFwOperation) -> IRBpOperation:
        """
        Create dummy backward operator for given forward operator
        """
        assert isinstance(fwop, IRFwOperation), "Expected IRFwOperation"
        fsegment: IRSegment = self.segment(fwop)
        igrads = [fsegment.grad(t) if t.requires_grad else None for t in fwop.inputs() if isinstance(t, IRSubTensor)]
        ograds = [fsegment.grad(t) if t.requires_grad else None for t in fwop.outputs() if isinstance(t, IRSubTensor)]
        bwop = IRBpOperation(ograds, igrads)
        IRCell.make_pair(fwop, bwop)
        return bwop
    
    def update_bwop(self, bwop: IRCell) -> IRBpOperation:
        """
        Update backward operator or a backward segment.

        This is neccessary when fwop is partitioned and reference count is changed.
        
        @param bwop IRBpOperation or IRSegment: the backward operation.
            It can be at any hierarchy of this segemtn

        @return bwop IRBpOperation: the updated operation (inplace)
        """
        assert isinstance(bwop, (IRBpOperation, IRSegment))
        if isinstance(bwop, IRSegment):
            assert bwop.isbw() and (not bwop.isfw())
        bsegment: IRSegment = self.segment(bwop)
        fsegment = bsegment.mirror
        with bsegment.update(bwop):
            fwop: Union[IRFwOperation, IRSegment] = bwop.mirror
            igrads = [fsegment.grad(t) if t.requires_grad else None for t in fwop.inputs() if isinstance(t, IRSubTensor)]
            for idx, igrad in enumerate(igrads):
                bwop.set_output(idx, igrad)
            ograds = [fsegment.grad(t) if t.requires_grad else None for t in fwop.outputs() if isinstance(t, IRSubTensor)]
            # Ad-hoc fix: remove float that could be caused by loss for segment
            if isinstance(bwop, IRSegment):
                ograds = [grad for grad in ograds if isinstance(grad, IRSubTensor)]
            for idx, ograd in enumerate(ograds):
                bwop.set_input(idx, ograd)
        return bwop

    def update_ftensor_bw(self, ftensor: IRFullTensor):
        """
        Update all backward operators for a full tensor.

        @param ftensor IRFullTensor: the full tensor. If the full
            tensor is not a gradient, will update backward operators
            of ftensor.grad
        
        @return None
        """
        fgrad = ftensor.grad if not ftensor.is_grad() else ftensor
        if fgrad is None:
            return
        for producer in self.producers(fgrad):
            self.update_bwop(producer)
        for consumer in self.consumers(fgrad):
            self.update_bwop(consumer)

    # ====================== Basic Graph manipulations ======================

    def _add_ftensor(self, ftensor: IRFullTensor):
        """
        Add a full tensor in segment if the segment doesn't have the tensor.
        """
        assert isinstance(ftensor, IRFullTensor)
        if ftensor not in self._ftensors:
            self._ftensors.add(ftensor)
            self._producers[ftensor] = []
            self._consumers[ftensor] = []
            self._ptensors[ftensor] = []
            self._ctensors[ftensor] = []
        if ftensor.is_attr():
            self._attributes.add(ftensor)
    
    def _remove_ftensor(self, ftensor: IRFullTensor):
        """
        Remove a full tensor in segment
        """
        assert isinstance(ftensor, IRFullTensor)
        if ftensor in self._ftensors:
            self._ftensors.remove(ftensor)
            del self._producers[ftensor]
            del self._consumers[ftensor]
            del self._ptensors[ftensor]
            del self._ctensors[ftensor]
        if ftensor.is_attr() and ftensor in self._attributes:
            self._attributes.remove(ftensor)

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
            itensors = set(t for t in node.inputs() if isinstance(t, IRSubTensor))
            for itensor in itensors:
                ftensor = itensor.parent
                self._add_ftensor(ftensor)
                idx = len([c for c in self._consumers[ftensor] if self._nodes.index(c) < index])
                self._consumers[ftensor].insert(idx, node)
                self._ctensors[ftensor].insert(idx, itensor)
            # producer
            otensors = set(t for t in node.outputs() if isinstance(t, IRSubTensor))
            for otensor in otensors:
                ftensor = otensor.parent
                self._add_ftensor(ftensor)
                idx = len([c for c in self._producers[ftensor] if self._nodes.index(c) < index])
                self._producers[ftensor].insert(idx, node)
                self._ptensors[ftensor].insert(idx, otensor)
        else:
            segment = self._nodes[pos[0]]
            assert isinstance(segment, IRSegment), "Expected IRSegment"
            pos = CellPosition(pos.indices[1:])
            segment.insert(node, pos)

    def remove(self, node: IRCell, _pos: CellPosition = None) -> CellPosition:
        """
        Remove a node at index

        # TODO: check input and output

        @param node IRCell: the removed node
        
        @return index CellPosition: the removed index
        """
        pos = self.index(node) if _pos is None else _pos
        assert self.node(pos) == node, "posititon doesn't not match with node"

        if len(pos.indices) == 1:
            index = pos[0]
            # remove
            self._nodes.pop(index)
            # update producer and consumer
            if isinstance(node, IRAdapter): return pos
            # consumer
            itensors = set(t for t in node.inputs() if isinstance(t, IRSubTensor))
            for itensor in itensors:
                ftensor = itensor.parent
                idx = self._consumers[ftensor].index(node)
                self._consumers[ftensor].pop(idx)
                self._ctensors[ftensor].pop(idx)
                if len(self._consumers[ftensor]) == 0 and len(self._producers[ftensor]) == 0:
                    self._remove_ftensor(ftensor)
            # producer
            otensors = set(t for t in node.outputs() if isinstance(t, IRSubTensor))
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

    def exist(self, node: IRCell) -> bool:
        """
        Check if the node is in this graph

        @param node IRCell: the queried node

        @return exsit bool: True if exist otherwise False
        """
        if node in self._nodes:
            return True
        for segment in self._nodes:
            if not isinstance(segment, IRSegment): continue
            if segment.exist(node):
                return True
        return False

    def finsert(self, fwop: IRFwOperation, index: Union[int, CellPosition]) -> IRFwOperation:
        """
        Insert a forward node and create its backward.
        The created backward operator will be happen right before
        the backward of fwop's previous forward node

        This requires the segment has its backward segment

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
        # insert backward
        assert fsegment.mirror is not None, "Missing backward segment"
        bsegment: IRSegment = fsegment.mirror
        bidx = 0
        for idx in range(index - 1, -1, -1):
            prev_fnode = fsegment.node(idx)
            if prev_fnode.mirror is not None:
                bidx = bsegment.index(prev_fnode.mirror)
                break
        bsegment.insert(bwop, bidx)
        return fwop

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
        segments: List[IRSegment] = [self.segment(node) for node in nodes]
        assert len(set(segments)) == 1, "Cross segment hierarchy grouping is not allowed"
        segment = segments[0]

        inputs, outputs = set(), set()
        for node in nodes:
            # update inputs
            itensors = [t for t in node.inputs() if isinstance(t, IRSubTensor)]
            for itensor in itensors:
                ftensor = itensor.parent
                if itensor.is_attr(): continue
                # from segment inputs
                if any(t.overlap(itensor) for t in segment.inputs() if isinstance(t, IRSubTensor)):
                    inputs.add(itensor)
                    continue
                # from outside producers
                for ptensor, producer in zip(segment.ptensors(ftensor), segment.producers(ftensor)):
                    if ptensor.overlap(itensor) and producer not in nodes:
                        inputs.add(itensor)
                        continue
            # update outputs
            otensors = [t for t in node.outputs() if isinstance(t, IRSubTensor)]
            for otensor in otensors:
                ftensor = otensor.parent
                if otensor.is_attr(): continue
                # loss doesn't have consumers
                if len(segment.consumers(ftensor)) == 0:
                    outputs.add(otensor)
                # from segment outputs
                if any(t.overlap(otensor) for t in segment.outputs() if isinstance(t, IRSubTensor)):
                    outputs.add(otensor)
                    continue
                # for outside consumers
                for ctensor, consumer in zip(segment.ctensors(ftensor), segment.consumers(ftensor)):
                    if ctensor.overlap(otensor) and consumer not in nodes:
                        outputs.add(otensor)
                        continue
        segment = IRSegment(nodes, tuple(inputs), tuple(outputs))
        return segment


    def dispatch(self, devid: int, mirror=True) -> Optional[IRCell]:
        """
        Instantiate the segement to a specific device.

        @param devid int: the target device

        @return segment IRSegment: the instantiated segment
        """
        if devid not in self.device:
            return None
        if len(self.device) == 1 and self.device == [devid]:
            return self
        inputs, outputs, nodes = [], [], []
        for node in self._nodes:
            if devid in node.device:
                if isinstance(node, IRAdapter):
                    nodes.append(node.dispatch(devid))
                elif isinstance(node, IRSegment):
                    nodes.append(node.dispatch(devid))
                else:
                    assert len(node.device) == 1
                    nodes.append(node)
                for itensor in node.inputs():
                    if itensor in self._inputs:
                        inputs.append(itensor)
                for otensor in node.outputs():
                    if otensor in self._outputs:
                        otensor.append(otensor)
                    outputs.append(otensor)
        segment = IRSegment(nodes, inputs, outputs, self.name)
        if mirror and segment.mirror is not None:
            msegment = segment.mirror.dispatch(devid, mirror=False)
            IRCell.make_pair(segment, msegment)
        return segment


    # ========================== Graph Visualize ================================

    def __repr__(self):
        fw = 'f' if self.isfw() else 'b'
        dscp = f"{fw}Graph{self.cid}-{self.device}(inputs={self.inputs()}, outputs={self.outputs()})"
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
