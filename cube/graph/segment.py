from typing import Union, List, Optional, Set
import copy

from cube.ir.tensor import IRFullTensor, IRSubTensor
from cube.ir.cten import IRTensor, IRCell
from cube.ir.operator import IRFwOperation, IRBpOperation
from cube.ir.adapter import IRAdapter



class IRSegment(IRCell):
    """
    A distributed sub-graph representing a piece of workload in parent IRGraph

    Once the segment is generated, its input and output will be fixed.
    Inserting and removing nodes that could change input/output are not allowed.
    """

    def __init__(self, nodes: List[IRCell], inputs: List[IRTensor], outputs: List[IRSubTensor], name='segment'):
        super().__init__(name, '', len(inputs), len(outputs), init_outputs=False)

        self._nodes: List[IRCell] = nodes
        self._idevice = [t.device for t in inputs]
        self._odevice = [t.device for t in outputs]

        self._inputs = list(inputs)
        self._outputs = list(outputs)
        # for idx, val in enumerate(inputs):
        #     self.set_input(idx, val)
        # for idx, val in enumerate(outputs):
        #     self.set_output(idx, val)

        # full tensors
        self._full_tensors: Set[IRFullTensor] = set()
        for node in nodes:
            for tensor in node.inputs() + node.outputs():
                if isinstance(tensor, IRSubTensor):
                    self._full_tensors.add(tensor.parent)

        self._have_forward = any(isinstance(n, IRFwOperation) for n in nodes)
        self._have_backward = any(isinstance(n, IRBpOperation) for n in nodes)

    @property
    def forward(self) -> bool:
        return self._have_forward

    def full_tensors(self) -> List[IRFullTensor]:
        """
        Return full tensor list
        """
        return list(self._full_tensors)

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

    def nodes(self, idx: Optional[int] = None) -> Union[IRCell, List[IRCell]]:
        """
        Get all the nodes.

        @return nodes List[IRCell]: all the nodes
        """
        if isinstance(idx, int):
            return self._nodes[idx]
        else:
            return copy.copy(self._nodes)

    def node(self, index: int) -> IRCell:
        """
        Get node at position index

        @param index int: the node index

        @return node IRCell: the node.
        """
        return self._nodes[index]

    def index(self, node: IRCell) -> int:
        """
        Get node index.

        @param node IRCell: the queried node

        @return index int: the index
        """
        return self._nodes.index(node)

    # ====================== Basic Graph manipulations ======================

    def insert(self, node: IRCell, index: int):
        """
        Insert a node at index.

        TODO: check input and output

        @param node IRCell: the inserted node
        @param index int: the index

        """
        assert node not in self._nodes, f"duplicated insertation of node: {node}"
        self._nodes.insert(index, node)

    def remove(self, node: IRCell) -> int:
        """
        Remove a node at index

        # TODO: check input and output

        @param node IRCell: the removed node
        
        @return index int: the removed index
        """
        assert node in self._nodes, f"The removed node doesn't exist"
        index = self._nodes.index(node)
        self._nodes.pop(index)
        return index

    def replace(self, node: IRCell, new_nodes: List[IRCell]) -> int:
        """
        Replace one node by multiple nodes

        # TODO: check input and output

        @param node IRCell: the replaced node
        @param new_nodes List[IRCell]: the nodes to be inserted.

        @return index int: the replaced node index
        """
        idx = self.remove(node)
        self._nodes = self._nodes[:idx] + list(new_nodes) + self._nodes[idx:]
        return idx

    def exist(self, node: IRCell) -> bool:
        """
        Check if the node is in this graph

        @param node IRCell: the queried node

        @return exsit bool: True if exist otherwise False
        """
        return node in self._nodes

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


    ###### ============ Transformation Primitives ============ #######


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


    # ========================== Graph Visualize ================================

    def to_str(self, skip_attr: bool = False) -> str:
        name = ('f' if self.forward else 'b') + 'Segment'
        inputs = tuple(t for t in self.inputs() if not (t.is_attr() and skip_attr))
        outputs = tuple(t for t in self.outputs() if not (t.is_attr() and skip_attr))
        return f'{name}{self._id}-{self.device}(inputs={inputs}, outputs={outputs})'

    def __repr__(self):
        return self.to_str()

    def extra_repr(self) -> str:
        dscp = repr(self)
        for node in self.nodes():
            dscp += '\n\t' + repr(node)
        return dscp
