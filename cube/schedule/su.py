from typing import List, Optional, Tuple
import copy
from enum import Enum

from cube.ir.cten import IRCell, IRTensor
from cube.graph.operator import IRBpOperation


class SUType(Enum):

    Dataloader = 'next(dataloader)'

    # outputs = cube.runtime.temporal.forward(model, *args)
    Forward = 'cube.runtime.executor.fexecute'

    # grads = cube.runtime.temporal.backward(
    #   input_tensors, output_tensors, output_grads
    # )
    Backward = 'cube.runtime.executor.backward'

    Transform = 'cube.runtime.transform'

    # cube.runtime.collectives.sendrecv(send_tensors, send_ranks,
    #   recv_shapes, from_ranks
    # )
    P2P = 'cube.runtime.adapter.sendrecv'
    Coll = 'cube.runtime.adapter.coll'

    Optimizer = 'cube.runtime.reducer.Reduce'

    Empty = 'None'


class ScheduleUnit(IRCell):
    r"""
    ScheduleUnit for policy scheduling.
    """

    def __init__(self, nodes: List[IRCell], stype: SUType, name='su'):
        """
        Create a ScheduleUnit.

        Args:
            nodes (List[IRCell]): A list of nodes in IRGraph
        """

        if not all([isinstance(node, IRCell) for node in nodes]):
            raise ValueError("Expected each nodes to be List[IRCell]")
        if not isinstance(stype, SUType):
            raise TypeError("Expected stype be SUType")

        # get inputs and outputs
        # TODO: fix bug on multi-branch
        inputs = IRCell.get_inputs(nodes)
        # inputs = [input for input in inputs if not input.is_param()]
        outputs = IRCell.get_outputs(nodes)
        # outputs = [output for output in outputs if not output.is_param()]
        super().__init__(
            name          = name,
            signature     = stype.value,
            input_length  = len(inputs),
            output_length = len(outputs)
        )

        self.stype = stype

        self._nodes = nodes
        for idx, input in enumerate(inputs):
            self.set_input(idx, input)
        for idx, output in enumerate(outputs):
            self.set_output(idx, output)

        # each input is associated with a reshape (merge) adatpers and
        # a couple of send adapters and recv adapters (send + recv)
        self._merge_adapters: List[ScheduleUnit] = [None] * len(inputs)
        self._send_in_adapters: List[List[ScheduleUnit]] = [
            list() for _ in range(len(inputs))
        ]
        self._recv_in_adapters: List[List[ScheduleUnit]] = [
            list() for _ in range(len(inputs))
        ]

        # each input is associated with a reshape (select) adatpers and
        # a couple of send adapters and recv adapters (send + recv)
        self._select_adapters: List[ScheduleUnit] = [None] * len(outputs)
        self._send_out_adapters: List[List[ScheduleUnit]] = [
            list() for _ in range(len(outputs))
        ]
        self._recv_out_adapters: List[List[ScheduleUnit]] = [
            list() for _ in range(len(outputs))
        ]

        # additional control dependency for add_flow
        self._ctrl_predecessors = list()
        self._ctrl_successors = list()

        self._tag = [node.tag for node in nodes]

    def __copy__(self):
        """
        Copy the SU. Note the mirror su is also copied
        """
        raise NotImplementedError("Copy SU is not supported yet")

    def in_adapters(self, index: Optional[int] = None) -> List:
        """
        Get adapter for the input tensor at index

        Returns:
            Tuple[List[ScheduleUnit], List[ScheduleUnit]]:
                the send_adapters and recv_adapters
        """
        if isinstance(index, int):
            if index >= len(self._inputs):
                raise RuntimeError(
                    f"Get index out of range ({index} >= {len(self._inputs)})"
                )
            send_adapters = copy.copy(self._send_in_adapters[index])
            recv_adapters = copy.copy(self._recv_in_adapters[index])
            return send_adapters, recv_adapters
        elif index is None:
            all_send_adapters = list()
            all_recv_adapters = list()
            for adapters in self._send_in_adapters:
                all_send_adapters += adapters
            for adapters in self._recv_in_adapters:
                all_recv_adapters += adapters
            return all_send_adapters, all_recv_adapters
        else:
            raise TypeError("Expected index to be None or int")

    def merge_adapters(self, index: Optional[int] = None) -> List:
        """
        Get select adapter for the input tensor at index

        Returns:
            Union[ScheduleUnit, List[ScheduleUnit]]
        """
        if isinstance(index, int):
            if index >= len(self._inputs):
                raise RuntimeError(
                    f"Get index out of range ({index} >= {len(self._inputs)})"
                )
            select_adapter = self._merge_adapters[index]
            return select_adapter
        elif index is None:
            return copy.copy(self._merge_adapters)
        else:
            raise TypeError("Expected index to be None or int")

    def out_adapters(self, index: Optional[int] = None) -> Tuple[List, List]:
        """
        Get adapter for the output tensor at index

        Returns:
            Tuple[List[ScheduleUnit], List[ScheduleUnit]]:
                the send_adapters and recv_adapters
        """
        if isinstance(index, int):
            if index >= len(self._outputs):
                raise RuntimeError(
                    f"Get index out of range ({index} >= {len(self._outputs)})"
                )
            send_adapters = copy.copy(self._send_out_adapters[index])
            recv_adapters = copy.copy(self._recv_out_adapters[index])
            return send_adapters, recv_adapters
        elif index is None:
            all_send_adapters = list()
            all_recv_adapters = list()
            for adapters in self._send_out_adapters:
                all_send_adapters += adapters
            for adapters in self._recv_out_adapters:
                all_recv_adapters += adapters
            return all_send_adapters, all_recv_adapters
        else:
            raise TypeError("Expected index to be None or int")

    def select_adapters(self, index: Optional[int] = None) -> List:
        """
        Get select adapter for the input tensor at index

        Returns:
            Union[ScheduleUnit, List[ScheduleUnit]]
        """
        if isinstance(index, int):
            if index >= len(self._outputs):
                raise RuntimeError(
                    f"Get index out of range ({index} >= {len(self._outputs)})"
                )
            select_adapter = self._select_adapters[index]
            return select_adapter
        elif index is None:
            return copy.copy(self._select_adapters)
        else:
            raise TypeError("Expected index to be None or int")

    def _clear_adapters(self):
        """
        Clear all adapters for this SU
        """
        self._send_in_adapters: List[List[ScheduleUnit]] = [
            list() for _ in range(len(self.inputs()))
        ]
        self._recv_in_adapters: List[List[ScheduleUnit]] = [
            list() for _ in range(len(self.inputs()))
        ]
        self._merge_adapters: List[ScheduleUnit] = [None] * len(self._inputs)
        self._select_adapters: List[ScheduleUnit] = [None] * len(self._outputs)
        self._send_out_adapters: List[List[ScheduleUnit]] = [
            list() for _ in range(len(self.outputs()))
        ]
        self._recv_out_adapters: List[List[ScheduleUnit]] = [
            list() for _ in range(len(self.outputs()))
        ]

    def _add_in_adapter(self, index: int, send_adapters, recv_adapters):
        """
        Add adapters to the input tensor of this SU

        Args:
            index (int): the input index
            send_adapter (ScheduleUnit)
            recv_adapter (ScheduleUnit)
        """
        if index >= len(self._inputs):
            raise ValueError(f"index {index} out of range {len(self._inputs)}")
        if isinstance(send_adapters, ScheduleUnit):
            send_adapters = [send_adapters]
        if not all(isinstance(adapter, ScheduleUnit) for adapter in send_adapters):
            raise TypeError("Expected send adapter to be (list of) ScheduleUnit")
        if isinstance(recv_adapters, ScheduleUnit):
            recv_adapters = [recv_adapters]
        if not all(isinstance(adapter, ScheduleUnit) for adapter in send_adapters):
            raise TypeError("Expected recv adapters to be (list of) ScheduleUnit")
        if len(send_adapters) != len(recv_adapters):
            raise ValueError("Expected same number of send / recv adapters")
        for send_adapter, recv_adapter in zip(send_adapters, recv_adapters):
            self._send_in_adapters[index].append(send_adapter)
            self._recv_in_adapters[index].append(recv_adapter)

    def _set_merge_adapter(self, index: int, merge_adapter):
        """
        Set adapters to the input tensor of this SU

        Args:
            index (int): the input index
            merge_adapter (ScheduleUnit)
        """
        if index >= len(self._inputs):
            raise ValueError(f"index {index} out of range {len(self._inputs)}")
        if merge_adapter is not None and not isinstance(merge_adapter, ScheduleUnit):
            raise TypeError("Expected merge adapter to be None or ScheduleUnit")
        self._merge_adapters[index] = merge_adapter

    def _add_out_adapter(self, index: int, send_adapters, recv_adapters):
        """
        Add adapters to the output tensor of this SU

        Args:
            index (int): the output index
            send_adapter (ScheduleUnit)
            recv_adapter (ScheduleUnit)
        """
        if index >= len(self._outputs):
            raise ValueError(f"index {index} out of range {len(self._outputs)}")
        if isinstance(send_adapters, ScheduleUnit):
            send_adapters = [send_adapters]
        if not all(isinstance(adapter, ScheduleUnit) for adapter in send_adapters):
            raise TypeError("Expected send adapter to be (list of) ScheduleUnit")
        if isinstance(recv_adapters, ScheduleUnit):
            recv_adapters = [recv_adapters]
        if not all(isinstance(adapter, ScheduleUnit) for adapter in send_adapters):
            raise TypeError("Expected recv adapters to be (list of) ScheduleUnit")
        if len(send_adapters) != len(recv_adapters):
            raise ValueError("Expected same number of send / recv adapters")
        for send_adapter, recv_adapter in zip(send_adapters, recv_adapters):
            self._send_out_adapters[index].append(send_adapter)
            self._recv_out_adapters[index].append(recv_adapter)

    def _set_select_adapter(self, index: int, select_adapter):
        """
        Set adapters to the output tensor of this SU

        Args:
            index (int): the output index
            select_adapter (ScheduleUnit)
        """
        if index >= len(self._outputs):
            raise ValueError(f"index {index} out of range {len(self._inputs)}")
        if select_adapter is not None and not isinstance(select_adapter, ScheduleUnit):
            raise TypeError("Expected merge adapter to be Optional[ScheduleUnit]")
        self._select_adapters[index] = select_adapter

    def _remove_adapter(self, adapter):
        """
        Remove the adapter
        """
        for send_adapters in self._send_in_adapters:
            if adapter in send_adapters:
                send_adapters.remove(adapter)
                return True
        for recv_adapters in self._recv_in_adapters:
            if adapter in recv_adapters:
                recv_adapters.remove(adapter)
                return True
        if adapter in self._merge_adapters:
            idx = self._merge_adapters.index(adapter)
            self._merge_adapters[idx] = None
            return True
        if adapter in self._select_adapters:
            idx = self._select_adapters.index(adapter)
            self._select_adapters[idx] = None
            return True
        for send_adapters in self._send_out_adapters:
            if adapter in send_adapters:
                send_adapters.remove(adapter)
                return True
        for recv_adapters in self._recv_out_adapters:
            if adapter in recv_adapters:
                recv_adapters.remove(adapter)
                return True
        return False

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

    def add_predecessor(self, input_index: int, su):
        """
        Add a predecessor cell in the input_index slot. 
        self.input[input_index] = node.output[out_index]
        """
        if input_index == -1:
            self._ctrl_predecessors.append(su)
        else:
            super().add_predecessor(input_index, su)

    def predecessors(self, index: Optional[int] = None) -> List:
        """
        Get 1-hop predecessor cells including control predecessors

        Args:
            index (Optional[int]):
                -1: return control predecessors
                None: return all predecessors including index
                >0 : return input SUs at input index

        Returns:
            cell(s): List[IRCell]
        """
        if isinstance(index, int):
            if index == -1:
                return copy.copy(self._ctrl_predecessors)
            if index >= len(self._inputs):
                raise RuntimeError(
                    f"Get the input out of range ({index} >= {len(self._inputs)}"
                )
            return copy.copy(self._predecessors[index])
        elif index is None:
            predecessors = list()
            for pre_cells in self._predecessors:
                predecessors += pre_cells
            predecessors += self._ctrl_predecessors
            return predecessors
        else:
            raise TypeError("Expected index to be None or int")

    def add_successor(self, output_index: int, su):
        """
        Set self node the output index node. 
        `node` will take the self.outputs(index) as the input
        """
        if output_index == -1:
            self._ctrl_successors.append(su)
        else:
            super().add_successor(output_index, su)

    def successors(self, index: Optional[int] = None) -> List:
        """
        Get 1-hop successor cells including control successors

        Args:
            index (Optional[int]):
                -1: return control successors
                None: return all successors including index
                >0 : return output SUs at output index

        Returns:
            cells: List[ScheduleUnit]
        """
        if isinstance(index, int):
            if index == -1:
                return copy.copy*self._ctrl_successors
            if index >= len(self._outputs):
                raise RuntimeError(
                    f"Get the output out of range ({index} >= {len(self._outputs)}"
                )
            return copy.copy(self._successors[index])
        elif index is None:
            successors = list()
            for post_cells in self._successors:
                successors += post_cells
            successors += self._ctrl_successors
            return successors
        else:
            raise TypeError("Expected index to be None or int")

    def __repr__(self):
        su_inputs = list()
        for tensor in self.inputs():
            if isinstance(tensor, IRTensor):
                anno = 't'
                if tensor.is_param():
                    anno = 'w'
                if tensor.is_grad():
                    anno = 'g'
                su_inputs.append(f'{anno}{tensor._id}')
            else:
                su_inputs.append(tensor)
        su_outputs = list()
        for tensor in self.outputs():
            if isinstance(tensor, IRTensor):
                anno = 't'
                if tensor.is_param():
                    anno = 'w'
                if tensor.is_grad():
                    anno = 'g'
                su_outputs.append(f'{anno}{tensor._id}')
            else:
                su_outputs.append(tensor)
        dscp = f'SU({self.stype}, nodes={len(self.nodes())})-dev{self.device}: {su_inputs} -> {su_outputs}'
        return dscp
