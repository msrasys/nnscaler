"""
IRGraph:
    a graph that is composed by node (IROperation) and edge (IRTensor).

    Note the device of graph.inputs() can be different of the same input
    tensor of operation node in the graph. In this case, a move operation
    will be inserted at scheduling time.
"""

from typing import Union, Tuple, List, Optional, Any

from cube.ir.cten import IRTensor, IRCell
from cube.graph.operator import IROperation
from cube.graph.comm import IRCommunication

import copy


__all__ = ['IRGraph']


class IRGraph(IRCell):
    """
    PyTorch IR Graph

    The IR Graph only contains forward graph
    """

    def __init__(self, 
                 nodes: List[IROperation],
                 input_tensors: Optional[List[IRTensor]], 
                 output_tensors: Optional[List[IRTensor]], 
                 module_name: str):
        
        self._nodes: List[IROperation] = nodes
        self.reset_dependency()

        if input_tensors is None:
            input_tensors = IRCell.get_inputs(nodes)
        if output_tensors is None:
            output_tensors = IRCell.get_outputs(nodes)

        super().__init__(
            name=module_name,
            signature=module_name,
            input_length=len(input_tensors),
            output_length=len(output_tensors)
        )

        for idx, tensor in enumerate(input_tensors):
            self.set_input(idx, tensor)
        for idx, tensor in enumerate(output_tensors):
            self.set_output(idx, tensor)

        self.tag = 'forward'

    def reset_dependency(self):
        """
        Reset the node dataflow dependency
        """
        # set node predecessors and successors
        for src_idx in range(len(self._nodes)):
            src_cell = self._nodes[src_idx]
            src_cell._successors = [
                list() for _ in range(len(src_cell.outputs()))
            ]
            for dst_idx in range(src_idx + 1, len(self._nodes)):
                dst_cell = self._nodes[dst_idx]
                dst_cell._predecessors = [
                    list() for _ in range(len(dst_cell.inputs()))
                ]
                for tensor in src_cell.outputs():
                    if isinstance(tensor, IRTensor):
                        if tensor in dst_cell.inputs():
                            src_output_idx = src_cell.outputs().index(tensor)
                            src_cell.add_successor(src_output_idx, dst_cell)
                            dst_input_idx = dst_cell.inputs().index(tensor)
                            dst_cell.add_predecessor(dst_input_idx, src_cell)

    def copy(self, reverse=False):
        """
        Copy the graph but re-new the intermediate tensor
        """
        new_tensors = dict()  # old graph tensor._id -> new tensor

        def _renew(val: Any):
            if not isinstance(val, IRTensor):
                return val
            # parameters
            if val.is_leaf(self.nodes()) and val not in self.inputs():
                return val
            # intermediate data
            if val._id not in new_tensors:
                tensor = val.renew()
                new_tensors[val._id] = tensor
            return new_tensors[val._id]

        nodes = list()
        for node in self.nodes():

            if isinstance(node, IRCommunication):
                send_tensors = [_renew(tensor) for tensor in node.inputs()]
                send_ranks = node.send_ranks
                recv_tensors = [_renew(tensor) for tensor in node.outputs()]
                recv_ranks = node.recv_ranks
                if reverse:
                    send_tensors, recv_tensors = recv_tensors, send_tensors
                    send_ranks, recv_ranks = recv_ranks, send_ranks

                new_node = IRCommunication(
                    send_tensors = send_tensors,
                    send_ranks = send_ranks,
                    recv_tensors = recv_tensors,
                    recv_ranks = recv_ranks
                )

            elif isinstance(node, IROperation):
                inputs = node.inputs()
                outputs = node.outputs()
                if reverse:
                    inputs, outputs = outputs, inputs

                new_node = IROperation(
                    node.name, node.signature,
                    len(inputs), len(outputs)
                )
                # set inputs
                for idx, val in enumerate(inputs):
                    new_node.set_input(idx, _renew(val))
                # set outputs
                for idx, val in enumerate(outputs):
                    new_node.set_output(idx, _renew(val))
            else:
                raise TypeError("Found node with unsupported copy")
            new_node.device = node.device
            nodes.append(new_node)
        
        inputs = [_renew(input) for input in self.inputs()]
        outputs = [_renew(output) for output in self.outputs()]

        if reverse:
            inputs, outputs = outputs, inputs
            nodes = nodes[::-1]

        copied_graph = IRGraph(
            nodes = nodes,
            input_tensors = inputs,
            output_tensors = outputs,
            module_name = self.name
        )
        copied_graph.tag = self.tag
        return copied_graph

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

    def insert(self, node, src_node=None, dst_node=None, replaced_tensor=None):
        """
        Insert a node between src_node and dst_node. In default,
        if dst_node is not None, the node will be inserted right before
        dst_node. If the replaced_tensor is provided, the replaced_tensor
        in dst_node's inputs will be removed, and the output of node will be
        set as input for dst_node.
        """
        if not isinstance(node, IRCell):
            raise TypeError("Expected IRCell to insert")
        if dst_node is not None:
            if dst_node not in self._nodes:
                raise KeyError("dst_node not found")
            if replaced_tensor is not None:
                if replaced_tensor not in dst_node.inputs():
                    raise RuntimeError(f"Expected dst_node input has {replaced_tensor}")
                # remove dst_node input
                input_index = dst_node.inputs().index(replaced_tensor)
                if len(node.outputs()) != 1:
                    raise RuntimeError("replaced node requires output length to be 1")
                dst_node.set_input(input_index, node.outputs(0))
            # insert node
            index = self._nodes.index(dst_node)
            self._nodes.insert(index, node)
        elif src_node is not None:
            if src_node not in self._nodes:
                raise KeyError("src_node not found")
            index = self._nodes.index(src_node)
            self._nodes = self._nodes[:index+1] + [node] + self._nodes[index+1:]
        else:
            raise TypeError("Expected at least one of [src_node, dst_node]")
        #TODO: optimize this
        self.reset_dependency()

    def _replace_tensor(self, old_tensor: IRTensor, new_tensor: IRTensor):
        """
        Replace tensor from old_tensor to new_tensor for all the graph.
        """
        def _replace_inputs(cell, old_tensor, new_tensor):
            index = cell.inputs().index(old_tensor)
            cell.set_input(index, new_tensor)

        def _replace_outputs(cell, old_tensor, new_tensor):
            index = cell.outputs().index(old_tensor)
            cell.set_output(index, new_tensor)

        if old_tensor in self.inputs():
            _replace_inputs(self, old_tensor, new_tensor)

        for node in self.nodes():
            if old_tensor in node.inputs():
                _replace_inputs(node, old_tensor, new_tensor)
            if old_tensor in node.outputs():
                _replace_outputs(node, old_tensor, new_tensor)
        
        if old_tensor in self.outputs():
            _replace_outputs(self, old_tensor, new_tensor)

    def forward(self, *args) -> Union[IRTensor, Tuple[IRTensor]]:
        """
        forward will divide the graph into Actions according to
        node device assignment

        Currently each forward call will result in a new flow
        even if the input is same

        Returns:
            IRTensors
        """
        from cube.schedule.translator import LogicTranslator
        return LogicTranslator.forward(self, *args)

    def __call__(self, *args):
        """
        Register forward action
        """
        return self.forward(*args)

    def subgraph(self, sub_nodes: List[IRCell]):
        """
        Create a subgraph with sub nodes.

        The remote tensor will be set as graph input (recv tensors)
        and graph output (send tensors)

        Return:
            IRGraph
        """
        # find input
        inputs = list()
        outputs = list()
        for node in sub_nodes:
            outer_cells = list(set(self.nodes()) - set(sub_nodes))
            for tensor in node.inputs():
                if isinstance(tensor, IRTensor) and tensor not in inputs:
                    # if a tensor is generated by other nodes out of sub_nodes,
                    # then this tensor should be the input
                    src_nodes = tensor.src(outer_cells)
                    if len(src_nodes) != 0 or tensor in self.inputs():
                        inputs.append(tensor)
            for tensor in node.outputs():
                if isinstance(tensor, IRTensor) and tensor not in outputs:
                    # if a tensor is used by other nodes out of sub_nodes,
                    # then this tensor should be output
                    dst_nodes = tensor.dst(outer_cells)
                    if len(dst_nodes) != 0 or tensor in self.outputs():
                        outputs.append(tensor)

        graph = IRGraph(
            nodes = sub_nodes,
            input_tensors = inputs,
            output_tensors = outputs,
            module_name = self.name
        )

        return graph

    def __repr__(self):
        dscp = f"\n{self.name}:\n{'=' * len(self.name)}\n"
        # inputs
        dscp += f'Inputs: {self._inputs}\n'
        # nodes
        for node in self._nodes:
            succ_node_ids = [None] * len(node.outputs())
            for out_idx in range(len(node.outputs())):
                node_list = [snode._id for snode in node.successors(out_idx)]
                succ_node_ids[out_idx] = node_list
            dscp += f"\n{node._id}: {node} -> node id {succ_node_ids}\n"
        # outputs
        dscp += f"\nOutputs: {self._outputs}\n{'=' * len(self.name)}\n"
        return dscp
