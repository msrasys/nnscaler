"""
IRGraph:
    a graph that is composed by node (IRFwOperation) and edge (IRTensor).

    Note the device of graph.inputs() can be different of the same input
    tensor of operation node in the graph. In this case, a move operation
    will be inserted at scheduling time.
"""

from typing import Union, Tuple, List, Optional, Any, Dict
import copy
from cube.graph.operator.operator import IRBpOperation

from cube.ir.cten import IRTensor, IRCell
from cube.graph.tensor import IRFullTensor, IRSubTensor

from cube.algorithm.generics import GenericDistAlgo


__all__ = ['IRGraph']


class IRGraph(IRCell):
    """
    PyTorch IR Graph

    The IR Graph only contains forward graph
    """

    def __init__(self, 
                 nodes: List[IRCell],
                 input_tensors: Optional[List[IRTensor]], 
                 output_tensors: Optional[List[IRTensor]], 
                 module_name: str):
        
        self._nodes: List[IRCell] = nodes
        self._parameters = list()

        if input_tensors is None:
            input_tensors = list()
            inputs = IRCell.get_inputs(nodes)
            for input in inputs:
                if not input.is_param():
                    input_tensors.append(input)
        if output_tensors is None:
            output_tensors = list()
            outputs = IRCell.get_outputs(nodes)
            for output in outputs:
                if not output.is_param():
                    output_tensors.append(output)

        super().__init__(
            name=module_name,
            signature=module_name,
            input_length=len(input_tensors),
            output_length=len(output_tensors)
        )

        # convert to SubTensor
        inputs = list()
        for tensor in input_tensors:
            if isinstance(tensor, IRFullTensor):
                tensor = tensor.tosub()
            inputs.append(tensor)
        outputs = list()
        for tensor in output_tensors:
            if isinstance(tensor, IRFullTensor):
                tensor = tensor.tosub()
            outputs.append(tensor)
        for node in self.nodes():
            for idx, tensor in enumerate(node.inputs()):
                if isinstance(tensor, IRFullTensor):
                    tensor = tensor.tosub()
                    node.set_input(idx, tensor)
            for idx, tensor in enumerate(node.outputs()):
                if isinstance(tensor, IRFullTensor):
                    tensor = tensor.tosub()
                    node.set_output(idx, tensor)

        for idx, tensor in enumerate(inputs):
            self.set_input(idx, tensor)
        for idx, tensor in enumerate(outputs):
            self.set_output(idx, tensor)

        # set parameter
        for node in self._nodes:
            for input in node.inputs():
                if isinstance(input, IRTensor):
                    if input.is_param():
                        # parameters already set
                        self._parameters.append(input)
                        continue
                    if input not in input_tensors and \
                       input.is_leaf(self._nodes):
                        input.as_param()
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
                for out_idx, out_tensor in enumerate(src_node.outputs()):
                    if not isinstance(out_tensor, IRTensor):
                        continue
                    for in_idx, in_tensor in enumerate(dst_node.inputs()):
                        if not isinstance(in_tensor, IRTensor):
                            continue
                        if out_tensor.overlap(in_tensor):
                            src_node.add_successor(out_idx, dst_node)
                            dst_node.add_predecessor(in_idx, src_node)

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

    ## Primitives for policy expression ##

    def partition(self, op: IRCell, algo: GenericDistAlgo, config: Dict) -> Optional[List[IRCell]]:
        """
        Policy primitive. Partition an operator by using
        op_partition_algorithm and its configuration. Note the
        backward op-partition will be automatically done.

        Args:
            op: cell to be partitioned
            algo: generic distributed algorithm related to the op
            config: dict

        Returns:
            nodes: List[IRCell] if partitioned successfully.
            None if failed
        """
        if not isinstance(op, IRCell):
            raise TypeError("Expected op to be IRCell (IRFwOperation)")
        if not isinstance(algo, GenericDistAlgo):
            raise TypeError("Expected algo to be GenericDistAlgo")

        if algo.logic_op != type(op):
            return None
        if not algo.satisfy(config):
            return None
        fnodes = algo.instantiate(op, config)
        # remove reference
        for idx in range(len(op.inputs())):
            op.set_input(idx, None)
        # set backward mirror node
        if op.mirror is not None:
            # generate mirror node
            for fnode in fnodes:
                bnode = IRBpOperation(
                    data_num=len(fnode.inputs()),
                    grad_num=len(fnode.outputs())
                )
                for idx, val in enumerate(fnode.inputs()):
                    grad = None
                    if isinstance(val, IRSubTensor):
                        # TODO: requires_grad = False should be set to None
                        grad = val.get_grad(fnode)
                        val.grad = grad
                    bnode.set_data(idx, val)
                    bnode.set_output(idx, grad)
                for idx, val in enumerate(fnode.outputs()):
                    grad = None
                    if isinstance(val, IRSubTensor):
                        # TODO: requires_grad = False should be set to None
                        grad = val.get_grad(fnode)
                        val.grad = grad
                    bnode.set_grad(idx, grad)
                fnode.mirror = bnode
                fnode.device = op.device
                bnode.mirror = fnode
                bnode.device = op.mirror.device
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

    def __repr__(self):
        dscp = f"\n{self.name}:\n{'=' * len(self.name)}\n"
        # inputs
        inputs = list()
        for tensor in self.inputs():
            if isinstance(tensor, IRTensor):
                anno = 't'
                if tensor.is_param():
                    anno = 'w'
                if tensor.is_grad():
                    anno = 'g'
                inputs.append(f'{anno}{tensor._id}')
            else:
                inputs.append(tensor)
        outputs = list()
        for tensor in self.outputs():
            if isinstance(tensor, IRTensor):
                anno = 't'
                if tensor.is_param():
                    anno = 'w'
                if tensor.is_grad():
                    anno = 'g'
                outputs.append(f'{anno}{tensor._id}')
            else:
                outputs.append(tensor)
        dscp += f"Inputs: {inputs}\n"
        # nodes
        for node in self._nodes:
            succ_node_ids = [None] * len(node.outputs())
            for out_idx in range(len(node.outputs())):
                node_list = [snode._id for snode in node.successors(out_idx)]
                succ_node_ids[out_idx] = node_list
            dscp += f"\n{node._id}: {node} -> node id {succ_node_ids}\n"
        # outputs
        dscp += f"\nOutputs: {outputs}\n{'=' * len(self.name)}\n"
        return dscp
