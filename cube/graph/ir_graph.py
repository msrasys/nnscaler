from cube.graph.ir_opten import IROperation, IRTensor
from cube.tschedule.action import Action
from cube.tschedule.pool import TSchedulePool

from typing import Union, Tuple, List, Optional
import copy


__all__ = ['IRGraph', 'IRLocalGraph']


class IRGraph:
    """
    PyTorch IR Graph

    The IR Graph only contains forward graph
    """

    def __init__(self, 
                 nodes: List[IROperation],
                 input_tensors: List[IRTensor], 
                 output_tensors: List[IRTensor], 
                 module_name: str):
        self.module_name = module_name
        self._nodes: List[IROperation] = nodes
        self._inputs = input_tensors
        self._outputs = output_tensors
        # default is forward graph
        self.tag = 'forward'

    def add_node(self, node: IROperation):
        if not isinstance(node, IROperation):
            raise TypeError("Expected node to be IROperation")
        self._nodes.append(node)

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
            return self._nodes
        else:
            raise TypeError("Expected index to be None or int")

    def inputs(self, index: Optional[int] = None):
        if isinstance(index, int):
            if index >= len(self._inputs):
                raise RuntimeError(
                    f"Get the input out of range ({index} >= {len(self._inputs)}"
                )
            return self._inputs[index]
        elif index is None:
            return self._inputs
        else:
            raise TypeError("Expected index to be None or int")

    def outputs(self, index: Optional[int] = None):
        """
        Get output tensor at output index

        Args:
            index (int or None): 
                index of the outputs, None will return the nodes
                for all the outputs
        """
        if isinstance(index, int):
            if index >= len(self._outputs):
                raise RuntimeError(
                    f"Get the output out of range ({index} >= {len(self._outputs)}"
                )
            return self._outputs[index]
        elif index is None:
            return self._outputs
        else:
            raise TypeError("Expected index to be None or int")

    def replace(self, target: IROperation, nodes: List[IROperation]):
        """
        Replace the node with new nodes (IRGraph)
        """
        raise NotImplementedError

    def forward(self, *args, **kwargs) -> Union[IRTensor, Tuple[IRTensor]]:
        """
        forward will divide the graph into Actions according to
        node device assignment

        Currently each forward call will result in a new flow
        even if the input is same

        Returns:
            List[Action]
        """
        if len(self._outputs) == 1:
            tensor = copy.copy(self._outputs[0])
            tensor.set_forward_graph(self)
            return tensor
        else:
            tensors = tuple([copy.copy(output) for output in self._outputs])
            for tensor in tensors:
                if isinstance(tensor, IRTensor):
                    tensor.set_forward_graph(self)
            return tensors

    def __call__(self, *args, **kwargs):
        """
        Register forward action
        """
        curr_nodes: List[IROperation] = list()
        curr_device = None

        def _wrap_to_action():
            sub_graph = IRLocalGraph(
                curr_nodes, self, device=curr_device[0]  #FIXME
            )
            action = Action(sub_graph, device=curr_device[0])  #FIXME
            action.tag(self.tag)
            return action

        for node in self.nodes():
            #FIXME: will fail in multi-branch placement (backward)
            device = node.device
            if len(node.device) == 0:
                raise RuntimeError("All the node should be assigned to devices")
            if device != curr_device and curr_device is not None:
                # note we still use same input and output to make consistency
                action = _wrap_to_action()
                # register to schedule space
                TSchedulePool().add_action(action)
                curr_nodes = list()
            curr_device = device
            curr_nodes.append(node)
        if curr_device is not None:
            action = _wrap_to_action()
            TSchedulePool().add_action(action)

        return self.forward(*args, **kwargs)

    def backward(self):
        """
        Backward will generate a backward action scheduling pool

        Construct a reverse graph of forward and seperate to actions
        """
        # travel graph in reverse order
        all_tensors = dict()

        def get_tensor_grad(tensor):
            if tensor._id not in all_tensors:
                new_tensor = copy.deepcopy(tensor)
                if tensor.name is None:
                    new_tensor.name = 'grad'
                else:
                    new_tensor.name = tensor.name + '_grad'
                new_tensor._src_nodes = list()
                new_tensor._dst_nodes = list()
                # reverse op
                devices = set()
                for node in tensor.dst():
                    devices.update(node.device)
                new_tensor.device = list(devices)
                all_tensors[tensor._id] = new_tensor
                return new_tensor
            else:
                return all_tensors[tensor._id]

        backward_nodes = list()
        for fnode in self._nodes[::-1]:
            inputs = list()
            for input in fnode.outputs():
                if isinstance(input, IRTensor) and input.requires_grad:
                    tensor = get_tensor_grad(input)
                    inputs.append(tensor)
                else:
                    inputs.append(None)
            outputs = list()
            for output in fnode.inputs():
                if isinstance(output, IRTensor) and output.requires_grad:
                    tensor = get_tensor_grad(output)
                    outputs.append(tensor)
                else:
                    outputs.append(None)
            bp_node = IROperation(
                name = fnode.name + '_backward',
                signature = fnode.signature,
                input_length = len(inputs),
                output_length = len(outputs),
                type=fnode.type
            )
            bp_node.device = fnode.device
            print(bp_node)
            for idx, arg in enumerate(inputs):
                bp_node.set_input(idx, arg)
            for idx, arg in enumerate(outputs):
                bp_node.set_output(idx, arg)
            backward_nodes.append(bp_node)
        # none inputs for loss
        inputs = list()
        # none outputs for loss
        outputs = list()
        graph = IRGraph(
            backward_nodes,
            inputs, outputs,
            self.module_name + 'Backward'
        )
        print(graph)
        graph.tag = 'backward'
        graph()
        

    def __repr__(self):
        dscp = ''
        # inputs
        dscp += f'Inputs: {self._inputs}\n'
        # nodes
        for node in self._nodes:
            succ_node_ids = [None] * len(node.outputs())
            for out_idx, node_list in enumerate(node.successors()):
                node_list = [snode._id for snode in node_list]
                succ_node_ids[out_idx] = node_list
            dscp += f"\n{node._id}: {node} -> node id {succ_node_ids}\n"
        # outputs
        dscp += f'\nOutputs: {self._outputs}'
        return dscp


class IRLocalGraph(IRGraph):

    def __init__(self, 
                 sub_nodes: List[IROperation],
                 global_graph: IRGraph,
                 device: int
        ):

        if not isinstance(global_graph, IRGraph):
            raise TypeError(f"Expected graph: IRGraph but go {type(global_graph)}")
        if not isinstance(device, int):
            raise TypeError(f"Expected device: int but not {type(device)}")
        for node in sub_nodes:
            if not node.on_device(device):
                raise RuntimeError(f"Local Graph requires all nodes on device {device}")
        self.global_graph = global_graph
        self.device = device
        self.send_tensors = list()
        self.send_devices = list()
        self.recv_tensors = list()
        self.recv_devices = list()
        # get nodes belong to this graph
        all_tensors = list()
        for node in sub_nodes:
            # collect recv tensors
            for input in node.inputs():
                if isinstance(input, IRTensor):
                    if self.device not in input.device:
                        if input not in self.recv_tensors:
                            self.recv_tensors.append(input)
                            self.recv_devices.append(self.device)
            # collect send tensors
            for output in node.outputs():
                if isinstance(output, IRTensor):
                    succ_nodes = output.dst()
                    for succ_node in succ_nodes:
                        if not succ_node.on_device(self.device):
                            if output not in self.send_tensors:
                                self.send_tensors.append(output)
                                self.send_devices.append(succ_node.device)
            # move semantic
            # if node.semantic == 'move':
            #     if device in node.inputs(0).device:
            #         self.send_tensors.append(node.inputs(0))
            #         self.send_devices.append(node.outputs(0).device)
            #     if device in node.outputs(0).device:
            #         self.recv_tensors.append(node.outputs(0))
            #         self.recv_devices.append(node.inputs(0).device)
            all_tensors += node.inputs()
            all_tensors += node.outputs()

        # model inputs and outputs
        model_inputs = list()
        model_outputs = list()
        for input in self.global_graph.inputs():
            if input in all_tensors and input not in self.recv_tensors:
                model_inputs.append(input)
        for output in self.global_graph.outputs():
            if output in all_tensors and output not in self.send_tensors:
                model_outputs.append(output)

        super().__init__(
            sub_nodes,
            model_inputs + self.recv_tensors,  # input tensors
            model_outputs + self.send_tensors,  # output tensors
            self.global_graph.module_name + f'Rank{self.device}'
        )
