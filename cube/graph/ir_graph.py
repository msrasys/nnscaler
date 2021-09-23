from cube.graph.ir_cten import IRTensor, IRCell
from cube.graph.ir_op import IROperation
from cube.tschedule.pool import TSchedulePool

from typing import Union, Tuple, List, Optional, Any


__all__ = ['IRGraph', 'IRAction']


class IRGraph(IRCell):
    """
    PyTorch IR Graph

    The IR Graph only contains forward graph
    """

    def __init__(self, 
                 nodes: List[IROperation],
                 input_tensors: List[IRTensor], 
                 output_tensors: List[IRTensor], 
                 module_name: str):
        
        self._nodes: List[IROperation] = nodes
        super().__init__(
            name=module_name,
            signature=module_name,
            input_length=len(input_tensors),
            output_length=len(output_tensors)
        )
        self._inputs = input_tensors
        self._outputs = output_tensors
        self.tag = 'forward'

    def add_node(self, node: IRCell):
        if not isinstance(node, IRCell):
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

    def replace(self, target: IROperation, nodes: List[IROperation]):
        """
        Replace the node with new nodes (IRGraph)
        """
        raise NotImplementedError

    def forward(self, *args) -> Union[IRTensor, Tuple[IRTensor]]:
        """
        forward will divide the graph into Actions according to
        node device assignment

        Currently each forward call will result in a new flow
        even if the input is same

        Returns:
            List[Action]
        """
        # check input num
        if len(args) != len(self.inputs()):
            raise RuntimeError(
                f"Expected {len(self.inputs())} input args but got {len(args)}"
            )
        # check input type
        if not all([type(arg) is type(input) for arg, input in zip(args, self.inputs())]):
            raise RuntimeError(f"Expected input type the same")
        
        curr_nodes: List[IROperation] = list()
        curr_device = list()

        total_actions = list()
        for node in self.nodes():
            device = node.device
            if len(node.device) == 0:
                raise RuntimeError("All the node should be assigned to devices")
            if set(device) != set(curr_device) and len(curr_device) != 0:
                # create action
                action = IRAction(curr_nodes, self, devices=curr_device)
                total_actions.append(action)
                # register to schedule space
                TSchedulePool().add_action(action)
                curr_nodes = list()
            curr_device = device
            curr_nodes.append(node)
        if curr_device is not None:
            action = IRAction(curr_nodes, self, devices=curr_device)
            total_actions.append(action)
            TSchedulePool().add_action(action)

        # setup action inputs
        head = total_actions[0]
        for idx, arg in enumerate(args):
            head.set_input(idx, arg)
        outputs_tensors = [*head.graph.outputs()]
        outputs_actions = [head] * len(head.graph.outputs())
        for action in total_actions[1:]:
            for idx, input in enumerate(action.graph.inputs()):
                if input not in outputs_tensors:
                    raise RuntimeError(f"Cannot find {input} tensors")
                pre_action = outputs_actions[outputs_tensors.index(input)]
                val = pre_action.map_output(input)
                action.set_input(idx, val)
            outputs_tensors += action.graph.outputs()
            outputs_actions += [action] * len(action.graph.outputs())

        # return tensors
        outputs = tuple(total_actions[-1].outputs())
        for output in outputs:
            output.set_gen_graph(self)
        if len(outputs) == 1:
            return outputs[0]
        elif len(outputs) == 0:
            return None
        else:
            return outputs

    def __call__(self, *args):
        """
        Register forward action
        """
        return self.forward(*args)

    def backward(self, loss: IRTensor):
        """
        Backward will generate a backward action scheduling pool

        Construct a reverse graph of forward and seperate to actions
        """
        # travel graph in reverse order
        all_tensors = dict()

        def get_tensor_grad(tensor):
            if tensor._id not in all_tensors:
                #name = 'grad' if tensor.name is None else tensor.name + '_grad'
                new_tensor = IRTensor(
                    shape=tensor.shape, name=tensor.name
                )
                new_tensor._id = tensor._id  # -> keep same tensor
                # reverse op
                devices = set()
                for node in tensor.dst():
                    devices.update(node.device)
                new_tensor.device = list(devices)
                all_tensors[tensor._id] = new_tensor
                return new_tensor
            else:
                return all_tensors[tensor._id]

        # backward graph inputs
        graph_inputs = list()
        # none outputs for loss
        graph_outputs = list()
        # nodes
        backward_nodes = list()
        all_bp_tensors = list()
        for fnode in self._nodes[::-1]:
            inputs = list()
            for input in fnode.outputs():
                if isinstance(input, IRTensor) and input.requires_grad:
                    tensor = get_tensor_grad(input)
                    if tensor not in all_bp_tensors:
                        graph_inputs.append(tensor)
                        all_bp_tensors.append(tensor)
                    inputs.append(tensor)
                else:
                    inputs.append(None)
            outputs = list()
            for output in fnode.inputs():
                if isinstance(output, IRTensor) and output.requires_grad:
                    tensor = get_tensor_grad(output)
                    all_bp_tensors.append(tensor)
                    outputs.append(tensor)
                else:
                    outputs.append(None)
            bp_node = IROperation(
                name = fnode.name + '_backward',
                signature = fnode.signature,
                input_length = len(inputs),
                output_length = len(outputs)
            )
            bp_node.device = fnode.device
            for idx, arg in enumerate(inputs):
                bp_node.set_input(idx, arg)
            for idx, arg in enumerate(outputs):
                bp_node.set_output(idx, arg)
            backward_nodes.append(bp_node)
        graph = IRGraph(
            backward_nodes,
            graph_inputs, graph_outputs,
            self.name + 'Backward'
        )
        print(graph)
        graph.tag = 'backward'
        graph(loss)

    def __repr__(self):
        dscp = ''
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
        dscp += f'\nOutputs: {self._outputs}'
        return dscp


# outputs = cube.runtime.temporal.forward(model, *args)
_forward_signature = 'cube.runtime.temporal.forward'
# grads = cube.runtime.temporal.backward(input_tensors, output_tensors, output_grads)
_backward_signature = 'cube.runtime.temporal.backward'


class IRAction(IRCell):

    def __init__(self, sub_nodes, global_graph, devices: Union[List[int], int]):

        if isinstance(devices, int):
            devices = [devices]

        if not isinstance(global_graph, IRGraph):
            raise TypeError(f"Expected graph: IRGraph but go {type(global_graph)}")

        if global_graph.tag == 'forward':
            signature = _forward_signature
        elif global_graph.tag == 'backward':
            signature = _backward_signature
        else:
            raise RuntimeError(f"Unsupported graph tag: {self.global_graph.tag}")

        # send tensors
        self.send_tensors = list()
        self.send_devices = list()

        # recv tensors
        self.recv_tensors = list()
        self.recv_devices = list()

        # get nodes belong to this graph
        all_tensors = list()
        for node in sub_nodes:
            # collect recv tensors
            for input in node.inputs():
                if isinstance(input, IRTensor):
                    recv_devices = list(set(devices) - set(input.device))
                    if len(recv_devices) != 0:
                        if input not in self.recv_tensors:
                            self.recv_tensors.append(input)
                            self.recv_devices.append(recv_devices)
            # collect send tensors
            for output in node.outputs():
                if isinstance(output, IRTensor):
                    succ_nodes = output.dst()
                    for succ_node in succ_nodes:
                        send_devices = list(set(devices) - set(succ_node.device))
                        if len(send_devices) != 0:
                            if output not in self.send_tensors:
                                self.send_tensors.append(output)
                                self.send_devices.append(send_devices)
            all_tensors += node.inputs()
            all_tensors += node.outputs()

        # action graph inputs and outputs
        inputs = list()
        outputs = list()
        for input in global_graph.inputs():
            if input in all_tensors and input not in self.recv_tensors:
                inputs.append(input)
        for output in global_graph.outputs():
            if output in all_tensors and output not in self.send_tensors:
                outputs.append(output)

        self.graph = IRGraph(
            nodes = sub_nodes,
            input_tensors = inputs + self.recv_tensors,
            output_tensors = outputs + self.send_tensors,
            module_name = global_graph.name
        )

        action_inputs = [None] * len(self.graph.inputs())
        super().__init__(
            name          = global_graph.tag,
            signature     = signature,
            input_length  = len(action_inputs),
            output_length = len(self.graph.outputs())
        )
        self.device = devices
        self._inputs = action_inputs
        print(self.graph)

    def map_output(self, graph_output_tensor: Any) -> Any:
        if graph_output_tensor not in self.graph.outputs():
            return None
        index = self.graph.outputs().index(graph_output_tensor)
        return self.outputs(index)

    def happen_before(self, action):
        """
        Check if the self -> (happened before) action
        """
        if not isinstance(action, IRAction):
            raise TypeError("Expected action to be an Action")
        for pre_actions in self.successors():
            if action in pre_actions:
                return True
        return False

    def happen_after(self, action):
        """
        Check if the action -> (happened before) self

        Note: this may return false negative as it will only check
        1-hop dependency
        """
        if not isinstance(action, IRAction):
            raise TypeError("Expected action to be an Action")
        for pre_actions in self.predecessors():
            if action in pre_actions:
                return True
        return False

    def add_flow(self, action):
        """
        self -> (happened before) action
        """
        raise NotImplementedError

    def __repr__(self):
        action_inputs = [f't{tensor._id}' for tensor in self.inputs()]
        action_outputs = [f't{tensor._id}' for tensor in self.outputs()]
        dscp = f'Action({self.name}):\n\t{self.graph.inputs()} ({action_inputs}) -> {self.graph.outputs()} ({action_outputs})'
        return dscp
