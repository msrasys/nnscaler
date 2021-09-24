from cube.graph.ir_cten import IRTensor, IRCell
from cube.graph.ir_op import IROperation

from typing import Union, Tuple, List, Optional


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
        from cube.tschedule.pool import TSchedulePool
        # check input num
        if len(args) != len(self.inputs()):
            raise RuntimeError(
                f"Expected {len(self.inputs())} input args but got {len(args)}"
            )
        # check input type
        if not all([type(arg) is type(input) for arg, input in zip(args, self.inputs())]):
            raise RuntimeError(f"Expected input type the same")
        
        curr_nodes: List[IRCell] = list()
        curr_device = self.nodes(0).device

        total_actions = list()
        for node in self.nodes():
            device = node.device
            if len(node.device) == 0:
                raise RuntimeError("All the node should be assigned to devices")
            if set(device) != set(curr_device):
                # create action
                action = IRAction(curr_nodes, self, devices=curr_device)
                total_actions.append(action)
                # register to schedule space
                TSchedulePool().add_action(action)
                # clear
                curr_nodes = list()
            curr_device = device
            curr_nodes.append(node)
        if curr_device is not None:
            action = IRAction(curr_nodes, self, devices=curr_device)
            total_actions.append(action)
            TSchedulePool().add_action(action)

        # setup action inputs
        output_map = {
            gten._id : aten for gten, aten in zip(self.inputs(), args)
        }
        for action in total_actions:
            for idx, input in enumerate(action.graph.inputs()):
                if isinstance(input, IRTensor):
                    input = output_map[input._id]
                action.set_input(idx, input)
            for action_out, graph_out in zip(action.outputs(), action.graph.outputs()):
                output_map[graph_out._id] = action_out

        # return tensors
        outputs = tuple(total_actions[-1].outputs())
        for output in outputs:
            output.set_gen_graph(self)

        if    len(outputs) == 1: return outputs[0]
        elif  len(outputs) == 0: return None
        else: return outputs

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
                devices = list(devices)
                if len(devices) == 0:
                    devices = tensor.device
                new_tensor.device = devices
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
        graph.tag = 'backward'
        graph(loss)

    def subgraph(self, sub_nodes: List[IRCell]):
        """
        Create a subgraph with sub nodes.

        The remote tensor will be set as graph input (recv tensors)
        and graph output (send tensors)

        Return:
            IRGraph,
            recv tensor starting offset (int) in input,
            send tensor starting offset (int) in output
        """
        def _update(x_tensors, x_devices, tensor, devices):
            if tensor not in x_tensors:
                x_tensors.append(tensor)
                x_devices.append(set(devices))
            else:
                idx = x_tensors.index(tensor)
                x_devices[idx].update(set(devices))

        # recv tensors
        recv_tensors = list()
        recv_devices = list()
        # send tensors
        send_tensors = list()
        send_devices = list()
        # get nodes belong to this graph
        all_tensors = list()
        for node in sub_nodes:
            # collect recv tensors
            tensors_and_devices = node.get_recv_tensors()
            for r_tensor, r_devices in zip(*tensors_and_devices):
                _update(recv_tensors, recv_devices, r_tensor, r_devices)
            # collect send tensors
            tensors_and_devices = node.get_send_tensors()
            for s_tensor, s_devices in zip(*tensors_and_devices):
                _update(send_tensors, send_devices, s_tensor, s_devices)
            all_tensors += node.inputs()
            all_tensors += node.outputs()

        # set extra graph inputs and outputs
        inputs = list()
        outputs = list()
        for input in self.inputs():
            if input in all_tensors and input not in recv_tensors:
                inputs.append(input)
        for output in self.outputs():
            if output in all_tensors and output not in send_tensors:
                outputs.append(output)

        graph = IRGraph(
            nodes = sub_nodes,
            input_tensors = inputs + recv_tensors,
            output_tensors = outputs + send_tensors,
            module_name = self.name
        )

        return graph, len(inputs), len(outputs)


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


# outputs = cube.runtime.temporal.forward(model, *args)
_forward_signature = 'cube.runtime.temporal.forward'
# grads = cube.runtime.temporal.backward(input_tensors, output_tensors, output_grads)
_backward_signature = 'cube.runtime.temporal.backward'


class IRAction(IRCell):
    """
    Action recv tensors must be inside of Action inputs,
    and can be mapped to Action.graph.inputs

    """

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

        self.graph, recv_ofst, send_ofst = global_graph.subgraph(sub_nodes)
        self._recv_ofst = recv_ofst
        self._send_ofst = send_ofst

        super().__init__(
            name          = global_graph.tag,
            signature     = signature,
            input_length  = len(self.graph.inputs()),
            output_length = len(self.graph.outputs())
        )
        # set action device
        self.device = devices
        # set output shape
        for output, g_out in zip(self.outputs(), self.graph.outputs()):
            output.device = devices
            output.shape = g_out.shape

    @property
    def send_tensors(self):
        return self._outputs[self._send_ofst:]
    
    @property
    def recv_tensors(self):
        return self._inputs[self._recv_ofst:]

    def happen_before(self, action):
        """
        Check if the self -> (happened before) action

        Note: this may return false negative as it will only check
        1-hop dependency
        """
        if not isinstance(action, IRAction):
            raise TypeError("Expected action to be an Action")
        return self in action.predecessors()

    def happen_after(self, action):
        """
        Check if the action -> (happened before) self

        Note: this may return false negative as it will only check
        1-hop dependency
        """
        if not isinstance(action, IRAction):
            raise TypeError("Expected action to be an Action")
        return self in action.successors()

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
