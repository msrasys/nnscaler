from typing import List, Any, Union

from cube.graph.ir_cten import IRCell, IRTensor
from cube.graph.ir_graph import IRGraph


__all__ = ['IRAction']

# outputs = cube.runtime.temporal.forward(model, *args)
__forward_signature = 'cube.runtime.temporal.forward'
# grads = cube.runtime.temporal.backward(input_tensors, output_tensors, output_grads)
__backward_signature = 'cube.runtime.temporal.backward'


class IRAction(IRCell):

    def __init__(self, sub_nodes, global_graph, devices: Union[List[int], int]):

        if isinstance(devices, int):
            devices = [devices]

        if not isinstance(global_graph, IRGraph):
            raise TypeError(f"Expected graph: IRGraph but go {type(global_graph)}")

        if global_graph.tag == 'forward':
            signature = __forward_signature
        elif global_graph.tag == 'backward':
            signature = __backward_signature
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
                            self.recv_devices += recv_devices
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

        action_inputs = [self.graph] + [None] * len(self.graph.inputs())
        super().__init__(
            name          = self.global_graph.tag,
            signature     = signature,
            input_length  = len(action_inputs),
            output_length = len(self.graph.outputs())
        )
        self.device = devices
        self._inputs = action_inputs

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
