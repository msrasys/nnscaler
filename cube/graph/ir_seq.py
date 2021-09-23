from typing import List, Tuple, NewType, Any, Optional
import numpy as np

from cube.graph.ir_cten import IRCell, IRTensor
from cube.graph.ir_graph import IRAction


class IRSequence(IRCell):

    def __init__(self, actions: List[IRAction]):

        if not all([isinstance(action, IRAction) for action in actions]):
            raise TypeError("Expected a list of IRActions")

        super().__init__(
            name = 'action',
            signature = 'None',
            input_length = 0,
            output_length = 0
        )
        self.sequence = actions

    def __iter__(self):
        return iter(self.sequence)

    def __len__(self):
        return len(self.sequence)

    def append(self, action: IRAction):
        self.sequence.append(action)

    def get_forward_inputs(self, action: IRAction) -> List[Any]:
        """
        Get corresponding forward action inputs

        The backward graph output tensor shuould be forward graph input tensor
        """
        if action.name == 'forward':
            return action.inputs()
        if action.name == 'backward':
            bp_graph_outputs = action.graph.outputs()
            fw_action_inputs = [None] * len(bp_graph_outputs)
            pre_actions = action.predecessors()
            while len(pre_actions) != 0:
                pre = list()
                for pre_action in pre_actions:
                    if pre_action.name == 'forward':
                        for bidx, output in enumerate(bp_graph_outputs):
                            for fidx, input in enumerate(pre_action.graph.inputs()):
                                if input == output:
                                    fw_action_inputs[bidx] = pre_action.inputs(fidx)
                    pre += pre_action.predecessors()
                pre_actions = pre
            if None in fw_action_inputs:
                raise RuntimeError("Couldn't found forward inputs")
            return fw_action_inputs
        raise RuntimeError(f"Unsupported action name: {action.name}")
    
    def get_forward_outputs(self, action: IRAction) -> List[Any]:
        """
        Get corresponding forward action outputs

        The backward graph input tensor should be forward graph output tensor
        """
        if action.name == 'forward':
            return action.inputs()
        if action.name == 'backward':
            bp_graph_inputs = action.graph.inputs()
            fw_action_outputs = [None] * len(bp_graph_inputs)
            pre_actions = action.predecessors()
            while len(pre_actions) != 0:
                pre = list()
                for pre_action in pre_actions:
                    if pre_action.name == 'forward':
                        for bidx, output in enumerate(bp_graph_inputs):
                            for fidx, input in enumerate(pre_action.graph.outputs()):
                                if input == output:
                                    fw_action_outputs[bidx] = pre_action.outputs(fidx)
                    pre += pre_action.predecessors()
                pre_actions = pre
            if None in fw_action_outputs:
                raise RuntimeError("Couldn't found forward inputs")
            return fw_action_outputs
        raise RuntimeError(f"Unsupported action name: {action.name}")


    def is_correct(self):
        """
        Check whether sequence 
        satisfies the sequential consistency model
        """

        for index, action in enumerate(self.sequence):
            for pre_action in action.predecessors():
                # find the pre-action not appear in sequence
                if not pre_action in self.sequence:
                    return False
                pre_idx = self.sequence.index(pre_action)
                # violate sequential consistency model
                if pre_idx >= index:
                    return False
        return True