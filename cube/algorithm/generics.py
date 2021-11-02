from typing import List, Dict, Optional


class GenericDistAlgo:

    def __init__(self,
                 input_shapes: List[Optional[List[int]]],
                 output_shapes: List[List[int]]):
        """
        Layout is the community distribution requirement for input and
        output logical tensors.

        Format is the dimension ordering based on the logical format,
        `None` indicates the format is consistent with logical op,
        otherwise should be a list of integers like torch.Tensor.permute()
        on the logical required format.

        Args:
            input_layout (list[Outliner, None]): outliner for each input.
                The length of outliner should be equal to the number of input
            output_layout (list[Outlinter, None]): outliner for each output
                The length of outliner should be equal to the number of output
        # TODO:
        input_format (list[list[int], None]): 
                input dim order compare with logical definition
        output_format (list[list[int], None]):
                output dim order compare with logical definition
        """

        self.input_shapes = input_shapes
        self.output_shapes = output_shapes

        self.logical_op = None

    def set_logic_op(self, logic_op):
        """
        Set logic op. This will be automatically called when the
        holistic op registered in a logical op.
        """
        # if not isinstance(logic_op, GenericLogicalOp):
        #     raise TypeError("Require a logic op to register")
        self.logical_op = logic_op

    def satisfy(self, config: Dict):
        """
        Check if the config satisfies instantiation conditions
        """
        raise NotImplementedError

    def instantiate(self, config: Dict):
        """
        Instantiate the algorithm given the config
        """
        raise NotImplementedError