from typing import Dict

from cube.ir.cten import IRCell, IRTensor


class GenericDistAlgo:

    def __init__(self, node: IRCell):
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
        if not isinstance(node, IRCell):
            raise TypeError("Expected node to be IRCell")

        input_shapes = list()
        for input in node.inputs():
            if isinstance(input, IRTensor):
                input_shapes.append(input.shape)
            else:
                input_shapes.append(None)
        output_shapes = list()
        for output in node.outputs():
            if isinstance(output, IRTensor):
                output_shapes.append(output.shape)
            else:
                output_shapes.append(None)

        self.input_shapes = input_shapes
        self.output_shapes = output_shapes

        self._logical_op = node

    @property
    def logic_op(self):
        return self._logical_op

    def satisfy(self, config: Dict):
        """
        Check if the config satisfies instantiation conditions
        """
        raise NotImplementedError

    def instantiate(self, node, config: Dict):
        """
        Instantiate the algorithm given the config
        """
        raise NotImplementedError