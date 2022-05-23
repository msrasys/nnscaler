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

        # TODO:
        input_format (list[list[int], None]): 
                input dim order compare with logical definition
        output_format (list[list[int], None]):
                output dim order compare with logical definition
        """
        if not isinstance(node, IRCell):
            raise TypeError("Expected node to be IRCell")
        self._node = node

    @property
    def node(self) -> IRCell:
        return self._node

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