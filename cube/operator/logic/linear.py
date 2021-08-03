from cube.operator.logic.generics import GenericLogicalOp
from cube.operator.holist.linear import kHolistLinearSets


class Linear(GenericLogicalOp):

    def __init__(self):
        super().__init__()
        
        # register holistic operators
        for holist_op in kHolistLinearSets:
            self.factory.register(holist_op)

    def shape_infer(self, input, weight, bias=None):
        """
        Return the outputs shape [list[int],]
        """
        output_shape = list(input.shape)
        output_shape[-1] = weight.shape[0]
        return [output_shape,]
