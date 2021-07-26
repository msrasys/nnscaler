from cube.operator.logic.generics import generics
from cube.operator.holist.linear import kHolistLinearSets


__all__ = ['Linear']


def Linear(generics.GenericLogicalOp):

    def __init__(self):
        super().__init__(self)
        
        # register holistic operators
        for holist_op in kHolistLinearSets:
            holist_op.set_logic_op(self)
            self.factory.register(holist_op)

    def shape_infer(self, input_shape, weight_shape, bias_shape=None)
        """
        Return the outputs shape [list[int],]
        """
        output_shape = list(input_shape)
        output_shape[-1] = weight_shape[-1]
        return [output_shape]

