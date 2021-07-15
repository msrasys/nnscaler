from combo.operator.logic.generics import generics
from combo.operator.holist.linear import kHolistLinearSets

__all__ = ['linear']

def Linear(generics.GenericLogicalOp):

    def __init__(self):
        super().__init__(self)
        
        # register holistic operators
        for holist_op in kHolistLinearSets:
            self.factory.register(holist_op)

# initialize op
linear = Linear()