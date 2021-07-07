"""
Physical Generic Operator definition.

The output communication works in a lazy execution way. Communication will only
happen in the front of the next executed op in case the layout doesn't match.
"""

class GenericOp:

    def __init__(self, func):
        """
        func: Should be a logical operator handling holistic tensors.
        """

        # operator: take any inputs and generate output
        self.F = func

        # function inputs requirement
        self.input_layout = dict()

        # the expected function output holistic layout
        self.output_layout = dict()
    
    def boundary_in(self, args, **kwargs):
        """
        Transform tensors in args and kwargs to match the
        input layout requirement
        """
        pass

    def warp_to_holistic_tensor(self, outputs):
        """
        Wrap local computed tensor into a holistic view
        by using self.output_layout
        """
        pass

    def execute(self, args, **kwargs):

        # data transformations to match input layout requirement
        self.boundary_in(args, kwargs)

        # do execution
        outputs = self.F(args, kwargs)

        # wrap in holistic tensor with output layout
        outputs = self.warp_to_holistic_tensor(outputs)

        return outputs