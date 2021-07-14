
"""
Holistic Operator Generics

The holistic operator needed to be registered into logical op

The output communication works in a lazy execution way. Communication will only
happen in the front of the next executed op in case the layout doesn't match.
"""

class GenericHolisticOp:

    def __init__(self, input_layout, output_layout):

        # holistic layout of input to work on
        self.input_layout = dict()

        # holistic layout of output
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

    def forward(self, args, **kwargs):
        """Expert code for doing operation
        Call to the physical operator for execution"""
        pass

    def __call__(self, args, **kwargs):

        # data transformations to match input layout requirement
        self.boundary_in(args, kwargs)

        # do execution
        outputs = self.forward(args, kwargs)

        # wrap in holistic tensor with output layout
        outputs = self.warp_to_holistic_tensor(outputs)

        return outputs
