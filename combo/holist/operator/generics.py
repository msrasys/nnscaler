
"""
Holistic Operator Generics

The holistic operator needed to be registered into logical op
"""

class GenericHolisticOp:

    def __init__(self, input_layout, output_layout):

        # holistic layout of input to wark on
        self.input_layout = input_layout
        # expected holistic layout of output
        self.output_layout = output_layout

    def input_transform(self, args, **kwargs):
        """input transformation to the required layout"""
        pass

    def forward(self, args, **kwargs):
        """Expert code for doing operation"""
        pass

    def __call__(self, args, **kwargs):
        """Operator execution"""

        self.input_transform(args, kwargs)

        outputs = self.forward(args, kwargs)

        return outputs
