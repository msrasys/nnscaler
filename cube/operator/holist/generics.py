
"""
Holistic Operator Generics

The holistic operator needed to be registered into logical op

The output communication works in a lazy execution way. Communication will only
happen in the front of the next executed op in case the layout doesn't match.
"""

class GenericHolisticOp:

    def __init__(self, 
                input_layout, output_layout,
                input_format=None, output_format=None):
        """
        Layout is the community distribution requirement for input and
        output logical tensors.

        Format is the dimension ordering based on the logical format,
        `None` indicates the format is consistent with logical op,
        otherwise should be a list of integers like torch.Tensor.permute()
        on the logical required format.
        """
        # holistic layout of input
        self.input_layout = dict()
        self.input_format = input_format

        # holistic layout of output
        self.output_layout = dict()
        self.output_format = output_format
    
    def input_adapter(self, args, **kwargs):
        """
        Transform tensors in args and kwargs to match the
        input layout requirement
        """
        # step 1: data reformat based on the input argument

        # step 2: physical tensor placement (policy)

        # step 3: community matching 
        pass

    def output_adapter(self, outputs):
        """
        Data reformat to logical op format
        """
        pass

    def forward(self, args, **kwargs):
        """Expert code for doing operation
        Call to the physical operator for execution"""
        pass

    def __call__(self, args, **kwargs):

        # data transformations to match input layout requirement
        self.input_adapter(args, kwargs)

        # do execution
        outputs = self.forward(args, kwargs)

        # wrap in holistic tensor with output layout
        outputs = self.output_adapter(outputs)

        return outputs
