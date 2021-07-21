
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
        self.input_layout = input_layout
        self.input_format = input_format

        # holistic layout of output
        self.output_layout = output_layout
        self.output_format = output_format
    
    def input_adapter(self, args, **kwargs):
        """
        Transform tensors in args and kwargs to match the
        input layout requirement
        """
        # step 1: data reformat based on the input argument
        #TODO: data dimension format transformation
        tensor_inputs = list()
        for arg in args:
            #TODO: kwargs
            if cube.is_tensor(arg):
                tensor_inputs.append(arg)
        tensor_segments = list()
        for outliner, tensor in zip(self.input_layout, tensor_inputs):
            segments = outliner(tensor.shape)
            tensor_segments.append(segments)

        # step 2: physical tensor placement (policy)
        #TODO: policy module
        tensor_communities = policy_module(tensor_segments)

        # step 3: community matching
        for communities, tensor in zip(tensor_communities, tensor_inputs):
            tensor.match(communities)

    def output_adapter(self, outputs):
        """
        Data reformat to logical op format
        """
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        output_tensors = list()
        for output in outputs:
            if cube.is_tensor(output):
                if cube.is_tensor(output):
                    output_tensors.append(output)
        for outliner, output in zip(self.output_layout, output_tensors):
            segments = outliner(output.shape)
            output.to_logic_tensor(segments)
        

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
