
"""
Holistic Operator Generics

The holistic operator needed to be registered into logical op

The output communication works in a lazy execution way. Communication will only
happen in the front of the next executed op in case the layout doesn't match.
"""

from cube.tensor.logic.tensor import LogicalTensor
from cube.tensor.logic.outline import BaseOutline

import z3


class GenericHolisticOp:

    def __init__(self, shapes):
        """
        Layout is the community distribution requirement for input and
        output logical tensors.

        Format is the dimension ordering based on the logical format,
        `None` indicates the format is consistent with logical op,
        otherwise should be a list of integers like torch.Tensor.permute()
        on the logical required format.

        Args:
            input_layout (list[Outliner, None]): outliner for each input
                The length of outliner should be equal to the number of input
            input_format (list[list[int], None]): 
                input dim order compare with logical definition
            output_layout (list[Outlinter, None]): outliner for each output
                The length of outliner should be equal to the number of output
            output_format (list[list[int], None]):
                output dim order compare with logical definition
        """
        self.solver = z3.Solver()
        self.shapes = shapes

        self.input_layouts = list()
        self.output_layouts = list()

        self.logical_op = None
        self.output_shapes = list()

        self.attributes = list()
        self.policy_fn = None
        self.config = None
    
    def set_input_layouts(self, layouts):
        """
        Set input layout

        Args:
            layouts (list[BaseOutline]): layout list for input logical tensor
        """
        for layout in layouts:
            if not isinstance(layout, BaseOutline):
                TypeError("Require input layout for HolistOp is a list[BaseOutline]")
            self.attributes += layout.get_attributes()
            self.input_layouts.append(layout)
    
    def set_output_layouts(self, layouts):
        """
        Set output layout

        Args:
            layouts (list[BaseOutline]): layout list for output logical tensor
        """
        for layout in layouts:
            if not isinstance(layout, BaseOutline):
                TypeError("Require input layout for HolistOp is a list[BaseOutline]")
            self.attributes += layout.get_attributes()
            self.output_layouts.append(layout)
    
    def set_logic_op(self, logic_op):
        """
        Set logic op. This will be automatically called when the
        holistic op registered in a logical op.
        """
        # if not isinstance(logic_op, GenericLogicalOp):
        #     raise TypeError("Require a logic op to register")
        self.logical_op = logic_op

    def set_config(self, config):
        if not isinstance(config, z3.z3.ModelRef):
            raise TypeError("Expected config from z3 solver.model()")
        self.config = config
    
    def input_adapter(self, *args, **kwargs):
        """
        Transform tensors in args and kwargs to match the
        input layout requirement, Currently kwargs is not allowed to
        have tensors
        """
        #TODO: kwargs

        input_num = len(args)
        if len(self.input_layouts) != input_num:
            raise RuntimeError("Fail to adapt input: layout length not equal")
        # if len(self.input_format) != input_num:
        #     raise RuntimeError("Fail to adapt input: format length not equal")
        
        # step 1: data reformat based on the input argument
        # for input, dim_order in zip(args, self.input_format):
        #     if dim_order is not None:
        #         input.permute(dim_order)

        # step 2: Policy: segmentation + deploy decision
        if self.policy_fn is None:
            raise RuntimeError("Expected a runtime configuration policy")
        config, input_ranks = self.policy_fn[0](self)
        self.set_config(config)

        # step 3: segmentation
        input_segments = list()
        for tensor, outliner in zip(args, self.input_layouts):
            if outliner is not None and isinstance(tensor, LogicalTensor):
                segments = outliner.interpret(tensor, self.config)
                input_segments.append(segments)
            else:
                input_segments.append(None)

        # step 4: deploy
        for tid in range(len(args)):
            tensor = args[tid]
            if isinstance(tensor, LogicalTensor):
                segments = input_segments[tid]
                ranks = input_ranks[tid]
                tensor.transform(segments, ranks)

    def forward(self, *args, **kwargs):
        """
        Expert code for doing operation
        Call to the physical operator for execution

        Expert needs to gurantee the returned value is list[tuple(OpResult,),]

        Each item in list is the corresponding output to logical op output.

        Each item in the logical op output is a OpResult to the segment specified
        by the expert. The order should be consistent with specified segment.
        """
        raise NotImplementedError("Error call to generics")

    def output_adapter(self, outputs):
        """
        Data reformat to logical op format

        Args:
            outputs (tuple(list[OpResult],))
                each `list[OpResult]` represents a output of the op
                with its segments
        Returns:
            logical outputs (tuple(LogicalTensor,)):
                the logical tensor list
        """
        #TODO: fix: data re-format order. Should be ahead of logical tensor construction
        if not isinstance(outputs, tuple):
            outputs = (outputs,)

        # step 1: construct to logical tensor
        for output, outliner in zip(outputs, self.output_layouts):
            logical_tensor = LogicalTensor(outliner.shape, init_data=False)
            segments = outliner.interpret(shape, self.config)
            for segment in segments:
                logical_tensor.add_segment(segment)
            logical_tensor.fill(
                physical_tensors=[op_res.res for op_res in output],
                ranks=[op_res.placement for op_res in output]
            )
            logical_outputs.append(logical_tensor)

        # step 2: data reformat based on the output
        # for out_id in range(len(self.output_format)):
        #     dim_order = self.output_format[out_id]
        #     if dim_order is not None and isinstance(logical_outputs[out_id], LogicalTensor):
        #         logical_ouputs[out_id] = logical_ouputs[out_id].permute(dim_order)
    
        if len(logical_outputs) == 1:
            return logical_outputs[0]
        else:
            return tuple(logical_outputs)

    def __call__(self, *args, **kwargs):

        # data transformations to match input layout requirement
        self.input_adapter(*args, **kwargs)

        # do execution
        outputs = self.forward(*args, **kwargs)

        # wrap to logical tensor
        outputs = self.output_adapter(outputs)

        return outputs

    def set_policy(self, policy_fn):
        """
        Register a policy to take layouts and solver,
        generate device placement for each community, and corresponding
        message mapping

        Args:
            plicy_fn (callable)
        """
        if not callable(policy_fn):
            raise TypeError("Expected callable function")
        self.policy_fn = (policy_fn,)

