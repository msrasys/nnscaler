
"""
Holistic Operator Generics

The holistic operator needed to be registered into logical op

The output communication works in a lazy execution way. Communication will only
happen in the front of the next executed op in case the layout doesn't match.
"""

from cube.tensor.logic.tensor import LogicalTensor
from cube.tensor.community import Community


class GenericHolisticOp:

    def __init__(self, 
                input_layout, output_layout,
                input_format=None, output_format=None
        ):
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

        # holistic layout (outliner) of input
        if not isinstance(input_layout, list):
            raise TypeError("Require input layout for HolistOp is a list")
        if not isinstance(input_format, list):
            raise TypeError("Require input format for HolistOp is a list")
        if not isinstance(output_layout, list):
            raise TypeError("Require output layout for HolistOp is a list")
        if not isinstance(output_format, list):
            raise TypeError("Require output format for HolistOp is a list")

        self.input_layout = input_layout
        self.input_format = input_format

        # holistic layout of output
        self.output_layout = output_layout
        self.output_format = output_format

        self.logical_op = None
        self.policy_fn = None
    
    def set_logic_op(self, logic_op):
        """
        Set logic op. This will be automatically called when the
        holistic op registered in a logical op.
        """
        self.logical_op = logic_op
    
    def input_adapter(self, *args, **kwargs):
        """
        Transform tensors in args and kwargs to match the
        input layout requirement, Currently kwargs is not allowed to
        have tensors
        """
        #TODO: kwargs

        input_num = len(args)
        if len(self.input_layout) != input_num:
            raise RuntimeError("Fail to adapt input: layout length not equal")
        if len(self.input_format) != input_num:
            raise RuntimeError("Fail to adapt input: format length not equal")
        
        # step 1: data reformat based on the input argument
        for input, dim_order in zip(args, self.input_format):
            if dim_order is not None:
                input.permute(dim_order)

        # step 2: get communities based on expert description
        input_communities = list()
        for tensor, outliner in zip(args, self.input_layout):
            if outliner is not None and isinstance(tensor, LogicalTensor):
                segments = outliner(tensor.shape)
                communities = [Community(seg) for seg in segments]
                input_communities.append(communities)
            else:
                input_communities.append(None)

        # step 3: physical tensor placement (policy)
        if self.policy_fn is not None:
            input_ranks, input_val_map_fns = \
                self.policy_fn[0](input_communities, *args)
        else:
            # TODO: default policy
            input_ranks = [None] * len(args)
            input_val_map_fns = [None] * len(args)

        # step 4: community matching
        for tid in range(len(args)):
            tensor = args[tid]
            if isinstance(tensor, LogicalTensor):
                communities = input_communities[tid]
                ranks = input_ranks[tid]
                val_map_fn = input_val_map_fns[tid]
                tensor.match(communities, ranks, val_map_fn)

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
            outputs (tuple(list[physical_tensor],))
                each `list[physical_tensor]` represents a output of the op
                with is communities
        Returns:
            logical outputs (tuple(LogicalTensor,)):
                the logical tensor list
        """
        #TODO: fix: data re-format order. Should be ahead of logical tensor construction
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        # step 1: construct to logical tensor
        logical_outputs = list()
        for output, outliner, shape in zip(outputs, self.output_layout, self.logical_shapes):
            segments = outliner(shape)
            communities = [Community(segment) for segment in segments]
            for community, op_res in zip(communities, output):
                #if DeviceGroup().rank == 0:
                #    print(op_res.res.size(), community.segment.shape)
                community.set_physical_tensor(op_res.res, op_res.placement)
            output = LogicalTensor.construct(shape, communities)
            logical_outputs.append(output)
        # step 2: data reformat based on the output
        for out_id in range(len(self.output_format)):
            dim_order = self.output_format[out_id]
            if dim_order is not None and isinstance(logical_outputs[out_id], LogicalTensor):
                logical_ouputs[out_id] = logical_ouputs[out_id].permute(dim_order)
    
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
        if self.logical_op is None:
            raise RuntimeError("This holistic op doesn't have logical op")
        self.logical_shapes = self.logical_op.shape_infer(*args, **kwargs)
        outputs = self.output_adapter(outputs)

        return outputs

    def set_deploy_policy(self, policy_fn):
        """
        Register a policy to take inputs (logical tensors) and segments,
        generate device placement for each community, and corresponding
        message mapping

        Args:
            plicy_fn (callable)
        """
        if not callable(policy_fn):
            raise TypeError("Expected callable function")
        self.policy_fn = (policy_fn,)
    
    def set_segmentation_policy(self, policy_fn):
        for outliner in self.input_layout:
            outliner.set_policy(policy_fn)
        for outliner in self.output_layout:
            outliner.set_policy(policy_fn)

