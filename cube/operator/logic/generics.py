"""
A Logical Operator: 
    * Statusless
    * Can be executed by only one kernel (atomic) on single device

Logical Operator
    |- Holistic Operator 1
    |   |- Physical Operator(s)
    |- Holistic Operator 2
    |- ...

Holistic operators are allowed to nested in hybrid-distribution strategy
"""

from cube.tensor.logic.tensor import LogicalTensor


class HolisticOpFactory:

    def __init__(self):

        self.holist_ops = list()

    def __len__(self):
        """
        Return the number of holistic op registered
        """
        return len(self.holist_ops)

    def register(self, holistic_op):
        """
        Register a holistic op (class) as one of the anchors 
        """
        self.holist_ops.append(holistic_op)

    def get_op(self, idx, shapes):
        """
        Get holistic operator based on idx

        The holistic operator will be initialized with shapes

        Returns:
            HolisticOp instance
        """
        return self.holist_ops[idx](shapes)


class GenericLogicalOp:

    _default_policy_fn = None

    def __init__(self):

        # candidate holistic operator
        self.factory = HolisticOpFactory()
        self.policy_fn = None
    
    def shape_infer(self, *args, **kwargs):
        """
        Output shape inference according to inputs

        Args:
            Operator input

        Returns:
            shapes tuple(list[int]): shape for each output tensor
        """
        raise NotImplementedError("Expected a shape infer engine")

    def get_shapes(self, *args, **kwargs):
        # get shapes of input and output
        shapes = list()
        for arg in args:
            if isinstance(arg, LogicalTensor):
                shapes.append(arg.shape)
            else:
                shapes.append(None)
        shapes += self.shape_infer(*args, **kwargs)
        return shapes

    def get_op(self, *args, **kwargs):
        # get shapes of input and output
        shapes = self.get_shapes(*args, **kwargs)
        print(shapes)
        # use default policy
        if self.policy_fn is None:
            composite_op = self._default_policy_fn[0](self.factory, shapes)
        # use user-customized policy
        else:
            composite_op = self.policy_fn[0](self.factory, shapes)
        return composite_op

    def __call__(self, *args, **kwargs):
        """
        Policy here to determine which holistic operator(s) are called
        """
        composite_op = self.get_op(*args, **kwargs)
        # run operator with the strategy plan
        outputs = composite_op(*args, **kwargs)
        return outputs

    def set_policy(self, policy_fn):
        """
        Register a policy function to customize how composite
        holistic op generated during runtime.

        The `policy_fn` takes self.factory as input and returns a composite
        holistic operator (callable)
        """
        if not callable(policy_fn):
            raise TypeError("Expected a callable function")
        self.policy_fn = (policy_fn,)
    
    @classmethod
    def set_default_policy(self, policy_fn):
        """
        Register a default policy function to all instances.
        Customize how composite holistic op generated during runtime.

        The `policy_fn` takes self.factory and shapes as input,
        and returns a composite holistic operator (callable)
        """
        if not callable(policy_fn):
            raise TypeError("Expected a callable function")
        self._default_policy_fn = (policy_fn,)
