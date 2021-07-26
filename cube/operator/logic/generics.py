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
        Register a holistic op as one of the anchors 
        """
        self.holist_ops.append(holistic_op)

    def get_op(self, idx):
        """
        Get holistic operator based on idx

        Returns:
            HolisticOp instance
        """
        return self.holist_ops[idx]

        

class GenericLogicalOp:

    def __init__(self):

        # candidate holistic operator
        self.factory = HolisticOpFactory()
        self.policy_fn = None

    def register_policy(self, policy_fn):
        """
        Register a policy function to customize how composite
        holistic op generated during runtime.

        The `policy_fn` takes self.factory as input and returns a composite
        holistic operator (callable)
        """
        if not callable(policy_fn):
            raise TypeError("Expected a callable function")
        self.policy_fn = [policy_fn]
    
    def shape_infer(self, *args, **kwargs):
        """
        Output shape inference according to inputs

        Args:
            Operator input

        Returns:
            shapes tuple(list[int]): shape for each output tensor
        """
        raise NotImplementedError("Expected a shape infer engine")

    def get_op(self, *args, **kwargs):
        # use default policy
        if self.policy_fn is None:
            composite_op = self.factory.get_op(0)
        # use user-customized policy
        else:
            composite_op = self.policy_fn[0](self.factory, *args, **kwargs)
        return composite_op

    def __call__(self, *args, **kwargs):
        """
        Policy here to determine which holistic operator(s) are called
        """
        composite_op = self.get_op(*args, **kwargs)
        # run operator with the strategy plan
        outputs = composite_op(*args, **kwargs)
        return outputs