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

    def register(self, holistic_op):
        """
        Register a holistic op as one of the anchors 
        """
        #TODO: type check
        self.holist_ops.append(holist_ops)

    def get_op(self, args, **kwargs):
        """
        Given input tensor args, choose holistic operator(s)
        for distributed execution plan

        Returns:
            An hybrid-operator function which may composite by
            nested holistic operators
        """
        # TODO: hybrid parallelism generation
        return self.holist_ops[0]

        

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
        self.policy_fn = policy_fn

    def __call__(self, args, **kwargs):
        """
        Policy here to determine which holistic operator(s) are called
        """
        # use default policy
        if self.policy_fn is None:
            composite_op = self.factory.get_op(args, kwargs)
        # use user-customized policy
        else:
            composite_op = self.policy_fn(self.factory)
        # run operator with the strategy plan
        outputs = composite_op(args, kwargs)
        return outputs