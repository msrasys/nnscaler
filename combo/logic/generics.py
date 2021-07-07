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

    def composite_op(self, args, **kwargs):
        """
        Given input tensor args, choose holistic operator(s)
        for distributed execution plan

        Returns:
            An hybrid-operator function which may composite by
            nested holistic operators
        """
        pass

        

class GenericLogicalOp:

    def __init__(self):

        # candidate holistic operator
        self.factory = HolisticOpFactory()

    def __call__(self, args, **kwargs):
        """
        Policy here to determine which holistic operator(s) are called
        """
        pass