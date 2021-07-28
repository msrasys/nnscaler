from cube.operator.physic.comm import replicate, reduce_sum
import torch


# TODO: reduction op should be in torch autograd function
class _Reduction(type):

    # forward: all_reduce, backward: identity
    Sum = (reduce_sum,)

    # forward: identity, backward: all_reduce
    Replica = (replicate,)

    def register(cls, name, udf):
        """
        Reduction functions should be in function format:

        Arguments:
            PhysicalTensor
            Communication Group

        Return:
            PhysicalTensor
        """
        if hasattr(cls, name):
            raise KeyError("{} is registered".format(name))
        setattr(cls, name, (udf,))


class ReductionOp(metaclass=_Reduction):
    pass
