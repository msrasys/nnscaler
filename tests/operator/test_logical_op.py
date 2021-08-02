from cube.operator.logic.generics import HolisticOpFactory, GenericLogicalOp


def test_factory():

    factory = HolisticOpFactory()
    assert len(factory) == 0

    class HolisticOp:
        def __init__(self, shape): pass

    factory.register(HolisticOp)
    assert len(factory) == 1

    op = factory.get_op(0, [(1024, 1024)])
    assert isinstance(op, HolisticOp)


def test_generic_logical_op_init():

    generic_op = GenericLogicalOp()
    assert len(generic_op.factory) == 0
    assert generic_op.policy_fn is None


def test_generic_logical_op_register():

    generic_op = GenericLogicalOp()

    class HolisticOp:
        def __init__(self, shape): pass

    generic_op.factory.register(HolisticOp)

    def policy_fn(factory, shapes):
        return factory.get_op(0, shapes)
    
    generic_op.set_policy(policy_fn)
    assert generic_op.policy_fn is not None



if __name__ == '__main__':

    test_factory()
    test_generic_logical_op_init()
    test_generic_logical_op_register()