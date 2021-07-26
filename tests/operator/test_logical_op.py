from cube.operator.logic.generics import HolisticOpFactory, GenericLogicalOp


def test_factory():

    factory = HolisticOpFactory()
    assert len(factory) == 0

    class HolisticOp: pass
    holistic_op = HolisticOp()

    factory.register(holistic_op)
    assert len(factor) == 1

    op = factory.get_op(0)
    assert op is holistic_op


def test_generic_logical_op_init():

    generic_op = GenericLogicalOp()
    assert len(generic_op.factory) == 0
    assert generic_op.policy_fn is None


def test_generic_logical_op_register():

    generic_op = GenericLogicalOp()

    class HolisticOp: pass
    holistic_op = HolisticOp()

    generic_op.factory.register(holistic_op)

    def policy_fn(factory):
        return factory.get_op(0)
    
    generic_op.register_policy(policy_fn)
    assert generic_op.policy_fn is not None

    op = generic_op.get_op()
    assert op is holistic_op


if __name__ == '__main__':

    test_factory
    test_generic_logical_op_init()
    test_generic_logical_op_register()