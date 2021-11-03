from cube.algorithm.factory import DistAlgorithmFactory
from cube.graph.operator.function import Linear
from cube.algorithm.generics import GenericDistAlgo


def test_factory_init():
    factory = DistAlgorithmFactory()
    assert len(factory.algorithms(Linear)) == 3


def test_factory_tag():

    factory = DistAlgorithmFactory()
    dp = factory.algorithms(Linear, tag='data')
    assert issubclass(dp, GenericDistAlgo)
