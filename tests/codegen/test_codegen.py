from cube.graph.tensor import IRFullTensor
from cube.graph.comm import IRCommunication
from cube.graph.operator import IROperation
from cube.graph.graph import IRGraph

from cube.codegen.codegen import SScheduleCodeGen, TScheduleCodeGen


def construct_graph():

    input = IRFullTensor(shape=[64,1024], name='data')
    weight1 = IRFullTensor(shape=[1024, 1024], name='weight')
    bias1 = IRFullTensor(shape=[1024, 1024], name='bias')
    weight2 = IRFullTensor(shape=[1024, 1024], name='weight')
    weight3 = IRFullTensor(shape=[1024, 1024], name='weight')
    bias3 = IRFullTensor(shape=[1024, 1024], name='bias')

    # linear1
    linear1 = IROperation(
        name='linear1',
        signature='torch.nn.functional.linear',
        input_length=3,
        output_length=1
    )
    linear1.set_input(0, input)
    linear1.set_input(1, weight1)
    linear1.set_input(2, bias1)

    # linear2
    linear2 = IROperation(
        name='linear2',
        signature='torch.nn.functional.linear',
        input_length=3,
        output_length=1
    )
    linear2.set_input(0, linear1.outputs(0))
    linear2.set_input(1, weight2)

    # linear3
    linear3 = IROperation(
        name='linear2',
        signature='torch.nn.functional.linear',
        input_length=3,
        output_length=1
    )
    linear3.set_input(0, linear2.outputs(0))
    linear3.set_input(1, weight3)
    linear3.set_input(2, bias3)

    graph = IRGraph(
        nodes=[linear1, linear2, linear3],
        input_tensors=[input],
        output_tensors=linear3.outputs(), 
        module_name="Test"
    )
    return graph


def test_model_gen():

    