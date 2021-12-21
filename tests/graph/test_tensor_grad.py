from cube.graph.graph import IRGraph
from cube.graph.tensor import IRFullTensor, ValueMap
from cube.graph.operator.function import Linear, ElementWise
import cube.graph.gpass as gpass
from cube.ir.cten import IRTensor


def construct_model():

    input1 = IRFullTensor(shape=[64,1024], name='data1')
    input2 = IRFullTensor(shape=[64,1024], name='data2')
    weight1 = IRFullTensor(shape=[1024, 1024], name='weight')
    bias1 = IRFullTensor(shape=[1024, 1024], name='bias')
    weight2 = IRFullTensor(shape=[1024, 1024], name='weight')
    weight3 = IRFullTensor(shape=[1024, 1024], name='weight')
    bias3 = IRFullTensor(shape=[1024, 1024], name='bias')
    weight3 = IRFullTensor(shape=[1024, 1024], name='weight')
    bias3 = IRFullTensor(shape=[1024, 1024], name='bias')

    # linear1
    linear1 = Linear(
        name='linear1',
        signature='torch.nn.functional.linear',
        inputs= [input1, weight1, bias1],
    )
    linear1.infer_shape()

    # linear2
    linear2 = Linear(
        name='linear2',
        signature='torch.nn.functional.linear',
        inputs= [linear1.outputs(0), weight2, None],
    )
    linear2.infer_shape()

    # linear3
    linear3 = Linear(
        name='linear3',
        signature='torch.nn.functional.linear',
        inputs= [linear2.outputs(0), weight3, bias3],
    )
    linear3.infer_shape()

    # linear4
    linear4 = Linear(
        name='linear4',
        signature='torch.nn.functional.linear',
        inputs= [input2, weight1, bias1],
    )
    linear4.infer_shape()

    # element-wise
    add5 = ElementWise(
        name='add',
        signature='torch.add',
        inputs=[linear2.outputs(0), linear3.outputs(0)]
    )
    add5.infer_shape()

    # element-wise
    add6 = ElementWise(
        name='add',
        signature='torch.add',
        inputs=[add5.outputs(0), linear4.outputs(0)]
    )
    add6.infer_shape()

    # return [input], [ops], [output]
    return [input1, input2], [linear1, linear2, linear3, linear4, add5, add6], [add6.outputs(0)]


def test_tensor_grad():

    inputs, ops, outputs = construct_model()
    linear1, linear2, linear3, linear4, add5, add6 = ops
    graph = IRGraph(ops, inputs, outputs, 'MLP')
    print(graph)

    all_parent_tids = list()
    all_parent_tensors = list()
    for op in ops:
        for input in op.inputs():
            if isinstance(input, IRTensor):
                if input.parent._id not in all_parent_tids:
                    all_parent_tensors.append(input.parent)
    
    for pten in all_parent_tensors:
        assert pten.grad is None
        print(pten.name, pten)
        cell_ids = [cell._id for cell in pten.consumers]
        print('consumers id:', cell_ids)
        print('')

    print('test grad:')

    input = linear1.inputs(0)
    assert input.grad is None
    gin = input.get_grad(linear1)
    assert gin.valmap == ValueMap(0, 1)
    print(gin.name, gin)

    weight = linear1.inputs(1)
    gw = weight.get_grad(linear1)
    assert gw.valmap == ValueMap(0, 2)
    print(gw.name, gw)

    weight = linear4.inputs(1)
    gw = weight.get_grad(linear4)
    assert gw.valmap == ValueMap(1, 2)
    print(gw.name, gw)

    out2 = linear2.outputs(0)
    gout2 = out2.get_grad(linear2)
    print(gout2.name, gout2)
    assert gout2.valmap == ValueMap(0, 1)
    gout2 = out2.get_grad(linear3)
    print(gout2.name, gout2)
    assert gout2.valmap == ValueMap(0, 2)
    gout2 = out2.get_grad(add5)
    print(gout2.name, gout2)
    assert gout2.valmap == ValueMap(1, 2)

    out3 = linear3.outputs(0)
    gout3 = out3.get_grad(linear3)
    print(gout3.name, gout3)
    assert gout3.valmap == ValueMap(0, 1)
    gout3 = out3.get_grad(add5)
    print(gout3.name, gout3)
    assert gout3.valmap == ValueMap(0, 1)

    for node in graph.nodes():
        assert node.mirror is None

    print('test forward graph:')
    inputs = [inputs[0].tosub(), inputs[1].tosub()]
    graph = gpass.forward(graph, *inputs)
    print(graph)
    for node in graph.nodes()[::-1]:
        print(node.mirror)
    
    gw1 = linear1.mirror.outputs(1)
    assert gw1.is_grad()
    print(gw1)
    gw4 = linear4.mirror.outputs(1)
    assert gw4.is_grad()
    print(gw4)

    assert gw1.parent == gw4.parent
    assert gw1.shape == gw4.shape
    assert gw1.indmap == gw4.indmap
    assert gw1.valmap != gw4.valmap

    # assert False
