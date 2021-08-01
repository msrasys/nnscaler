from cube.tensor.logic.tensor import LogicalTensor
import cube.tensor.logic.outline as outline
from cube.tensor.segment import Segment

import torch
import z3


def iter_each_config(solver, attrs):
    if len(attrs) == 0:
        solver.check()
        yield solver.model()
    else:
        while solver.check() == z3.sat:
            config = solver.model()
            solver.add(z3.Or([z3.Not(attr == config[attr]) for attr in attrs]))
            yield config


def test_full():
    shape = (10,10,10)
    tensor = torch.randn(shape)
    solver = z3.Solver()

    full_dsp = outline.Full(solver, shape)
    assert len(full_dsp.get_attributes()) == 0
    
    configs = list()
    for config in iter_each_config(solver, full_dsp.get_attributes()):
        configs.append(config)

    assert len(configs) == 1
    config = configs[0]
    
    tensor = LogicalTensor(shape=shape)
    segments = full_dsp.interpret(tensor, config)
    assert len(segments) == 1
    assert tuple(segments[0].shape) == tuple(tensor.shape)
    assert torch.allclose(tensor.data, tensor.data[segments[0].indices.get()]) is True


def test_split_axis():

    axis = 1
    shape = [1024, 16]
    solver = z3.Solver()

    tensor = torch.randn(shape)
    split_dsp = outline.SplitAxis(
        solver, shape, axis, chunk_num=None, overlap=0
    )

    # test config space
    configs = list()
    for config in iter_each_config(solver, split_dsp.get_attributes()):
        configs.append(config)
    assert len(configs) == 5

    # test segments
    tensor = LogicalTensor(shape=shape)
    segments = split_dsp.interpret(tensor, configs[0])
    shape_axis = [segment.shape[axis] for segment in segments]
    assert sum(shape_axis) == shape[axis]


def test_split_axis_with_constraints():

    axis = 1
    shape = [1024, 16]
    solver = z3.Solver()

    split_dsp = outline.SplitAxis(
        solver, shape, axis, chunk_num=None, overlap=0
    )

    # this can be set due to device number constraints
    split_dsp.solver.add(split_dsp.chunk_num <= 8)

    configs = list()
    for config in iter_each_config(solver, split_dsp.get_attributes()):
        configs.append(config)
        # print(config)
    assert len(configs) == 4


def test_split_value():

    shape = [1024, 32]
    split_op = lambda tensor, rank, world_size : tensor / world_size
    solver = z3.Solver()

    split_dsp = outline.SplitValue(solver, shape, None, split_op)
    split_dsp.solver.add(split_dsp.chunk_num <= 4)
    configs = list()
    for config in iter_each_config(solver, split_dsp.get_attributes()):
        configs.append(config)
    assert len(configs) == 4

    tensor = LogicalTensor(shape=shape)
    segments = split_dsp.interpret(tensor, configs[0])
    for segment in segments:
        assert torch.allclose(tensor.data, tensor.data[segment.indices.get()]) is True


def test_align():

    shape = [1024, 16]
    solver = z3.Solver()

    dsp1 = outline.SplitAxis(
        solver, shape, axis=0, chunk_num=None, overlap=0, 
    )
    
    dsp2 = outline.SplitAxis(
        solver, shape, axis=1, chunk_num=dsp1.chunk_num, overlap=0, 
    )

    configs = list()
    attrs = dsp1.get_attributes() + dsp2.get_attributes()
    for config in iter_each_config(solver, attrs):
        configs.append(config)
    assert len(configs) == 5


if __name__ == '__main__':

    # test_base()
    test_full()
    test_split_axis()
    test_split_axis_with_constraints()
    test_split_value()
    test_align()