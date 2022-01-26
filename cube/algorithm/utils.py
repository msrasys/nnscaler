from typing import List, Union
from cube.graph.tensor import IRSubTensor


def split_axis(tensor: IRSubTensor, axis: int, chunk_num: int):
    """
    Split tensor along an axis. The axis can be positive or negative.
    """
    if axis < 0:
        axis = len(tensor.shape) + axis
    if axis >= len(tensor.shape):
        raise RuntimeError(f"Axis should within dims ({axis} >= {len(tensor.shape)})")
    
    chunk_size = int(tensor.shape[axis] // chunk_num)

    shape_slicer = list()
    chunk_shape = list()
    for dim, nele in enumerate(tensor.shape):
        if dim != axis:
            shape_slicer.append(slice(0, nele, 1))
            chunk_shape.append(nele)
        else:
            shape_slicer.append(None)
            chunk_shape.append(chunk_size)

    sub_tensors = list()
    for cid in range(chunk_num):
        shape_slicer[axis] = slice(chunk_size * cid, chunk_size * (cid + 1), 1)
        sub_tensors.append(tensor.select(
            indmap = tuple(shape_slicer),
            valmap = None,
            shape = chunk_shape
        ))
    return sub_tensors


def split_axis_custom(tensor: IRSubTensor, axis: int, chunks: List[slice]):
    """
    Split tensor along an axis with cutomized selection
    """
    if axis < 0:
        axis = len(tensor.shape) + axis
    if axis >= len(tensor.shape):
        raise RuntimeError(f"Axis should within dims ({axis} >= {len(tensor.shape)})")
    chunk_num = len(chunks)

    slicers, shape = list(), list()
    for nele in tensor.shape:
        slicers.append(slice(0, nele, 1))
        shape.append(nele)
    sub_tensors = list()
    for cid in range(chunk_num):
        slicers[axis] = chunks[cid]
        shape[axis] = chunks[cid].stop - chunks[cid].start
        sub_tensors.append(tensor.select(
            indmap = tuple(slicers),
            valmap = None,
            shape = shape
        ))
    return sub_tensors


def split_value(tensor: IRSubTensor, chunk_num: int):

    # full shape
    shape_slicer = list()
    for nele in tensor.shape:
        shape_slicer.append(slice(0, nele, 1))

    sub_tensors = list()
    for idx in range(chunk_num):
        sub_tensor = tensor.select(
            indmap = tuple(shape_slicer),
            valmap = (idx, chunk_num),
            shape = tensor.shape
        )
        sub_tensors.append(sub_tensor)

    return sub_tensors
