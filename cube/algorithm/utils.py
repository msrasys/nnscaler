
from cube.ir.cten import IRTensor


def split_axis(tensor: IRTensor, axis: int, chunk_num: int):

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
    for cid in range(chunk_size):
        shape_slicer[axis] = slice(chunk_size * cid, chunk_size * (cid + 1))
        sub_tensors.append(tensor.select(
            indices = tuple(shape_slicer),
            val_op = None,
            shape = chunk_shape
        ))
    return sub_tensors


def split_value(tensor: IRTensor, chunk_num: int):
    raise NotImplementedError