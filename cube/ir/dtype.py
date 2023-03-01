from typing import List
from enum import Enum


class IRDType(Enum):
    float64 = 'float64'
    float16 = 'float16'
    float32 = 'float32'
    int64   = 'int64'
    int32   = 'int32'
    int16   = 'int16'
    int8    = 'int8'
    uint8   = 'uint8'
    boolean = 'bool'
    unknown = 'unknown'


def dtype2byte_size(dtype: IRDType) -> int:
    return {
        IRDType.float64: 8,
        IRDType.float32: 4,
        IRDType.float16: 2,
        IRDType.int64: 8,
        IRDType.int32: 4,
        IRDType.int16: 2,
        IRDType.int8: 1,
        IRDType.uint8: 1,
        IRDType.boolean: 1,
    }.get(dtype, 0)


class DTypeInferRule:
    """
    Infer the output shape according to given input shapes.
    This will follow the dtype promotion rule, which is same with PyTorch.

    Reference:
    https://pytorch.org/docs/stable/tensor_attributes.html#type-promotion-doc

    complex > floating > integral > boolean
    """
    @staticmethod
    def infer(node, dtypes: List[IRDType]) -> IRDType:
        dtypes = [dtype for dtype in dtypes if dtype != IRDType.unknown]
        if IRDType.unknown in dtypes:
            raise RuntimeError(f"Find an unkown dtype")
        if IRDType.float32 in dtypes and IRDType.float16 in dtypes:
            raise RuntimeError(f"Find node has both fp32 and fp16 inputs {node}")
        # TODO(yizhu1): hack
        if node.signature == 'torch.ne':
            return IRDType.boolean
        elif node.signature == 'torch.Tensor.long':
            return IRDType.int64
        # in priority: fp32 > fp16 > bool > int64 > int16 >
        priority = [
            IRDType.float64, IRDType.float32, IRDType.float16,
            IRDType.int64, IRDType.int32, IRDType.int16, IRDType.int8,
            IRDType.boolean
        ]
        for dtype in priority:
            if dtype in dtypes:
                return dtype
        return IRDType.unknown


float64 = IRDType.float64
float16 = IRDType.float16
float32 = IRDType.float32
int64   = IRDType.int64
int32   = IRDType.int32
int16   = IRDType.int16
int8    = IRDType.int8
uint8   = IRDType.uint8
boolean = IRDType.boolean
