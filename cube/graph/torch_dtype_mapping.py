from cube import ir
import torch

class DType2IRDType:

    @staticmethod
    def map(dtype: torch.dtype):
        """
        Map the torch dtype to IRDType
        """
        return DType2IRDType.kDtypeMap[dtype]

    kDtypeMap = {
        torch.double:  ir.float64,
        torch.float64: ir.float64,
        torch.float32: ir.float32,
        torch.float  : ir.float32,
        torch.float16: ir.float16,
        torch.half   : ir.float16,
        torch.uint8  : ir.uint8,
        torch.int8   : ir.int8,
        torch.int16  : ir.int16,
        torch.short  : ir.int16,
        torch.int32  : ir.int32,
        torch.int    : ir.int32,
        torch.int64  : ir.int64,
        torch.long   : ir.int64,
        torch.bool   : ir.boolean
    }


# see https://github.com/pytorch/pytorch/blob/master/c10/core/ScalarType.h
#
# ScalarType enum is totally a PyTorch-internal object. Neither itself nor its underlying ints
# are accessible from its Python frontend.
class TorchScalarTypeEnumMap:

    @staticmethod
    def map(underlying: int) -> torch.dtype:
        
        assert isinstance(underlying, int), """
        This function is to convert an underlying 'int' for a Torch-internal 'at::ScalarType' enum
        to its corresponding Python-frontend 'torch.dtype' enum.
        """

        dtype = TorchScalarTypeEnumMap._fields[underlying]

        assert dtype is not None, f"""
        Referenced to an unsupported ScalarType with underlying int being {underlying}
        """
        
        return dtype

    # Less used dtypes are masked out because PyTorch keeps **exposing and hiding** them recently
    # from a view of Python frontend.
    _fields = [
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.half,
        torch.float32,
        torch.float64,
        None, #torch.complex32,    # complexHalf
        None, #torch.complex64,    # complexFloat
        None, #torch.complex128,   # complexDouble
        torch.bool,
        None, #torch.qint8,
        None, #torch.quint8,
        None, #torch.qint32,
        None, #torch.bfloat16,
        None, #torch.quint4x2,
        None, #torch.quint2x4,
    ]

    assert len(_fields) == 18, "Do not remove any item, mask it out with None"