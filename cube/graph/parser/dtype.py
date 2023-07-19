import torch
import cube.ir as ir


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
        torch.bfloat16: ir.bfloat16,
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


class IRDType2TorchDType:

    @staticmethod
    def map(ir_dtype: ir.IRDType):
        """
        Map the IRDtype to torch dtype
        """
        assert ir_dtype in IRDType2TorchDType.kDtypeMap, f'unexpected ir_dtype {ir_dtype}'
        return IRDType2TorchDType.kDtypeMap[ir_dtype]
    
    kDtypeMap = {val: key for key, val in DType2IRDType.kDtypeMap.items()}