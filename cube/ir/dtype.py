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


float64 = IRDType.float64
float16 = IRDType.float16
float32 = IRDType.float32
int64   = IRDType.int64
int32   = IRDType.int32
int16   = IRDType.int16
int8    = IRDType.int8
uint8   = IRDType.uint8
boolean = IRDType.boolean
