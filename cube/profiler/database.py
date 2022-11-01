"""
Usage:
    python -m cube.profiler.database --export ./profile.dat.json
"""
from typing import Callable, Tuple, Union, Optional, Dict, NewType, List, Any
import torch
import time
import os
import json

import cube
from cube.ir.cten import IRTensor
from cube.ir.operator import IRFwOperation
from cube.graph.parser.mapping import Sign2Op, IRDType2TorchDType


Shapes = NewType('Shapes', Tuple[Tuple[int]])
DTypes = NewType('DTypes', Tuple[torch.dtype])
ShapesDTypes = NewType('ShapesDTypes', Tuple[Shapes, DTypes])
NameOrFunc = Union[str, Callable]


_train_module_ref: torch.nn.Module = torch.nn.Module().train()
_eval_module_ref: torch.nn.Module = torch.nn.Module().eval()


class CompProfiler:

    @staticmethod
    def profile(func: Callable, shapes: Shapes, dtypes: DTypes,
                warmup_sec: float = 2, prof_times: int = 50,
                **kwargs) -> Tuple[float, float, int, int]:
        """
        Profile a function

        @param func Callable: the callable function, e.g., torch.nn.functional.linear
        @param shapes Tuple[Tuple[int]]: the shapes of each input tensor
        @param dtypes Optional[Tuple[torch.dtype]]: the dtype of each input tensor. Default will use torch.float32
        @param warmup_sec float: warmup seconds
        @param prof_times int: profile times
        @param kwargs Dict: other keyword argument for func call.

        @return fw_span float: the time in milliseconds for forward time
        @return bw_span float: the time in milliseconds for backward time
        @return infer_memory int: the peak memory in bytes after inference of the function
        @return train_memory int: the peak memory in bytes after forward with autograd enabled
        """
        assert len(shapes) == len(dtypes), \
            f"func {func.__name__}: expected each shape has a corresponding dtype, but got {shapes} and {dtypes}"
        # create data
        dtypes = [torch.float32] * len(shapes) if dtypes is None else dtypes
        def gen_torch_tensors(shape, dtype):
            constructor = torch.zeros if dtype == torch.int64 else torch.rand
            requires_grad = False if dtype == torch.int64 else True
            return constructor(tuple(shape), dtype=dtype, device=torch.cuda.current_device(), requires_grad=requires_grad)
        tensors = tuple(
            gen_torch_tensors(shape, dtype) for shape, dtype in zip(shapes, dtypes)
        )
        # repalce kwargs starting with 'self.xxx'
        train_kwargs, eval_kwargs = {}, {}
        for name, value in kwargs.items():
            if isinstance(value, str) and value.startswith('self.'):
                train_val = getattr(_train_module_ref, value[5:])
                eval_val = getattr(_eval_module_ref, value[5:])
            else:
                train_val = eval_val = value
            train_kwargs[name] = train_val
            eval_kwargs[name] = eval_val
        # run one sample
        outputs = func(*tensors, **train_kwargs)
        outputs = (outputs,) if torch.is_tensor(outputs) else outputs
        assert all(torch.is_tensor(otensor) for otensor in outputs), \
            f"{func.__name__}: require all the outputs to be tensors"
        grads = tuple(torch.zeros_like(otensor) for otensor in outputs)

        def run_step(func, tensors, kwargs, backward: bool):
            outputs = func(*tensors, **kwargs)
            if backward:
                torch.autograd.backward(outputs, grads)
            return outputs

        # profile inference peak memory
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        mtic = torch.cuda.max_memory_allocated()  # in bytes
        with torch.no_grad():
            run_step(func, tensors, eval_kwargs, backward=False)
        mtoc = torch.cuda.max_memory_allocated()  # in bytes
        infer_memory = mtoc - mtic

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        mtic = torch.cuda.max_memory_allocated()  # in bytes
        outs = run_step(func, tensors, train_kwargs, backward=False)
        mtoc = torch.cuda.max_memory_allocated()  # in bytes
        train_memory = mtoc - mtic

        # warmup
        tic = time.time()
        while time.time() - tic < warmup_sec:
            run_step(func, tensors, train_kwargs, backward=True)

        # profile forward only
        torch.cuda.synchronize()
        tic = time.perf_counter()
        for _ in range(prof_times):
            with torch.no_grad():
                run_step(func, tensors, eval_kwargs, backward=False)
        torch.cuda.synchronize()
        toc = time.perf_counter()
        fw_span = (toc - tic) / prof_times * 1000 # in milliseconds

        # profile forward + backward
        torch.cuda.synchronize()
        tic = time.perf_counter()
        for _ in range(prof_times):
            run_step(func, tensors, train_kwargs, backward=True)
        torch.cuda.synchronize()
        toc = time.perf_counter()
        fwbw_span = (toc - tic) / prof_times * 1000 # in milliseconds
        bw_span = fwbw_span - fw_span

        return fw_span, bw_span, infer_memory, train_memory


class ProfileDataBase:

    def __init__(self, filename: Optional[str] = None) -> None:
        """!
        Create a database for profiling result
        """

        self._data: Dict[str, Dict[str, Tuple[float, float, int]]] = dict()
        if filename is not None:
            self.load(filename)

    @staticmethod
    def get_func(node: IRFwOperation) -> Tuple[Callable, Shapes, DTypes, Dict]:
        """
        Get function call and its arguments from a cude IRGraph node
        """
        assert isinstance(node, IRFwOperation), f"Only support profiling forward operation but got {type(node)}"
        if node.signature in Sign2Op.kOpCodeDef:
            code_impl: str = Sign2Op.kOpCodeDef[node.signature]
            local = {}
            exec(code_impl, globals(), local)
            fn = list(local.values())[0]
        else:
            fn = eval(node.signature)
        shapes, dtypes = [], []
        for t in node.inputs():
            assert isinstance(t, IRTensor), f"Only support node inputs with tensor shape"
            shapes.append(t.shape)
            dtypes.append(IRDType2TorchDType.map(t.dtype))
        return fn, shapes, dtypes, node.kwargs

    def profile(self, node: IRFwOperation, device: Optional[int] = None):
        """
        Profile a forward node in IRGraph on a specific device (default current device)
        
        @param node IRFwOperation: node of IRGraph
        @param device int: the device that the node will execute on

        @return fw_span float: the forward span time in milliseconds
        @return bw_span float: the backward span time in milliseconds
        @return infer_memory int: the peak memory in bytes after inference of the function
        @return train_memory int: the peak memory in bytes after forward with autograd enabled
        """
        fn, shapes, dtypes, kwargs = ProfileDataBase.get_func(node)

        if self.exist(node):
            return self.query(node)

        if isinstance(device, int):
            orig_device = torch.cuda.current_device()
            torch.cuda.set_device(device)
        
        # run profiling
        fw_span, bw_span, infer_memory, train_memory = \
            CompProfiler.profile(fn, shapes, dtypes, **kwargs)
        # log to database
        key = self._serialize(node)
        self.insert(node.signature, key, fw_span, bw_span, infer_memory, train_memory)
        print(
            f"profiled {node.signature} | shapes: {shapes} | dtypes: {dtypes} "
            f"=> fw: {round(fw_span, 2)} ms | bw: {round(bw_span, 2)} ms | "
            f"infer mem: {infer_memory} | train mem: {train_memory}")

        if isinstance(device, int):
            torch.cuda.set_device(orig_device)
        return fw_span, bw_span, infer_memory, train_memory

    def insert(self, name: str, key: str, fw_span: float, bw_span: float,
               infer_memory: int, train_memory: int):
        """
        log the span of a function name with key

        @param name str: the function signature
        @param key str: the encoded shapes and dtypes of node inputs
        @param fw_span float: the forward span time in milliseconds
        @param bw_span float: the backward span time in milliseconds
        @param infer_memory int: the peak memory in bytes after inference of the function
        @param train_memory int: the peak memory in bytes after forward with autograd enabled
        """
        assert isinstance(name, str) and isinstance(key, str)
        if name not in self._data:
            self._data[name] = dict()
        self._data[name][key] = (fw_span, bw_span, infer_memory, train_memory)

    def exist(self, node: IRFwOperation) -> bool:
        """
        Check if the node has the performance recorded in the database

        @param node IRFwOperation: forward operation

        @return exist bool: True if the performance is recorded, else False
        """
        key = self._serialize(node)
        if node.signature not in self._data:
            return False
        if key not in self._data[node.signature]:
            return False
        return True

    def query(self, node: IRFwOperation) -> Tuple[float, float, int, int]:
        """!
        Get the performance number of a node in IRGraph

        @param node IRFwOperation: node in IRGraph

        @return fw_span float: the forward span time in milliseconds
        @return bw_span float: the backward span time in milliseconds
        @return infer_memory int: the peak memory in bytes after inference of the function
        @return train_memory int: the peak memory in bytes after forward with autograd enabled
        """
        key = self._serialize(node)
        if node.signature not in self._data:
            return None
        if key not in self._data[node.signature]:
            return None
        return self._data[node.signature][key]

    def query_func(self, signature, shapes, dtypes) -> Tuple[float, float, int, int]:
        """
        Get performance number of given name (signature), shapes and dtypes
        
        @param signature str: function signature
        @param shapes Tuple[Tuple[int]]: the shape of each input tensor
        @param dtypes Tuple[torch.dtype]: the dtype of each tensor

        @return fw_span float: the forward span time in milliseconds
        @return bw_span float: the backward span time in milliseconds
        @return infer_memory int: the peak memory in bytes after inference of the function
        @return train_memory int: the peak memory in bytes after forward with autograd enabled
        """
        key = self._serialize(shapes, dtypes)
        if signature not in self._data:
            return None
        if key not in self._data[signature]:
            return None
        return self._data[signature][key]

    def query_args(self, signature: str) -> Tuple[List[Shapes], List[DTypes]]:
        """
        Get the recorded shapes and dtypes of 
        """
        item_shapes, item_dtypes = [], []
        if signature not in self._data:
            return item_shapes, item_dtypes
        for shapes_dtypes_str in self._data[torch.signature].keys():
            shapes, dtypes = self._deserialize(shapes_dtypes_str)
            item_shapes.append(shapes)
            item_dtypes.append(dtypes)
        return item_shapes, item_dtypes

    def _serialize(self, node: IRFwOperation) -> str:
        """
        Serialize the shapes, dtypes and kwargs into a string

        e.g.,
            shapes: ((1024,), (1024,1024))
            dtypes: (torch.float32, torch.float32)
        => (1024,)-(1024,1024) : torch.float32-torch.float32

        @param shapes Tuple[Tuple[int]]: the shape of each tensor
        @param dtypes Tuple[torch.dtype]: the dtype of each tensor

        @return key str: the serialized string
        """
        shapes, dtypes = [], []
        for t in node.inputs():
            assert isinstance(t, IRTensor), f"Only support node inputs with tensor shape"
            shapes.append(t.shape)
            dtypes.append(IRDType2TorchDType.map(t.dtype))
        shapes = '-'.join(str(tuple(shape)) for shape in shapes)
        dtypes = '-'.join(str(dtype) for dtype in dtypes)
        return shapes + ' : ' + dtypes

    def _deserialize(self, key: str) -> ShapesDTypes:
        """
        De-serialize the key string to shapes and dtypes

        e.g., (1024,)-(1024,1024)=torch.float32-torch.float32
        =>  shapes: ((1024,), (1024,1024))
            dtypes: (torch.float32, torch.float32)

        @param key str: the serialized string
        @return shapes_and_dtypes ShapesDTypes: shapes and dtypes
        """
        shapes, dtypes = key.split(' : ')
        shapes = tuple(eval(shape) for shape in shapes.split('-'))
        dtypes = tuple(eval(dtype) for dtype in dtypes.split('-'))
        return shapes, dtypes

    def dump(self, file: str, override=False):
        """!
        dump the profiled data into json format

        @param file str: the file name
        @param override bool: True if the existed can be overrided else False
        """
        if os.path.exists(file):
            assert override, f"File {file} exists. Set override = True to force dump."
        with open(file, 'w') as f:
            json.dump(self._data, f)

    def load(self, file: str):
        """!
        load the profiled data into data base. The original existed one will be
        overrided by the loaded data.

        @param file str: the file name
        """
        with open(file, 'r') as f:
            self._data = json.load(f)

    def __repr__(self) -> str:
        data = []
        for signature in self._data:
            for key in self._data[signature]:
                shapes, dtypes = self._deserialize(key)
                fw_span, bw_span, infer_mem, train_mem = self._data[signature][key]
                data.append(f'{signature}: shapes={shapes}, dtypes={dtypes}, fw span: {fw_span} ms, bw span: {bw_span} ms, infer mem {infer_mem} bytes, train mem {train_mem} bytes')
        data = '\n'.join(data)
        return data
