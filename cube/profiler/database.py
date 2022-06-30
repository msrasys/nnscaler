"""
Usage:
    python -m cube.profiler.database --export ./profile.dat.json
"""

from typing import Callable, Tuple, Union, Optional, Dict, NewType
import torch
import time
import os
import json


Shapes = NewType('Shapes', Tuple[Tuple[int]])
DTypes = NewType('DTypes', Tuple[torch.dtype])
ShapesDTypes = NewType('ShapesDTypes', Tuple[Shapes, DTypes])
NameOrFunc = NewType('NameOrFunc', Union[str, Callable])


class CompProfiler:

    @staticmethod
    def profile(func: Callable, shapes: Shapes, dtypes: DTypes,
                warmup_sec: float = 2, prof_times: int = 50, backward = True,
                **kwargs):
        """
        Profile a function

        @param func Callable: the callable function, e.g., torch.nn.functional.linear
        @param shapes Tuple[Tuple[int]]: the shapes of each input tensor
        @param dtypes Optional[Tuple[torch.dtype]]: the dtype of each input tensor. Default will use torch.float32
        @param warmup_sec float: warmup seconds
        @param prof_times int: profile times
        @param backward bool: whether profile backward times. Default true.
        @param kwargs Dict: other keyword argument for func call.

        @return span float: the time in milliseconds for forward (+backward) time
        """
        assert len(shapes) == len(dtypes), \
            f"func {func.__name__}: expected each shape has a corresponding dtype, but got {shapes} and {dtypes}"
        # create data
        dtypes = [torch.float32] * len(shapes) if dtypes is None else dtypes
        tensors = tuple(
            torch.rand(tuple(shape), dtype=dtype, device=torch.cuda.current_device(), requires_grad=True) \
                for shape, dtype in zip(shapes, dtypes)
        )
        outputs = func(*tensors, **kwargs)
        outputs = (outputs,) if torch.is_tensor(outputs) else outputs
        assert all(torch.is_tensor(otensor) for otensor in outputs), \
            f"{func.__name__}: require all the outputs to be tensors"
        grads = tuple(torch.zeros_like(otensor) for otensor in outputs)

        # warmup
        tic = time.time()
        while time.time() - tic < warmup_sec:
            # forward
            outputs = func(*tensors, **kwargs)
            # backward
            if backward:
                torch.autograd.backward(outputs, grads)
        
        # profile forward
        torch.cuda.synchronize()
        tic = time.perf_counter()
        for _ in range(prof_times):
            # forward
            outputs = func(*tensors, **kwargs)
            # backward
            if backward:
                torch.autograd.backward(outputs, grads)
        torch.cuda.synchronize()
        toc = time.perf_counter()
        span = (toc - tic) / prof_times * 1000 # in milliseconds
        return span


class ProfileDataBase:

    def __init__(self, filename: Optional[str] = None) -> None:
        """!
        Create a database for profiling result
        """

        self._data: Dict[str, Dict[str, float]] = dict()
        if filename is not None:
            self.load(filename)

    def profile(self, func: Callable, shapes: Shapes, dtypes: DTypes,
                backward=True, **kwargs):
        """!
        Profile the function and log into the database

        @param func Callable: the callable function, e.g., torch.nn.functional.linear
        @param shapes Tuple[Tuple[int]]: the shapes of each input tensor
        @param dtypes Optional[Tuple[torch.dtype]]: the dtype of each input tensor. Default will use torch.float32
        @param backward bool: whether profile backward times. Default true.
        @param kwargs Dict: other keyword argument for func call.
        """
        try:
            assert callable(func), "func should be callable"
            span = CompProfiler.profile(func, shapes, dtypes, backward=backward, **kwargs)
        except Exception as e:
            print(f'fail to profile {func.__name__}: reason: {str(e)}')
        name = func.__name__
        key = self.serialize(shapes, dtypes)
        self.log(name, key, span)
        print(f'profiled {func.__name__} | shapes: {shapes} | dtypes: {dtypes} => span: {round(span, 2)} ms')
    
    def log(self, name: str, key: str, span: float):
        """
        log the span of a function name with key 
        """
        assert isinstance(name, str) and isinstance(span, float) and isinstance(key, str)
        if name not in self._data:
            self._data[name] = dict()
        self._data[name][key] = span

    def query(self, func: NameOrFunc, shapes: Shapes, dtypes: DTypes) -> float:
        """!
        Get the performance number of the function name and its key

        @param name str: function name
        @param shapes Tuple[Tuple[int]]: the shape of each input tensor
        @param dtypes Tuple[torch.dtype]: the dtype of each tensor

        @return span float: the performance number
        """
        name = func if isinstance(func, str) else func.__name__
        key = self.serialize(shapes, dtypes)
        return self._data[name][key]

    def exist_item(self, func: NameOrFunc, shapes: Shapes, dtypes: DTypes) -> bool:
        """!
        Check if the required data exists

        @param name Union[str, Callable]: function name
        @param shapes Tuple[Tuple[int]]: the shape of each input tensor
        @param dtypes Tuple[torch.dtype]: the dtype of each tensor

        @return exist bool: True if the item exists else False
        """
        name = func if isinstance(func, str) else func.__name__
        if name not in self._data:
            return False
        key = self.serialize(self, shapes, dtypes)
        if key not in self._data[key]:
            return False
        return True

    def exist_func(self, func: NameOrFunc) -> bool:
        """!
        Check if the required function exists

        @param name Union[str, Callable]: function name

        @return exist bool: True if the function exists else False
        """
        name = func if isinstance(func, str) else func.__name__
        return name in self._data

    def shapes_and_dtypes(self, func: NameOrFunc) -> Tuple[ShapesDTypes]:
        """
        Get recorded shapes and dtypes of the func.

        @param func UnShapesDTypesion[str, Callable]: function name

        @return shapes_and_dtypes Tuple[ShapesDTyptes]
        """
        name = func if isinstance(func, str) else func.__name__
        rets = []
        for shapes_dtypes_str in self._data[name].keys():
            (shapes, dtypes) = self.deserialize(shapes_dtypes_str)
            rets.append((shapes, dtypes))
        return tuple(rets)

    def serialize(self, shapes: Shapes, dtypes: DTypes) -> str:
        """
        Serialize the shapes, dtypes and kwargs into a string

        e.g.,
            shapes: ((1024,), (1024,1024))
            dtypes: (torch.float32, torch.float32)
        => (1024,)-(1024,1024)=torch.float32-torch.float32

        @param shapes Tuple[Tuple[int]]: the shape of each tensor
        @param dtypes Tuple[torch.dtype]: the dtype of each tensor

        @return key str: the serialized string
        """
        shapes = '-'.join(str(tuple(shape)) for shape in shapes)
        if dtypes is not None:
            dtypes = '-'.join(str(dtype) for dtype in dtypes)
        else:
            dtypes = '-'.join([str(torch.float32)] * len(shapes))
        return shapes + '=' + dtypes

    def deserialize(self, key: str) -> ShapesDTypes:
        """
        De-serialize the key string to shapes and dtypes

        e.g., (1024,)-(1024,1024)=torch.float32-torch.float32
        =>  shapes: ((1024,), (1024,1024))
            dtypes: (torch.float32, torch.float32)

        @param key str: the serialized string
        @return shapes_and_dtypes ShapesDTypes: shapes and dtypes
        """
        shapes, dtypes = key.split('=')
        print(shapes)
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


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='database')
    parser.add_argument('--export', type=str, default='./profile.dat.json',
                        help='saved profiling database')
    args = parser.parse_args()

    db = ProfileDataBase()
    
    # profile
    dtype = torch.float32
    # func: [
    #   [shapes, dtypes, kwargs],
    # ]
    funcs = {
        torch.nn.functional.linear: [
            [([1024, 1, 2304], [2304, 2304]), (dtype, dtype), {}],
            [([1024, 4, 2304], [2304, 2304]), (dtype, dtype), {}],
            [([1024, 8, 2304], [2304, 2304]), (dtype, dtype), {}]
        ],
    
        torch.nn.functional.gelu: [
            [((1024, 8, 2304),), (dtype,), {}]
        ],
    
        torch.nn.functional.softmax: [
            [((1024, 8, 2304),), (dtype,), dict(dim=-1)]
        ]
    }
    
    for func, keys in funcs.items():
        for shapes, dtypes, kwargs in keys:
            db.profile(func, shapes, dtypes, backward=True, **kwargs)
    
    db.dump(args.export, override=True)

    # db = ProfileDataBase(args.export)
    # for shapes, dtypes in db.shapes_and_dtypes(torch.nn.functional.linear):
    #     span = db.query(torch.nn.functional.linear, shapes, dtypes)
    #     print(f'logged shapes: {shapes}, dtypes: {dtypes} => span: {span} ms')
