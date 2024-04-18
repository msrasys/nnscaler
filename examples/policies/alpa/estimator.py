from typing import Callable, Tuple, Union, Optional, Dict, NewType, List
import time
import os
import json

# ===== neccesaary for profiling =====
import nnscaler
import torch
# ====================================

from nnscaler.ir.cten import IRTensor, IRObject, IRCell
from nnscaler.ir.operator import IRFwOperation
from nnscaler.graph.parser.register import CustomizedOps
from nnscaler.graph.segment import IRSegment
from nnscaler.graph.function.dimops import IRDimops
from nnscaler.graph.function import IRGraphAnchor


Shapes = NewType('Shapes', Tuple[Tuple[int]])
DTypes = NewType('DTypes', Tuple[torch.dtype])
ShapesDTypes = NewType('ShapesDTypes', Tuple[Shapes, DTypes])
NameOrFunc = Union[str, Callable]


_train_module_ref: torch.nn.Module = torch.nn.Module().train()
_eval_module_ref: torch.nn.Module = torch.nn.Module().eval()


class CompProfiler:

    @staticmethod
    def profile(node: IRCell, train: bool = True,
                warmup_sec: float = 2, prof_times: int = 50) -> Tuple[float, float, int, Tuple[int]]:
        """
        Profile a function

        @param func Callable: the callable function, e.g., torch.nn.functional.linear
        @param warmup_sec float: warmup seconds
        @param prof_times int: profile times

        @return latency float: average latency in ms
        @return memory int: average memory in bytes
        """
        torch.cuda.empty_cache()
        # print(f'current GPU memory: {torch.cuda.memory_allocated() / 1024 / 1024 / 1024} GB')

        func: Callable = CompProfiler.get_func(node)
        args, kwargs = CompProfiler.get_inputs(node, train=train)
    
        # prepare gradients
        with torch.no_grad():
            outputs = func(*args, **kwargs)
        outputs = (outputs,) if torch.is_tensor(outputs) else outputs
        assert all(torch.is_tensor(otensor) for otensor in outputs), \
            f"{func.__name__}: require all the outputs to be tensors"
        grads = tuple(torch.zeros_like(otensor) for otensor in outputs)
        del outputs

        def run_step(func, tensors, kwargs, backward: bool):
            if not backward:
                with torch.no_grad():
                    outputs = func(*tensors, **kwargs)
            else:
                outputs = func(*tensors, **kwargs)
                torch.autograd.backward(outputs, grads)

        # memory
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        mtic = torch.cuda.max_memory_allocated()  # in bytes
        memory = 0
        if train:
            used_tensor = set()
            def pack_hook(x):
                nonlocal memory, used_tensor
                if x.storage().data_ptr() not in used_tensor:
                    used_tensor.add(x.storage().data_ptr())
                    byte_size = x.element_size()
                    for dim in list(x.size()):
                        byte_size = byte_size * dim
                    memory += byte_size
                return x
            def unpack_hook(x): return x

            with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
                run_step(func, args, kwargs, backward=True)
            torch.cuda.synchronize()
            del used_tensor
        else:
            run_step(func, args, kwargs, backward=False)
            torch.cuda.synchronize()
            mtoc = torch.cuda.max_memory_allocated()
            memory = mtoc - mtic

        # warmup
        torch.cuda.synchronize()
        tic = time.time()
        while time.time() - tic < warmup_sec:
            run_step(func, args, kwargs, backward=train)
            torch.cuda.synchronize()

        torch.cuda.synchronize()
        tic = time.perf_counter()
        for _ in range(prof_times):
            run_step(func, args, kwargs, backward=train)
        torch.cuda.synchronize()
        toc = time.perf_counter()
        latency = (toc - tic) / prof_times * 1000  # in milliseconds
        
        return latency, memory

    @staticmethod
    def get_inputs(node: IRFwOperation, train: bool) -> Tuple[List, Dict]:
        # create data
        def dummy_torch_tensor(tensor: IRTensor):
            """Generate dummy input tenosrs"""
            dtype = tensor.dtype
            constructor = torch.zeros if dtype in (torch.int64, torch.int32, torch.bool) else torch.rand
            return constructor(tuple(tensor.shape), dtype=dtype, device=torch.cuda.current_device(), requires_grad=tensor.requires_grad)

        args = [dummy_torch_tensor(t) if isinstance(t, IRTensor) else t for t in node.inputs()]
        # replace kwargs starting with 'self.xxx'
        kwargs = {}
        for name, value in node.kwargs.items():
            if isinstance(value, str) and value.startswith('self.'):
                value = getattr(_train_module_ref, value[5:]) if train else getattr(_eval_module_ref, value[5:])
            kwargs[name] = value
        
        return args, kwargs

    @staticmethod
    def get_func(node: IRFwOperation) -> Callable:
        """
        Get function call
        """
        assert isinstance(node, IRFwOperation), f"Only support profiling forward operation but got {type(node)}"

        def get_dep_names(sign: str):
            ret = []
            code_impl = CustomizedOps.kOpCodeDef[sign]
            for code_line in code_impl.split('\n'):
                idx = code_line.find('# call: ')
                if idx != -1:
                    dep_name = code_line[idx + 8:]
                    assert dep_name in CustomizedOps.kOpCodeDef, dep_name
                    ret = ret + get_dep_names(dep_name)
                    ret.append(dep_name)
            return ret

        if node.signature in CustomizedOps.kOpCodeDef:
            code_impl: str = CustomizedOps.kOpCodeDef[node.signature]
            local = {}
            exec(code_impl, globals(), local)
            fn = list(local.values())[0]
        else:
            fn = eval(node.signature)
        return fn


class ProfileDataBase:

    def __init__(self, filename: Optional[str] = None) -> None:
        """!
        Create a database for profiling result
        """
        self._data: Dict[str, Dict[str, Tuple[float, float, int]]] = dict()
        if filename is not None:
            self.load(filename)

    def profile(self, node: IRFwOperation, train: bool = True, device: Optional[int] = None):
        """
        Profile a forward node in IRGraph on a specific device (default current device)
        
        @param node IRFwOperation: node of IRGraph
        @param device int: the device that the node will execute on
        
        @return latency float: average latency in ms
        @return memory int: average memory in bytes
        """
        if self.exist(node):
            return self.query(node)

        if isinstance(device, int):
            orig_device = torch.cuda.current_device()
            torch.cuda.set_device(device)

        color, default = '\033[31m', '\033[0m'

        #FIXME: OOM will increase cuda allocated memory
        try:
            latency, memory = CompProfiler.profile(node, train)
            # log to database
            self.insert(node, latency, memory)
        except Exception as e:
            err = f'{color}profil error:\n {str(e)}{default}'
            print(err)
            latency, memory = e, e
        
        shapes = tuple(t.shape if isinstance(t, IRTensor) else None for t in node.inputs())
        dtypes = tuple(t.dtype if isinstance(t, IRTensor) else None for t in node.inputs())
        error = f'{color}None{default}'
        print(
            f"profiled {node.signature} | shapes: {shapes} | dtypes: {dtypes} | train {train} => "
            f"latency: {round(latency, 2) if isinstance(latency, float) else error} ms | "
            f"memory {memory if isinstance(memory, int) else None} bytes")

        if isinstance(device, int):
            torch.cuda.set_device(orig_device)
        return latency, memory

    def insert(self, node: IRCell, latency: float, memory: int):
        """
        log (reset) the span of a node with key

        @param node IRCell
        @param latency float: inference time in milliseconds
        @param memory int: inference peak memory in bytes
        """
        name = node.signature
        key = self._serialize(node)
        assert isinstance(name, str) and isinstance(key, str)
        if name not in self._data:
            self._data[name] = dict()
        latency = latency if isinstance(latency, float) else None
        memory = memory if isinstance(memory, int) else None
        self._data[name][key] = (latency, memory)

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

    def query(self, node: IRFwOperation) -> Tuple[Tuple[int], Tuple[int], float, float, int, Tuple[int]]:
        """!
        Get the performance number of a node in IRGraph

        @param node IRFwOperation: node in IRGraph

        @return latency float: average latency in ms
        @return memory int: average memory in bytes
        """
        key = self._serialize(node)
        if node.signature not in self._data:
            return None
        if key not in self._data[node.signature]:
            return None
        return self._data[node.signature][key]

    def _serialize(self, node: IRFwOperation) -> str:
        """
        Serialize the shapes, dtypes and kwargs into a string

        e.g.,
            shapes: ((1024,), (1024,1024))
            dtypes: (torch.float32, torch.float32)
        => ((1024,), (1024,1024)) : (torch.float32, torch.float32)

        @param shapes Tuple[Tuple[int]]: the shape of each tensor
        @param dtypes Tuple[torch.dtype]: the dtype of each tensor

        @return key str: the serialized string
        """
        shapes, dtypes = [], []
        for t in node.inputs():
            if isinstance(t, IRTensor):
                shapes.append(t.shape)
                dtypes.append(t.dtype)
            elif isinstance(t, IRObject):
                raise RuntimeError('IRObject has not been supported in _serialize')
            else:
                shapes.append(None)
                dtypes.append(type(t))
        shapes = str(tuple(shapes))
        dtypes= str(tuple(dtypes))
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
        shapes = eval(shapes)
        dtypes = eval(dtypes)
        # shapes = tuple(eval(shape) for shape in shapes.split('-'))
        # dtypes = tuple(eval(dtype) for dtype in dtypes.split('-'))
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
                latency, memory = self._data[signature][key]
                data.append(f'{signature}: shapes={shapes}, dtypes={dtypes}, latency {latency:.2f} msm, memory {memory} bytes')
        data = '\n'.join(data)
        return data


class Estimator:
    """
    Estimator to measture the computation / memory cost of a subgraph
    """
    def __init__(self, cache='./profile_database.json'):

        self.cache_file = cache
        reload = cache if os.path.exists(cache) else None
        self.database = ProfileDataBase(reload)

    def profile(self, node: IRFwOperation, train: bool) -> Tuple[float, int]:
        if node.name == 'multiref' or isinstance(node, IRGraphAnchor): return 0.0, 0
        trials = Estimator.special_rules(node, [None])
        for config in trials:
            if config is None:
                num = 1
                latency, memory = self.database.profile(node, train)
            else:
                idx, dim, num = config
                print(f'> ... try node {node.name} with idx={idx}, dim={dim}, num={num}')
                sub_node = node.algorithms('dim').instantiate(idx=idx, dim=dim, num=num)[0]
                latency, memory = self.database.profile(sub_node, train)
                if isinstance(latency, float): break
            if isinstance(latency, float): break
        assert isinstance(latency, float), f"Failed to profile: {node}"
        latency, memory = latency * num, memory * num
        self.database.insert(node, latency, memory)
        return latency, memory

    def __call__(self, nodes_or_segment: Union[Tuple[IRFwOperation], IRSegment], 
                 train: bool = True):
        """
        Profile the computation cost of a subgraph

        @param nodes_or_segment Tuple[IRFwOperation] | IRSegment

        @return latency float: latency in ms
        @return memory int: memory in bytes
        """
        nodes = nodes_or_segment.nodes() if isinstance(nodes_or_segment, IRSegment) else nodes_or_segment
        memory, latency = 0.0, 0.0
        for node in nodes:
            if self.database.exist(node):
                node_latency, node_memory = self.database.query(node)
            else:
                node_latency, node_memory = self.profile(node, train)
            if train:
                memory += node_memory
                latency += node_latency
            else:
                memory = max(memory, node_memory)
                latency += node_latency
        return latency, memory

    def save(self):
        self.database.dump(self.cache_file, override=True)

    def special_rules(node, trials):
        # if node.name == 'embedding':  # for GPT
        #     trials = [(1, 0, 4),]
        # if node.name == 'self_attention':  # for GPT
        #     trials = [(1, 0, 4),]
        # if node.name == 'window_attn':  # for Swin
        #     trials = [(1, 0, 4),]
        return trials
