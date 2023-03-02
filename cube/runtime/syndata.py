r"""
Synthetic Data Loader
"""

from typing import Any, List, Optional, Tuple, Union
import torch
import warnings


class CubeDataLoader:
    r"""
    Cube Dataloader.
    User should provide a dataloader to runtime with at least these functionalities:

    1) `__iter__()`: get the dataloder iterator
    2) `__next__()` get the next batch of data
    3) `get_batch_size()` return the batch size (int)
    4) `set_batch_size(bs)` reset the batch size (int)
    5) `get_batch_dims(self)` get the batch dimension of each output data
    """
    def __init__(self, batch_size: int, batch_dims: Tuple[Optional[int]]):
        """
        Create a dataloader for cube runtime

        @param batch_size int: dataloader batch size
        @param batch_dims Tuple[Optional[int]]: the batch dimension of each output data,
            None indicates the output (tensor or non-tensor) doesn't have the batch dimension.
        """
        self.batch_size: int = batch_size
        self.batch_dims: Tuple[Optional[int]] = batch_dims

    def __iter__(self):
        raise NotImplementedError("Required implementation for derived class")

    def __next__(self):
        return NotImplementedError("Required implementation for derived class")

    def get_batch_size(self) -> int:
        """
        get batch size
        """
        return self.batch_size

    def set_batch_size(self, batch_size: int):
        """
        set batch size
        """
        return NotImplementedError("Required implementation for derived class")
    
    def get_batch_dims(self) -> Tuple[Optional[int]]:
        return tuple(self.batch_dims)


class SciLoopVariables(CubeDataLoader):
    r"""Scientific loop variable loader
    """
    def __init__(self, variables: List[Any], constants: List[Any]):
        shapes = []
        dtypes = []
        for var in variables + constants:
            if torch.is_tensor(var):
                shapes.append(list(var.size()) if len(var.size()) != 0 else [1,])
                dtypes.append(var.dtype)
            else:
                shapes.append([1,])
                dtypes.append(type(var))
        super().__init__(0, [None] * len(shapes))
        self.variables = list()
        self.constants = list()
        for var in variables:
            if torch.is_tensor(var) and var.device != torch.cuda.current_device():
                var = var.cuda()
            self.variables.append(var)
        for const in constants:
            if torch.is_tensor(const) and const.device != torch.cuda.current_device():
                const = const.cuda()
            self.constants.append(const)

    def get_batch_size(self) -> int:
        return 0

    def set_batch_size(self, batch_size: int):
        return

    def __iter__(self):
        return self

    def __next__(self):
        if len(self.variables) + len(self.constants) == 1:
            return (self.variables + self.constants)[0]
        return tuple(self.variables + self.constants)

    def update(self, variables: Optional[List[Any]] = None, constants: Optional[List[Any]] = None):
        """
        Update variables and constants
        """
        if variables is not None:
            if len(variables) != len(self.variables):
                raise ValueError(f"Expected {len(self.shapes)} but only got {len(variables)} varaibales to update")
            for var, expected_shape in zip(variables, self.shapes):
                expected_shape = tuple(expected_shape)
                if not torch.is_tensor(var) and expected_shape != (1,):
                    raise ValueError(f"Non-tensor variable: Expected shape is (1,)")
                if torch.is_tensor(var) and tuple(var.size()) != expected_shape:
                    raise ValueError(f"Shape update mismatch: var: {var.size()} != expected: {expected_shape}")
            self.variables = variables
        if constants is not None:
            if len(constants) != len(self.constants):
                raise ValueError(f"Expected {len(self.shapes)} but only got {len(constants)} varaibales to update")
            for const, expected_shape in zip(constants, self.shapes):
                expected_shape = tuple(expected_shape)
                if not torch.is_tensor(const) and expected_shape != (1,):
                    raise ValueError(f"Non-tensor constant: Expected shape is (1,)")
                if torch.is_tensor(const) and tuple(const.size()) != expected_shape:
                    raise ValueError(f"Shape update mismatch: const: {const.size()} != expected: {expected_shape}")
            self.constants = constants


class SynDataLoader(CubeDataLoader):
    r"""
    Synthetic dataloader to produce tensors
    for given shapes, dtypes.
    """
    def __init__(self, shapes: Tuple[List[int]], dtypes: Tuple[torch.dtype] = None,
                 batch_dims: Tuple[int] = None, names: Tuple[str] = None, append_args=None, device=None):
        """
        shapes Tuple[Tuple[int]]:
            The shape for each data
        dtypes Tuple[torch.dtype]:
            The dtype for each data (Default None: use torch.float32)
        batch_dims Tuple[int]:
            The batch dimension of each data (Default None: dimension 0 is the batch dim)
        """
        if batch_dims is None:
            batch_dims = tuple([0] * len(shapes))
        if dtypes is None:
            dtypes = tuple([torch.float] * len(shapes))
        self.shapes = tuple([list(shape) for shape in shapes])
        self.dtypes = dtypes
        batch_size = shapes[0][batch_dims[0]]
        super().__init__(batch_size, batch_dims)
        self.names = names
        self.append_args=append_args
        self.device = device if device else torch.cuda.current_device()
        self.buffer: Union[torch.Tensor, Tuple[torch.Tensor]] = None
        datas = self.random_sample()
        self.set_output(datas)

    def __iter__(self):
        return self

    def __next__(self):
        if self.names:
            assert len(self.names) == len(self.buffer)
            return dict(zip(self.names, self.buffer)).update(self.append_args)
        else:
            return self.buffer

    def random_sample(self) -> Tuple[torch.Tensor]:
        torch.manual_seed(0)
        datas = []
        for shape, dtype in zip(self.shapes, self.dtypes):
            datas.append(
                torch.rand(
                    shape,
                    device=self.device,
                    requires_grad=False).to(dtype)
                if torch.is_floating_point(torch.zeros([1], dtype=dtype)) else
                torch.ones(
                    shape,
                    device=self.device,
                    requires_grad=False
                ).to(dtype)
            )
        return tuple(datas)
    
    def set_output(self, datas: Union[torch.Tensor, Tuple[torch.Tensor]]):
        datas = (datas,) if torch.is_tensor(datas) else tuple(datas)
        if len(datas) == 0:
            self.buffer = None
        else:
            self.buffer = datas[0] if len(datas) == 1 else datas

    def set_batch_size(self, batch_size: int):
        self.batch_size = batch_size
        for shape, dim in zip(self.shapes, self.batch_dims):
            shape[dim] = batch_size
        rank = 0 if not torch.distributed.is_initialized() else torch.distributed.get_rank()
        print(f'rank [{rank}]: > set batch size to {batch_size}. dataloader outputs change to: {self.shapes}')
        datas = self.random_sample()
        self.set_output(datas)
