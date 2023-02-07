r"""
Synthetic Data Loader
"""

from typing import Any, List, Optional, Tuple, Union
import torch
import warnings


class CubeDataLoader:
    r"""
    Cube Dataloader
    """
    def __init__(self, shapes: Tuple[List[int]], dtypes: Tuple[torch.dtype], batch_dims: Tuple[int] = None):
        """
        shapes Tuple[Tuple[int]]:
            The shape for each data
        dtypes Tuple[torch.dtype]:
            The dtype for each data
        batch_dims Tuple[int]:
            The batch dimension of each data
        """
        if not all(isinstance(shape, list) for shape in shapes):
            raise TypeError("Expected each shape in shapes to be a list")
        if len(shapes) != len(batch_dims) or len(shapes) != len(dtypes):
            raise TypeError("Expected number batch dim and dtypes to len(shapes)")
        self.shapes = tuple([list(shape) for shape in shapes])
        self.dtypes = dtypes
        self.batch_dims = (0,) * len(self.shapes) if batch_dims is None else batch_dims
        bs = [shape[dim] for shape, dim in zip(self.shapes, self.batch_dims)]
        assert len(set(bs)) == 1, f"Expect batch size same in each data shapes"
        self.batch_size = bs[0]

    def get_batch_size(self) -> int:
        """
        get batch size
        """
        all_batch_size = set([shape[dim] for shape, dim in zip(self.shapes, self.batch_dims)])
        if len(all_batch_size) != 1:
            raise ValueError("Heterogenous batch size in dataloader")
        return list(all_batch_size)[0]

    def set_batch_size(self, batch_size: int):
        """
        set batch size
        """
        self.batch_size = batch_size
        for shape, dim in zip(self.shapes, self.batch_dims):
            shape[dim] = batch_size
        rank = 0 if not torch.distributed.is_initialized() else torch.distributed.get_rank()
        print(f'rank [{rank}]: > set batch size to {batch_size}. dataloader outputs change to: {self.shapes}')


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
        batch_dims = [-1] * (len(variables) + len(constants))
        super().__init__(shapes, dtypes, batch_dims)
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
                 batch_dims: Tuple[int] = None):
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

        super().__init__(shapes, dtypes, batch_dims)
        datas = self.random_sample()
        self.set_output(datas)

    def __iter__(self):
        return self

    def __next__(self):
        return self.buffer

    def random_sample(self) -> Tuple[torch.Tensor]:
        torch.manual_seed(0)
        datas = []
        for shape, dtype in zip(self.shapes, self.dtypes):
            datas.append(
                torch.rand(
                    shape, dtype=dtype,
                    device=torch.cuda.current_device(),
                    requires_grad=False)
            )
        datas 
        if len(datas) == 0:
            self.buffer = None
        else:
            datas = tuple(datas) if len(datas) > 1 else datas[0]
        return tuple(datas) if len(datas) > 0 else datas
    
    def set_output(self, datas: Union[torch.Tensor, Tuple[torch.Tensor]]):
        datas = (datas,) if torch.is_tensor(datas) else tuple(datas)
        if len(datas) == 0:
            self.buffer = None
        else:
            self.buffer = datas[0] if len(datas) == 1 else datas

    def set_batch_size(self, batch_size: int):
        super().set_batch_size(batch_size)
        datas = self.random_sample()
        self.set_output(datas)

