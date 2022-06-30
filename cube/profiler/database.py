from typing import Callable, Tuple
import torch
import time


class CompProfiler:

    @staticmethod
    def profile(func: Callable, shapes: Tuple[Tuple[int]], dtypes=None, warmup_sec: float=2, prof_times: int = 50, **kwargs):
        """
        Profile a function

        @param func Callable: the callable function, e.g., torch.nn.functional.linear
        @param shapes Tuple[Tuple[int]]: the shapes of each input tensor
        @param dtypes Optional[Tuple[torch.dtype]]: the dtype of each input tensor. Default will use torch.float32

        @return span float: the time in milliseconds for forward + backward time
        """
        
        # create data
        dtypes = [torch.float32] * len(shapes) if dtypes is None else dtypes
        tensors = tuple(
            torch.rand(tuple(shape), dtype=dtype, device=torch.cuda.current_device(), requires_grad=True) \
                for shape, dtype in zip(shapes, dtypes)
        )
        outputs = func(*tensors, **kwargs)
        outputs = (outputs,) if torch.is_tensor(outputs) else outputs
        assert all(torch.is_tensor(otensor) for otensor in outputs), f"{func.__name__}: require all the outputs to be tensors"
        grads = tuple(torch.zeros_like(otensor) for otensor in outputs)

        # warmup
        tic = time.time()
        while time.time() - tic < warmup_sec:
            # forward
            outputs = func(*tensors, **kwargs)
            outputs = (outputs,) if torch.is_tensor(outputs) else outputs
            # backward
            torch.autograd.backward(outputs, grads)
        
        # profile forward
        torch.cuda.synchronize()
        tic = time.perf_counter()
        for _ in range(prof_times):
            # forward
            outputs = func(*tensors, **kwargs)
            outputs = (outputs,) if torch.is_tensor(outputs) else outputs
            # backward
            torch.autograd.backward(outputs, grads)
        torch.cuda.synchronize()
        toc = time.perf_counter()
        span = (toc - tic) / prof_times * 1000 # in milliseconds
        return span


if __name__ == '__main__':

    func = torch.nn.functional.linear

    shapes = ([2, 1024, 2304], [2, 2304])
    span = CompProfiler.profile(torch.nn.functional.linear, shapes)
    print(f'span of {func.__name__}: shapes: {shapes}: {span} ms')

    shapes = ([8, 1024, 2304], [8, 2304])
    span = CompProfiler.profile(torch.nn.functional.linear, shapes)
    print(f'span of {func.__name__}: shapes: {shapes}: {span} ms')
