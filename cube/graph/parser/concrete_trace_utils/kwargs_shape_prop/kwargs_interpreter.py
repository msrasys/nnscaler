import torch
import torch.fx
import torch.fx.traceback as fx_traceback
from torch.fx import Interpreter, Node, GraphModule
from typing import Optional, Union, Tuple, Dict, List, Any, Iterator, Callable, MutableMapping, Mapping
from torch.utils._pytree import tree_map

Target = Union[Callable[..., Any], str]

BaseArgumentTypes = Union[str, int, float, bool, complex, torch.dtype,
                          torch.Tensor, torch.device, torch.memory_format, torch.layout]

Argument = Optional[Union[
    Tuple[Any, ...],
    List[Any],
    Dict[str, Any],
    slice,
    Node,
    BaseArgumentTypes
]]


class KwargsInterpreter(Interpreter):
    def __init__(self, module : GraphModule, garbage_collect_values : bool = True, fake_device_type='cpu'):
        super().__init__(module, garbage_collect_values)
        assert fake_device_type in ('cpu', 'cuda')
        self.fake_device_type = fake_device_type

    def run(self,
            concrete_args: Union[Dict[str, Any], Tuple, MutableMapping[str, Any], Mapping[str, Any]] = None,
            initial_env: Optional[Dict[Node, Any]] = None,
            enable_io_preocessing: bool = True) -> Any:

        self.env = initial_env if initial_env else {}

        if isinstance(concrete_args, tuple):
            # if concrete_args is a tuple, then they are positional args
            # then they are consumed left-to-right by `placeholder` nodes.
            # Use an iterator to keep track of position and extract those values
            if enable_io_preocessing:
                args = self.module.graph.process_inputs(*concrete_args)
            self.args_iter: Iterator[Any] = iter(args)
            self.concrete_kwargs = None
        else:
            try:
                # concrete_args is a kwargs dict/mapping
                self.args_iter = None
                self.concrete_kwargs = concrete_args
                self.used_concrete_kwargs = []
                # get default values of parameters in `forward()` method
                import inspect
                fw = inspect.unwrap(self.module.forward)
                args_default_values = fw.__defaults__
                if args_default_values is not None:
                    fw_code = fw.__code__
                    n_args = fw_code.co_argcount + fw_code.co_kwonlyargcount
                    names_iter = iter(fw_code.co_varnames)
                    start_idx = 0
                    if fw_code.co_varnames[0] == 'self':
                        _ = next(names_iter)  # skip self
                        start_idx = 1
                    args_names = [next(names_iter) for idx in range(start_idx, n_args)]
                    diff_len = len(args_names) - len(args_default_values)
                    self.default_args = {args_names[idx + diff_len]: args_default_values[idx] for idx in
                                         range(len(args_default_values))}
                else:
                    self.default_args = {}
            except:
                raise RuntimeError(f'invalid concrete_args type: {type(concrete_args)}')

        assert (
                    self.args_iter is None or self.concrete_kwargs is None), 'can not use positional args and keyword args at the same time'

        for node in self.module.graph.nodes:
            if node in self.env:
                continue
            try:
                self.env[node] = self.run_node(node)
            except Exception as e:
                print(node.name, node.op, node.target)
                msg = f'While executing {node.format_node()}'
                msg = '{}\n\n{}'.format(e.args[0], msg) if e.args else str(msg)
                msg += f"\nOriginal traceback:\n{node.stack_trace}"
                e.args = (msg,) + e.args[1:]
                if isinstance(e, KeyError):
                    raise RuntimeError(*e.args)
                raise

            if self.garbage_collect_values:
                for to_delete in self.user_to_last_uses.get(node, []):
                    del self.env[to_delete]

            if node.op == 'output':
                output_val = self.env[node]
                return self.module.graph.process_outputs(output_val) if enable_io_preocessing else output_val

    def run_node(self, n: Node) -> Any:
        """
        Run a specific node ``n`` and return the result.
        Calls into placeholder, get_attr, call_function,
        call_method, call_module, or output depending
        on ``node.op``

        Args:
            n (Node): The Node to execute

        Returns:
            Any: The result of executing ``n``
        """
        args, kwargs = self.fetch_args_kwargs_from_env(n)
        return getattr(self, n.op)(n.target, args, kwargs)

    def placeholder(self, target: Target, args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
        """
        Execute a `placeholder` node.

        Args:
            target(Target): The call target for this node,
                exactly the argument name of the forward function
            args(Tuple): Tuple of positional args for this invocation
            kwargs(Dict): Dict of keyword arguments for this invocation

        Returns:
            Any: The argument value that was retrieved.
        """
        assert isinstance(target, str)
        if target.startswith('**'):
            # For a douvle-starred parameter, e.g., `**kwargs`,
            # retrieve all the remaining values from the concrete kwargs dict
            remaining_keys = [key for key in self.concrete_kwargs if key not in self.used_concrete_kwargs]
            return {key: self.concrete_kwargs[key] for key in remaining_keys}
        elif target.startswith('*'):
            assert self.concrete_kwargs is None, 'unexpected positional args in kwargs mode'
            return list(self.args_iter)
        else:
            if self.concrete_kwargs is not None:
                try:
                    ret_arg = self.concrete_kwargs[target]
                except KeyError:
                    return self.default_args[target]
                else:
                    self.used_concrete_kwargs.append(target)
                    return ret_arg
            else:
                try:
                    return next(self.args_iter)
                except StopAsyncIteration:
                    if len(args) > 0:
                        return args[0]
                    else:
                        raise RuntimeError(
                            f'Expected positional argument for parameter {target}, but one was not passed in!')

    def call_function(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
        assert not isinstance(target, str)
        if self.fake_device_type == 'cpu':
            to_cuda = lambda t: t.cuda() if isinstance(t, torch.Tensor) else t
            args = tree_map(to_cuda, args)
            kwargs = tree_map(to_cuda, kwargs)
            result = target(*args, **kwargs)
            if isinstance(result, torch.Tensor):
                return result.cpu()
            else:
                to_cpu = lambda t: t.cpu() if isinstance(t, torch.Tensor) else t
                return tree_map(to_cpu, result)
        else:
            return target(*args, **kwargs)

    def call_method(self, target : 'Target', args : Tuple[Argument, ...], kwargs : Dict[str, Any]) -> Any:
        # args[0] is the `self` object for this method call
        self_obj, *args_tail = args
        assert isinstance(target, str)
        if self.fake_device_type == 'cpu':
            self_obj = self_obj.cuda()
            to_cuda = lambda t: t.cuda() if isinstance(t, torch.Tensor) else t
            args_tail = tree_map(to_cuda, args_tail)
            kwargs = tree_map(to_cuda, kwargs)
            result = getattr(self_obj, target)(*args_tail, **kwargs)
            self_obj = self_obj.cpu()
            if isinstance(result, torch.Tensor):
                return result.cpu()
            else:
                to_cpu = lambda t: t.cpu() if isinstance(t, torch.Tensor) else t
                return tree_map(to_cpu, result)
        else:
            return getattr(self_obj, target)(*args_tail, **kwargs)

    def call_module(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
        assert isinstance(target, str)
        mod = self.fetch_attr(target)
        if self.fake_device_type == 'cpu':
            mod = mod.cuda()
            to_cuda = lambda t: t.cuda() if isinstance(t, torch.Tensor) else t
            args = tree_map(to_cuda, args)
            kwargs = tree_map(to_cuda, kwargs)
            result = mod(*args, **kwargs)
            if isinstance(result, torch.Tensor):
                return result.cpu()
            else:
                to_cpu = lambda t: t.cpu() if isinstance(t, torch.Tensor) else t
                return tree_map(to_cpu, result)
        else:
            return mod(*args, **kwargs)
