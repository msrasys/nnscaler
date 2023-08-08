from typing import Callable, Any, Optional, Type, Union
from pathlib import Path
import inspect
import sys
import importlib
from dataclasses import dataclass

import torch
from cube.graph.parser.fx.parser import FxModuleParser

from cube.ir.cten import IRObject
from cube.ir.tensor import IRFullTensor

from cube.graph import IRGraph
from cube.graph import parser
from cube.graph.parser.dtype import DType2IRDType
from cube.graph.function.anchor import IRGraphAnchor
from cube.graph.function.pyfunc import IRPyFunc
from cube.graph.schedule.schedplan import SchedulePlan
from cube.graph.gener.gen import IRAdapterGener

from cube.codegen import ModuleCodeGen
from cube.execplan import ExecutionPlan
from cube.execplan.planpass.grouping import Grouping
from cube.execplan.planpass.fusion import DiffFusion
from cube.ir.unique import IDGenerator
from cube.program import Program
from cube.runtime.module import CubeModule, ParallelModule


@dataclass
class ComputeConfig:
    plan_ngpus: int
    runtime_ngpus: int


def _complex(val: Any):
    """Complex to CPU"""
    if isinstance(val, tuple):
        return tuple(_complex(t) for t in val)
    if isinstance(val, list):
        return list(_complex(t) for t in val)
    if isinstance(val, dict):
        return {_complex(key):_complex(val) for key, val in val.items()}
    if isinstance(val, set):
        return {_complex(t) for t in val}
    if isinstance(val, torch.Tensor):
        return val.cpu()
    return val


def _get_full_qualified_name(obj: Any) -> str:
    """Get full qualified name of an object"""
    if inspect.isclass(obj):
        return obj.__module__ + '.' + obj.__qualname__
    return obj.__module__ + '.' + obj.__class__.__qualname__


def _add_cube_savedir_to_syspath(cube_savedir: str) -> Path:
    cube_savedir = Path(cube_savedir).resolve()
    cube_savedir.mkdir(parents=True, exist_ok=True)
    if str(cube_savedir) not in sys.path:
        sys.path.append(str(cube_savedir))
    return cube_savedir


def _is_any_gencode_loaded(namespace: str) -> bool:
    """Check if a module is loaded"""
    for m in sys.modules.values():
        if m.__name__.startswith(namespace + '.' + _GENCODE_FILE_PREFIX):
            return True
    return False


_GENCODE_FILE_PREFIX = 'gencode'
_GENCODE_FILE_TEMPLATE = _GENCODE_FILE_PREFIX + '{}.py'  # 'gencode{}.py'
_CUBE_MODULE_NAMESPACE = '_cube_modules'


def _gencode(
        module: torch.nn.Module,
        dummy_input: dict,
        pas_policy: Callable[[IRGraph, ComputeConfig], IRGraph],
        compute_config: ComputeConfig,
        *,
        dynamic_shape: bool = True,
        cube_savedir: Union[str, Path] = './.cube',
        override: bool = False,
        instance_name: Optional[str] = None
    ) -> None:
    """
    Generate cube module source code from a torch module, and save it to file.
    Generated module will be save according to its full qualified name.

    If you want to save multiple instances of the same module,
    you can specify the instance_name to distingish them.

    For example, if the module is `torchscale.x.y`, then the generated module will be save to
    `cube_savedir/_cube_modules/torchscale/x/y/instance_name`.

    Args:
        module (torch.nn.Module): the module to be compiled
        dummy_input (dict): the dummy input for the module
        pas_policy (Callable[[IRGraph, ComputeConfig], IRGraph]): the pas policy
        compute_config (ComputeConfig): the environment resource
        dynamic_shape (bool): whether to use dynamic shape
        override (bool): If true, source code will be regenerated even if generated code exists.
        cube_savedir (Union[str, Path]): the directory to save generated code
        instance_name (Optional[str]): the instance name of the generated module.

    Returns:
        None
    """
    # put cube_savedir into sys.path
    # so we can import the generated module with its namespace later
    cube_savedir = _add_cube_savedir_to_syspath(cube_savedir)

    instance_name = instance_name.strip('.') if instance_name else ''
    instance_namespace = f'.{instance_name}' if instance_name else ''
    namespace = f'{_CUBE_MODULE_NAMESPACE}.{_get_full_qualified_name(module)}{instance_namespace}'
    outdir = cube_savedir / Path(namespace.replace('.', '/').strip('/'))
    outdir.mkdir(parents=True, exist_ok=True)

    # decision matrix for code generation
    # override flag | dir condition(imported, empty, match, unmatched) | action
    # ---------------------------------------------------------
    #   True   | empty | generate
    #   True   | imported | raise error
    #   True   | match | generate
    #   True   | unmatch | generate
    #   False  | empty | generate
    #   False  | match | do nothing
    #   False  | unmatch | raise error
    #   False  | imported | doesn't matter
    if not override:
        # check if the module is already generated
        expected_output_files = [outdir / _GENCODE_FILE_TEMPLATE.format(rank) for rank in range(compute_config.plan_ngpus)]
        expected_output_files.append(outdir / FxModuleParser.ATTR_CONTENT_FILE)
        expected_output_files.append(outdir / FxModuleParser.ATTR_MAP_FILE)
        existing_output_files = [f for f in outdir.glob('*') if f.is_file()]
        if existing_output_files:
            if all([output_file.exists() for output_file in expected_output_files]) \
                and len(existing_output_files) == len(expected_output_files):
                return
            else:
                raise RuntimeError(f'Output directory {outdir} is not empty. '
                                   f'And the existing files do not match with current config.')
    else:
        # check if the module is already loaded
        if _is_any_gencode_loaded(namespace):
            raise RuntimeError(f'Output directory {outdir} is already loaded. '
                               f'You can not override a loaded module.')
        # clear existing generated files
        for f in outdir.glob('*'):
            if f.is_file():
                f.unlink()

    # reset environment
    program = Program()
    program.clear()
    IDGenerator().clear()

    module = module.to(device=torch.device("cpu"))
    module.train()

    # generate fx graph
    dummy_input = _complex(dummy_input)
    fx_graph = parser.to_fx_graph(module, dummy_input)

    # generate ir logic graph
    ir_graph = parser.to_ir_graph(
        fx_graph, dummy_input, outdir, dynamic_shape
    )

    # generate dummy inputs for logic graph
    # that is, generate IRObject/IRFullTensor for fx graph dummpy input
    fx_input_nodes = [node for node in fx_graph.graph.nodes if node.op == 'placeholder']
    # the inputs of graph is different with original forward args
    # so we get the real forward args from fx inputs
    forward_args = [node.target for node in fx_input_nodes]
    ir_dummy_inputs = []
    for node in fx_input_nodes:
        if node.target.startswith('*'):  # *args or **kwargs
            if node.target.strip('*') in dummy_input:
                raise ValueError(f"Input {node.target}: *args or **kwargs is not suppported")
            ir_dummy_inputs.append(None)  # always set None to *args/**kwargs
        elif node.target in dummy_input:
            ir_dummy_inputs.append(dummy_input[node.target])
        else:
            raise ValueError(f"Input {node.target} not in dummy input. Default value is not supported.")
    for i in range(len(ir_dummy_inputs)):
        if isinstance(ir_dummy_inputs[i], torch.Tensor):
            # note: we will always set tensor to require gradient, which may
            # generate backward communications in adapter. However, as long as
            # the data doesn't require gradient in real runtime, the backward
            # communication will not be triggered.
            ir_dummy_inputs[i] = IRFullTensor(
                shape=ir_dummy_inputs[i].size(),
                name=fx_input_nodes[i].target,
                requires_grad=True,
                dtype=DType2IRDType.map(ir_dummy_inputs[i].dtype)).tosub()
            ir_dummy_inputs[i].grad = ir_dummy_inputs[i].parent.grad.tosub()
        else:
            ir_dummy_inputs[i] = IRObject(
                name=fx_input_nodes[i].target,
                value=ir_dummy_inputs[i]
            )
    # generate complete ir graph
    ir_dummy_outputs = ir_graph(*ir_dummy_inputs)

    graph = program.get_graph()
    graph.backward()
    program.finalize()
    program.set_input(ir_dummy_inputs)
    if ir_dummy_outputs is None: ir_dummy_outputs = []
    elif not (isinstance(ir_dummy_outputs, tuple) or isinstance(ir_dummy_outputs, list)):
        ir_dummy_outputs = [ir_dummy_outputs]
    program.set_output(ir_dummy_outputs)

    graph = pas_policy(graph, compute_config)
    if not isinstance(graph, IRGraph):
        raise RuntimeError("Expected policy return IRGraph")

    # check assignment and remove anchor node
    for node in graph.nodes(flatten=True):
        # skip graph anchor and multiref: they will be removed or replaced by system
        if isinstance(node, IRGraphAnchor) or node.name == 'multiref':
            graph.assign(node, 0)
        if isinstance(node, IRPyFunc):
            graph.assign(node, 0)
        if len(node.device) == 0:
            raise RuntimeError(f"Node {node} device is not set")
    graph = IRAdapterGener.gen(graph, cost_fn=None)
    if graph.sched is not None:
        graph.sched.apply()

    if isinstance(graph.sched, SchedulePlan):
        execplan = ExecutionPlan.from_schedplan(graph.sched)
    else:
        execplan = ExecutionPlan.from_graph(graph)

    execplan = DiffFusion.apply(execplan)
    # plan pass for computation grouping
    if not graph.sched:
        execplan = Grouping.apply(execplan)

    # code generation
    runtime_ngpus = None if compute_config.plan_ngpus == compute_config.runtime_ngpus else compute_config.runtime_ngpus
    assert len(execplan.graph.device) == compute_config.plan_ngpus, f"{execplan.graph.device}"
    mgener = ModuleCodeGen(execplan, scale_ndevs=runtime_ngpus)
    for rank in range(compute_config.plan_ngpus):
        filename = _GENCODE_FILE_TEMPLATE.format(rank)
        mgener.gen(rank, forward_arg_names=forward_args, outfile=outdir / filename, attach=False, as_parallel_module=True)


def _load_cube_module_class(
    module_class: Type[torch.nn.Module],
    *,
    cube_savedir: Union[str, Path] = './.cube',
    instance_name: Optional[str] = None,
):
    """
    Load the generated cube module class.

    Please note that the cube module class should be generated beforehand by _gencode().

    Args:
        module_class (Type[torch.nn.Module]): the original module class
        cube_savedir (Union[str, Path]): the directory to load generated code
        instance_name (Optional[str]): the instance name of the generated module.
    """
    _add_cube_savedir_to_syspath(cube_savedir)
    rank = torch.distributed.get_rank()
    instance_name = instance_name.strip('.') if instance_name else ''
    instance_namespace = f'.{instance_name}' if instance_name else ''
    gen_imported = importlib.import_module(
        f'{_CUBE_MODULE_NAMESPACE}.{_get_full_qualified_name(module_class)}{instance_namespace}.{Path(_GENCODE_FILE_TEMPLATE.format(rank)).stem}'
    )
    cube_module_class = gen_imported.GenModel
    # rewrite class name and module name
    cube_module_class.__name__ = module_class.__name__
    cube_module_class.__qualname__ = module_class.__qualname__
    # cube_module_class.__module__ = module_class.__module__
    cube_module_class.__orig_module_class__ = module_class  # save the original module class
    return cube_module_class


def parallelize(
    module_or_module_class: Union[torch.nn.Module, Type[torch.nn.Module]],
    dummy_input: dict,
    pas_policy: Callable[[IRGraph, ComputeConfig], IRGraph],
    compute_config: ComputeConfig,
    *,
    dynamic_shape: bool = True,
    cube_savedir: Union[str, Path] = './.cube',
    override: bool = False,
    instance_name: Optional[str] = None,
) -> Union[CubeModule, Type[CubeModule]]:
    """
    Convert a torch.nn.Module object or class to CubeModule object or class.

    If you want to save multiple instances of the same module,
    you can specify the instance_name to distingish them.

    Args:
        module_or_module_class (Union[torch.nn.Module, Type[torch.nn.Module]]): the module or module class to be compiled
        dummy_input (dict): the dummy input for the module
        pas_policy (Callable[[IRGraph, ComputeConfig], IRGraph]): the pas policy
        compute_config (ComputeConfig): the environment resource
        dynamic_shape (bool): whether to use dynamic shape
        override (bool): If true, source code will be regenerated even if generated code exists.
        cube_savedir (Union[str, Path]): the directory to save generated code
        instance_name (Optional[str]): the instance name of the generated module.

    Returns:
        Union[CubeModule, Type[CubeModule]]: the converted CubeModule object or class
    """
    if (
        isinstance(module_or_module_class, CubeModule) or
        (inspect.isclass(module_or_module_class) and issubclass(module_or_module_class, CubeModule))
    ):
        return module_or_module_class

    if not torch.distributed.is_initialized(): # we only support distributed training
        raise RuntimeError("Distributed training is not initialized.")

    rank = torch.distributed.get_rank()
    is_module_class = inspect.isclass(module_or_module_class)
    module_class = module_or_module_class if is_module_class else module_or_module_class.__class__

    if rank == 0:
        if is_module_class:
            # it should only have 1 `self` parameter
            if len(inspect.signature(module_or_module_class.__init__).parameters) > 1:
                raise ValueError("Module class __init__ should be parameter-free.")
            try:
                module = module_or_module_class()
            except Exception as e:
                raise RuntimeError(f"Error when create module instance.") from e
        else:
            module = module_or_module_class

        # TODO: copy generated files to other nodes
        # Currently you must use a shared file system to share the generated files (like mounted Azure Blob)
        # Or you can manually copy the generated files to other nodes
        _gencode(
            module,
            dummy_input,
            pas_policy,
            compute_config,
            dynamic_shape=dynamic_shape,
            override=override,
            cube_savedir=cube_savedir,
            instance_name=instance_name,
        )
        if is_module_class:
            del module
    torch.distributed.barrier()
    cube_module_class = _load_cube_module_class(
        module_class,
        cube_savedir=cube_savedir,
        instance_name=instance_name,
    )
    return cube_module_class if is_module_class else cube_module_class()


def parallel_module(
        dummy_input: dict,
        pas_policy: Callable[[IRGraph, ComputeConfig], IRGraph],
        compute_config: ComputeConfig,
        *,
        dynamic_shape: bool = True,
        cube_savedir: Union[str, Path] = './.cube'
) -> Callable[[Union[torch.nn.Module, Type[torch.nn.Module]]], Union[CubeModule, Type[CubeModule]]]:
    """
    Work as a class decorator to convert a torch.nn.Module to CubeModule.

    Please make sure the Module's __init__ is paremeter-free.
    Please note that
    1. Returned CubeModule will replace the torch.nn.Module in-place.
    And all member functions/variables of original torch.nn.Module will be gone.
    2. The parameters of CubeModule will be fixed,
    which means all instances of CubeModule will use the same parameters (which are from the tracing).

    Args:
        dummy_input (dict): the dummy input for the module
        pas_policy (Callable[[IRGraph, ComputeConfig], IRGraph]): the pas policy
        compute_config (ComputeConfig): the environment resource
        dynamic_shape (bool): whether to use dynamic shape
        cube_savedir (Union[str, Path]): the directory to save generated code
    """
    def wrap(module_or_module_class: Union[torch.nn.Module, Type[torch.nn.Module]]) -> Union[CubeModule, Type[CubeModule]]:
        return parallelize(
                module_or_module_class,
                dummy_input,
                pas_policy,
                compute_config,
                dynamic_shape=dynamic_shape,
                override=False,
                cube_savedir=cube_savedir
        )
    return wrap
