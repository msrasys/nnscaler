from typing import Callable, Tuple, Union, Optional, Any
import torch
import time
import os

import cube

from cube.graph.gener.gen import IRAdapterGener
from cube.graph.graph import IRGraph
from cube.ir.cten import IRObject
from cube.graph.parser.dtype import DType2IRDType
from cube.ir.tensor import IRFullTensor
from cube.graph.function.anchor import IRGraphAnchor
from cube.graph.schedule.schedplan import SchedulePlan
from cube.graph.function.pyfunc import IRPyFunc

from cube.execplan import ExecutionPlan
from cube.execplan.planpass.fusion import DiffFusion
from cube.execplan.planpass.grouping import Grouping

from cube.codegen import ModuleCodeGen, ScheduleCodeGen

from cube.profiler.timer import print_each_rank
from cube.runtime.device import DeviceGroup
from cube.runtime.syndata import CubeDataLoader

from cube.program import Program, SemanticDataLoader, SemanticModel
from cube.ir.unique import IDGenerator
from cube.flags import CompileFlag


def compile(model: SemanticModel, *args,
            PAS: Union[Callable, Tuple[Callable, Callable, Callable]] = None,
            model_dummy_inputs: Tuple[Any] = None,
            model_dynamic_shape: bool = False,
            comm_cost_fn: Optional[Callable] = None,
            override = True,
            load_content = True,
            scale: Union[bool, int] = False) -> Callable:
    """Cube compile entry

    Examples:

    ```
    @cube.compile(model, data, PAS=policy)
    def train_iter(model, dataloader):
        data = next(dataloader)
        loss = model(data)
        loss.backward()
    ```

    Args:
        model (SemanticModel | torch.nn.Module): single-device model
        args (Tuple[Any]): compile function example inputs
        PAS (Callable | Tuple[Callable, Callable, Callable]): policy to transform and schedule graph
        model_dummy_inputs (Tuple[Any]): model example inputs when using torch.fx parser
        model_dynamic_shape (bool): whether to compile model with dynamic shape
        comm_cost_fn (Optional[Callable]): communication cost function, which
            takes in an IRAdapterPrim, and outputs a cost in float. By default (None) use
            communication volume.
        override (bool): If true, the generated code will override exsisting
            files (if they are already existed.), otherwise, use the already existed
            generated code, i.e., the policy won't take effect. Default true.
        load_content (bool): If true, will load parameter from exsiting saved models.
            Otherwise, will initial model parameters with empty tensor.
        scale (Union[bool, int]): If true, will scale the generated code to the
            total launched number of GPUs. If int, will scale to the specified number.
            Default False, no scaling.

    Returns:
        Callable: compiled training iteration
    """

    # clean global status
    Program().clear()
    IDGenerator().clear()
    assert PAS is not None, f'PAS should be callable function'

    if isinstance(model, torch.nn.Module):
        model = SemanticModel(model)
    assert isinstance(model, SemanticModel), f'Require cube.SemanticModel or torch.nn.Module, but got model: {type(model)}'
    model.save_content = load_content
    model.dynamic_shape = model_dynamic_shape
    model.dummy_input = model_dummy_inputs

    dataloader = None
    inputs = [model]
    for arg in args:
        assert not isinstance(arg, (torch.nn.Module, SemanticModel)), f"Only one model can be input for compile"
        if isinstance(arg, (torch.utils.data.Dataset, CubeDataLoader)):
            assert dataloader is None
            dataloader = arg
            arg = SemanticDataLoader(dataloader)
        elif isinstance(arg, torch.Tensor):
            arg = IRFullTensor(arg.shape, name='tensor', 
                               requires_grad=arg.requires_grad,
                               dtype=DType2IRDType.map(arg.dtype)).tosub()
        else:
            arg= IRObject('obj')
        inputs.append(arg)

    myrank = DeviceGroup().rank

    def decorator(fn: Callable) -> Callable:
        filename = 'gencode{}.py'

        if not override and os.path.exists(filename.format(myrank)):
            filename = filename.format(myrank)
            # TODO: set batch size
            print('warning: dataloader batch size stay as default.')
            # load module code
            print_each_rank(f'loading existed module from {filename} ...')
            model.load_module(filename)
            # load schedule code
            print_each_rank(f'loading existed schedule from {filename} ...')
            return cube.load_default_schedule(filename)

        if DeviceGroup().local_rank == 0:

            compile_start = time.time()
            resource = cube.runtime.resource.EnvResource()

            # run once to get model structure and tensor shape
            start = time.time()
            outputs = fn(*inputs)
            Program().finalize()
            if outputs is None:
                outputs = []
            elif not (isinstance(outputs, tuple) or isinstance(outputs, list)):
                outputs = [outputs]
            # setup program input
            pinputs = []
            for input in inputs[1:]: # we don't consider `model` as inputs
                if isinstance(input, SemanticModel):
                    pinputs.append('model')
                elif isinstance(input, SemanticDataLoader):
                    pinputs.append('dataloader')
                else:
                    pinputs.append(input)
            Program().set_input(pinputs)
            # setup program output
            Program().set_output(outputs)
            span = time.time() - start
            print('> finish parsing iteration: {:.2f} s'.format(span))

            # run policy
            start = time.time()
            graph = Program().get_graph()
            assert callable(PAS), f"Policy PAS is not callable"
            graph = PAS(graph, resource)
            span = time.time() - start
            print('> finish policy expression: {:.2f} s'.format(span))

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

            # generate adapter
            start = time.time()
            graph = IRAdapterGener.gen(graph, cost_fn=comm_cost_fn)
            span = time.time() - start
            print('> finish generating adapters: {:.2f} s'.format(span))

            if graph.sched is not None:
                start = time.time()
                graph.sched.apply()
                # print(graph.sched)qq
                if CompileFlag.log_schedule:
                    print(graph.sched)
                span = time.time() - start
                print('> finish planpass on applying schedule strategy: {:.2f} s'.format(span))

            # to execution plan
            start = time.time()
            if isinstance(graph.sched, SchedulePlan):
                execplan = ExecutionPlan.from_schedplan(graph.sched)
            else:
                execplan = ExecutionPlan.from_graph(graph)
            if CompileFlag.visualize_plan:
                execplan.visualize('plan.png')
            span = time.time() - start
            print('> finish lowering to execution plan: {:.2f} s'.format(span))

            # plan pass for communication optimization
            start = time.time()
            execplan = DiffFusion.apply(execplan)
            span = time.time() - start
            print('> finish planpass on diff-fusion operations: {:.2f} s'.format(span))

            # execplan.visualize(outfile='plan.png')

            # plan pass for computation grouping
            if not graph.sched:
                start = time.time()
                execplan = Grouping.apply(execplan)
                span = time.time() - start
                print('> finish planpass on grouping operations: {:.2f} s'.format(span))

            # execplan.graph.reset_dependency()
            # execplan.analyze(outfile='execplan.png')

            start = time.time()
            local_world_size = DeviceGroup().local_world_size
            # code generation
            scale_ndevs = None
            if scale:
                scale_ndevs = resource.ngpus if isinstance(scale, bool) else scale
            mgener = ModuleCodeGen(execplan, scale_ndevs=scale_ndevs)
            sgener = ScheduleCodeGen(execplan, scale_ndevs=scale_ndevs)
            for local_rank in range(local_world_size):
                rank = DeviceGroup().node_rank * local_world_size + local_rank
                fname = filename.format(rank)
                # generate spatial module code
                mgener.gen(rank, outfile=fname, attach=False)
                # generate temporal schedule code
                sgener.gen(
                    device = rank,
                    outfile = fname,
                    attach=True
                )
            span = time.time() - start
            print('> finish generating code: {:.2f} seconds'.format(span))

            compile_end = time.time()
            compile_time = compile_end - compile_start
            print('> compile time: {:.2f} seconds'.format(compile_time))

        if torch.distributed.is_initialized():
            if DeviceGroup().local_rank != 0 and CompileFlag.worker_sleep > 0:
                print(f'rank [{DeviceGroup().rank}] starts sleeping {CompileFlag.worker_sleep} seconds...')
                time.sleep(CompileFlag.worker_sleep)
            torch.distributed.barrier()

        # load module
        filename = filename.format(myrank)
        print_each_rank(f'loading generated module from {filename} ...')
        model.load_module(filename)

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        model.dummy_input = None
        # set dataloder batch size (serialize output)
        if dataloader is not None:
            bs = model.get_gen_module().get_batch_size()
            print_each_rank(f'> setting batch size to: {bs}')
            if torch.distributed.is_initialized():
                for rank in range(torch.distributed.get_world_size()):
                    if rank == torch.distributed.get_rank():
                        if bs is not None and dataloader is not None:
                            dataloader.set_batch_size(bs)
                    torch.distributed.barrier()
            else:
                if bs is not None and dataloader is not None:
                    dataloader.set_batch_size(bs)
        
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        # load temporal schedule
        print_each_rank(f'loading generated schedule from {filename} ...')
        return cube.load_default_schedule(filename)

    return decorator
