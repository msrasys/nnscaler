from typing import Callable, Tuple, Union, Optional
import torch
import time
import os

import cube

from cube.graph.gener.gen import IRAdapterGener
from cube.graph.graph import IRGraph
from cube.ir.operator import IRDataOperation
from cube.graph.function.anchor import IRGraphAnchor

from cube.execplan import ExecutionPlan
from cube.execplan.planpass.fusion import DiffFusion
from cube.execplan.planpass.grouping import Grouping

from cube.codegen.codegen import ModelCodeGen, ScheduleCodeGen

from cube.profiler.timer import print_each_rank
from cube.runtime.device import DeviceGroup
from cube.runtime.syndata import CubeDataLoader

from cube.program import Program, SemanticDataLoader, SemanticModel
from cube.flags import CompileFlag


def compile(model: SemanticModel, dataloader: Optional[CubeDataLoader] = None,
            PAS: Union[Callable, Tuple[Callable, Callable, Callable]] = None,
            override = True, load_content = True) -> Callable:
    """
    AI Scientist calls like:

        @cube.compile(model, dataloader, policy=(trans_policy, schedule_policy))
        def train_step(model, dataloader):
            # do a 4-time gradient accumulation
            for acc_step, (data, label) in enumerate(dataloader):
                if acc_step < 4:
                    loss = model(data, label)
                    loss.backward()
                else:
                    break
        ...
        
        for epoch in range(100):
            train_step(model, data_loader)
            optimizer.step()
            optimizer.zero_grad()

        ...

    @param model SemanticModel: AI Scientist specified SemanticModel
    @param dataloader CubDataLoader: dataloader used for training
    @param policy Callable: policy to transform and schedule graph
    @param override bool: If true, the generated code will override exsisting
        files (if they are already existed.), otherwise, use the already existed
        generated code, i.e., the policy won't take effect. Default true.
    @param load_content bool: If true, will load parameter from exsiting saved models.
        Otherwise, will initial model parameters with empty tensor.

    @return sched_fn Callable: the scheduling function loaded from generated code.
    """
    if not isinstance(model, SemanticModel):
        raise TypeError("Expect Semantic Model")
    if dataloader is None:
        # create empty dataloader
        dataloader = cube.runtime.syndata.SynDataLoader(shapes=(),dtypes=())
    if not isinstance(dataloader, CubeDataLoader):
        raise TypeError("Expect dataloader derived from CubeDataLoader")

    model.save_content = load_content
    ir_dataloader = SemanticDataLoader(dataloader)

    myrank = DeviceGroup().rank

    def _load_tschedule_fn(filename) -> Callable:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_train_step", filename
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module._train_step

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
            return _load_tschedule_fn(filename)

        if DeviceGroup().local_rank == 0:

            compile_start = time.time()

            resource = cube.runtime.resource.EnvResource()

            # run once to get model structure and tensor shape
            outputs = fn(model, ir_dataloader)
            Program().finalize()
            if outputs is None:
                outputs = []
            elif not (isinstance(outputs, tuple) or isinstance(outputs, list)):
                outputs = [outputs]
            # setup program output
            Program().set_output(outputs)

            # run policy
            graph = Program().get_graph()
            assert callable(PAS), f"Policy PAS is not callable"
            graph = PAS(graph, resource)

            if not isinstance(graph, IRGraph):
                raise RuntimeError("Expected policy return IRGraph")

            # check assignment and remove anchor node
            for node in graph.nodes(flatten=True):
                if isinstance(node, IRGraphAnchor) or isinstance(node.mirror, IRGraphAnchor):
                    continue
                if len(node.device) == 0:
                    raise RuntimeError(f"Node {node} device is not set")

            # generate adapter
            start = time.time()
            graph = IRAdapterGener.gen(graph)
            span = time.time() - start
            print('> finish generating adapters: {:.2f} s'.format(span))

            if graph.sched is not None:
                start = time.time()
                graph.sched.apply()
                if CompileFlag.log_schedule:
                    print(graph.sched)
                span = time.time() - start
                print('> planpass on applying schedule strategy: {:.2f} s'.format(span))

            # to execution plan
            execplan = ExecutionPlan(graph)

            # plan pass for communication optimization
            start = time.time()
            execplan = DiffFusion.apply(execplan)
            span = time.time() - start
            print('> planpass on diff-fusion operations: {:.2f} s'.format(span))

            # execplan.visualize(outfile='plan.png')

            # plan pass for computation grouping
            if not graph.sched:
                start = time.time()
                execplan = Grouping.apply(execplan)
                span = time.time() - start
                print('> planpass on grouping operations: {:.2f} s'.format(span))

            # execplan.graph.reset_dependency()
            # execplan.analyze(outfile='execplan.png')

            local_world_size = DeviceGroup().local_world_size
            # code generation
            mgener = ModelCodeGen(execplan)
            sgener = ScheduleCodeGen(execplan)
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

        # set dataloder batch size (serialize output)
        bs = model.get_gen_module().get_batch_size()
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
        return _load_tschedule_fn(filename)

    return decorator
