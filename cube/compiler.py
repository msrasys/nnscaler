from typing import Callable, Tuple, Union, Optional
import torch
import time

import cube

from cube.graph import parser
from cube.graph.adapter.gen import AdapterGener
from cube.graph.operator.operator import IRDataOperation

from cube.logics.pool import SchedulePool
from cube.logics.translator import LogicTranslator

from cube.execplan import ExectuionPlan
from cube.execplan.planpass.grouping import Grouping
from cube.execplan.planpass.fusion import P2PFusion

from cube.codegen.codegen import ModelCodeGen, ScheduleCodeGen

from cube.profiler.timer import print_each_rank
from cube.runtime.syndata import CubeDataLoader

class SemanticModel:

    def __init__(self, model: torch.nn.Module, input_shapes):
        """
        Create semantic model based on AI Scientist description.
        """
        from cube.graph import parser
        self.ir_graph = parser.convert_model(
            model, input_shapes=input_shapes
        )
        self._loaded_module = None

    def get_graph(self):
        return self.ir_graph

    def load_module(self, filename: str):
        import importlib.util
        spec = importlib.util.spec_from_file_location("GenModel", filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._loaded_module = module.GenModel().cuda()
        self._loaded_module.init_param()
        # sync parameters before start training
        self._loaded_module.sync_params()

    def get_gen_module(self):
        return self._loaded_module

    def clear_module(self):
        self._loaded_module = None

    def __call__(self, *args):
        if self._loaded_module:
            return self._loaded_module(*args)
        else:
            return self.ir_graph(*args)


def compile(model: SemanticModel, dataloader: Optional[CubeDataLoader] = None,
            PAS: Union[Callable, Tuple[Callable, Callable, Callable]] = None):
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

    Args:
        model: AI Scientist specified SemanticModel
        dataloader: dataloader used for training
        policy: tuple of transformation policy and scheduling policy
    """
    if not isinstance(model, SemanticModel):
        raise TypeError("Expect Semantic Model")
    if dataloader is None:
        # create empty dataloader
        dataloader = cube.runtime.syndata.SynDataLoader(shapes=(),dtypes=())
    if not isinstance(dataloader, CubeDataLoader):
        raise TypeError("Expect dataloader derived from CubeDataLoader")
    if callable(PAS):
        PAS = (PAS,)

    model_graph = model.get_graph()
    ir_dataloader = parser.convert_dataloader(dataloader)

    if torch.distributed.is_initialized():
        # multiple device
        myrank = torch.distributed.get_rank()
    else:
        # single device
        myrank = 0

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
        batch_size = torch.tensor([-1], dtype=torch.int).cuda()
        if myrank == 0:

            compile_start = time.time()

            SchedulePool().clear()
            resource = cube.runtime.resource.EnvResource()

            # logic translator
            outputs = fn(model_graph, ir_dataloader)
            if outputs is None:
                outputs = []
            graph = LogicTranslator.gen_logic_graph(outputs=outputs)

            if len(PAS) == 1:
                graph = PAS[0](graph, resource)
            elif len(PAS) == 3:
                P, A, S = PAS
                graph = P(graph, resource)
                graph = A(graph, resource)
                graph = S(graph, resource)

            # check assignment and order
            for node in graph.nodes():
                if len(node.device) == 0:
                    raise RuntimeError(f"Node {node} device is not set")

            # generate adapter
            graph = AdapterGener.gen(graph)

            # to execution plan
            execplan = ExectuionPlan(graph)

            # plan pass for communication optimization
            start = time.time()
            execplan = Grouping.apply(execplan)
            span = time.time() - start
            print('> planpass on grouping operations: {:.2f} s'.format(span))

            start = time.time()
            execplan = P2PFusion.apply(execplan)
            span = time.time() - start
            print('> planpass on p2pfusion operations: {:.2f} s'.format(span))

            # execplan.draw(outfile='execplan.png')

            if torch.distributed.is_initialized():
                world_size = torch.distributed.get_world_size()
            else:
                world_size = 1

            # code generation
            mgener = ModelCodeGen(execplan)
            sgener = ScheduleCodeGen(execplan)
            for rank in range(world_size):
                fname = filename.format(rank)
                # generate spatial module code
                mgener.gen(rank, outfile=fname, attach=False)
                # generate temporal schedule code
                sgener.gen(
                    device = rank,
                    outfile = fname,
                    attach=True
                )

            # setup batch size
            all_batch_size = set()
            dnodes = [node for node in graph.nodes() if isinstance(node, IRDataOperation)]
            for dnode in dnodes:
                bs = [out.shape[dim] for out, dim in zip(dnode.outputs(), dnode.get_batch_dims())]
                all_batch_size.update(bs)
            if len(all_batch_size) != 1:
                raise NotImplementedError(f"Heterogenous batch size {bs} is not supported")
            batch_size = torch.tensor(list(all_batch_size), dtype=torch.int).cuda()

            compile_end = time.time()
            compile_time = compile_end - compile_start
            print('> compile time: {:.2f} seconds'.format(compile_time))

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        # reset dataloader
        torch.distributed.broadcast(batch_size, src=0)
        batch_size = batch_size.item()
        print_each_rank(f'reseting dataloader batch size to {batch_size}')
        dataloader.reset(batch_size=batch_size)

        # load module
        filename = filename.format(myrank)
        print_each_rank(f'loading generated module from {filename} ...')
        model.load_module(filename)
        # load temporal schedule
        print_each_rank(f'loading generated schedule from {filename} ...')
        return _load_tschedule_fn(filename)

    return decorator
