from typing import Callable, Optional, Tuple, Union
import torch
import time

import cube

from cube.graph import parser
from cube.graph.adapter.gen import AdapterGener
from cube.graph.graph import IRGraph
from cube.graph.operator.operator import IRDataOperation

from cube.logics.pool import SchedulePool
from cube.logics.translator import LogicTranslator

from cube.execplan import ExectuionPlan
# from cube.execplan.planpass.torchadapt import TorchRefAdapter
# from cube.execplan.planpass.redundant import RemoveRedundantAdapters
# from cube.execplan.planpass.merge import MergeComputeSU
# from cube.execplan.planpass.gfuse import WeightGradAllreduceFusion
# from cube.execplan.planpass.p2pfusion import P2PFusion
from cube.execplan.planpass.grouping import Grouping
from cube.execplan.planpass.fusion import P2PFusion

from cube.codegen.codegen import ModelCodeGen, ScheduleCodeGen


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
        print(f'> loading generated spatial moduel from {filename}')
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


def compile(model: SemanticModel, dataloader,
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
        print(f'> [{myrank}] loading generated schedule from {filename} ...')
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
            fn(model_graph, ir_dataloader)
            graph = LogicTranslator.gen_logic_graph()

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
            # if not SUGraph.is_topo_order(sugraph.sus()):
            #     raise RuntimeError(f"SUGraph order is not topological order")

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


            # plan pass to adapt to pytorch semantic: multi branch gradient
            # TODO: residual support
            # execplan = TorchRefAdapter.apply(execplan)
            # plan pass to remove redundant sus
            # start = time.time()
            # execplan = RemoveRedundantAdapters.apply(execplan)
            # span = time.time() - start
            # print('> planpass on remove redundant adapter: {:.2f} s'.format(span))
            # # print(f'> after remove redundant adapters:\n {execplan}')
            # start = time.time()
            # execplan = MergeComputeSU.apply(execplan)
            # span = time.time() - start
            # print('> planpass on merge compute: {:.2f} s'.format(span))
            # # print(f'> after merge backward SU:\n {execplan}')
            # start = time.time()
            # execplan = WeightGradAllreduceFusion.apply(execplan)
            # span = time.time() - start
            # print('> planpass on grad allreduce: {:.2f} s'.format(span))
            # print(f'> after add allreduce:\n{execplan}')

            # start = time.time()
            # execplan = P2PFusion.apply(execplan)
            # span = time.time() - start
            # print('> planpass on p2p fusion: {:.2f} s'.format(span))
            # print(f'> after fuse P2P SU:\n {execplan}')

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

            # get dataloader batch size
            batch_size = dict()  # {devid: batch size}
            for node in graph.nodes():
                if isinstance(node, IRDataOperation):
                    batch_dim = node.get_batch_dims()[0]
                    dev_batch_size = node.outputs(0).shape[batch_dim]
                    batch_size[node.device[0]] = dev_batch_size
            all_batch_size = set([batch_size[dev] for dev in batch_size])
            if len(all_batch_size) != 1:
                raise NotImplementedError("Heterogenous batch size it not supported")
            batch_size = list(all_batch_size)[0]
            # assume batch_size is always first dimension
            batch_size = torch.tensor([batch_size], dtype=torch.int).cuda()

            compile_end = time.time()
            compile_time = compile_end - compile_start
            print(f'> compile time: {compile_time} seconds')

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        # reset dataloader
        torch.distributed.broadcast(batch_size, src=0)
        batch_size = batch_size.item()
        print(f'> reseting dataloader batch size to {batch_size}')
        dataloader.reset(batch_size=batch_size)

        # load module
        model.load_module(filename.format(myrank))
        # load temporal
        return _load_tschedule_fn(filename.format(myrank))
    
    return decorator
