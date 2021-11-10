from typing import Callable, Optional
import torch
from cube.graph.graph import IRGraph

from cube.schedule.pool import SchedulePool
from cube.schedule.su import SUType
from cube.schedule.translator import IRDataLoader
from cube.schedule.sugraph import SUGraph, SUGraphGener
from cube.schedule.graphpass import SUGraphPass

from cube.codegen.codegen import ModelCodeGen, ScheduleCodeGen


class SemanticModel:

    def __init__(self, model: torch.nn.Module, input_shapes):
        """
        Create semantic model based on AI Scientist description.
        """
        from cube.graph import parser
        self.ir_graph = parser.convert(
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

    def get_gen_module(self):
        return self._loaded_module

    def clear_module(self):
        self._loaded_module = None

    def __call__(self, *args):
        if self._loaded_module:
            return self._loaded_module(*args)
        else:
            return self.ir_graph(*args)


def schedule(model: SemanticModel, dataloader,
             transform_policy: Optional[Callable] = None,
             schedule_policy:  Optional[Callable] = None):
    """
    AI Scientist calls like:

        @cube.tschedule.schedule(model, dataloader, policy_fn=policy)
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
    """
    if not isinstance(model, SemanticModel):
        raise TypeError("Expect Semantic Model")

    ir_graph = model.get_graph()
    ir_dataloader = IRDataLoader(dataloader)

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
            SchedulePool().clear()

            # logic translator
            fn(ir_graph, ir_dataloader)

            nodes = SchedulePool().nodes()

            # graph transformation
            graph = IRGraph(nodes, None, None, ir_graph.name)
            if transform_policy:
                graph = transform_policy(graph, None)

            # sugraph
            sugraph = SUGraphGener.gen_sugraph(graph.nodes())
            if schedule_policy:
                # TODO: add resource
                sugraph = schedule_policy(sugraph, None)

            # check assignment and order
            # print(sugraph)
            for su in sugraph.sus():
                if len(su.device) == 0:
                    raise RuntimeError(f"SU {su} device is not set")
            if not SUGraph.is_topo_order(sugraph.sus()):
                raise RuntimeError(f"SUGraph order is not topological order")

            # graph pass to remove redundant sus 
            sugraph = SUGraphPass.remove_redundant_adapters(sugraph)
            sugraph = SUGraphPass.merge_small_sus(sugraph)
            print(f'> after merge small sus:\n {sugraph}')

            if torch.distributed.is_initialized():
                world_size = torch.distributed.get_world_size()
            else:
                world_size = 1

            # code generation
            mgener = ModelCodeGen(sugraph)
            sgener = ScheduleCodeGen(sugraph)
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
            data = None
            for su in sugraph.sus():
                if su.stype == SUType.Dataloader:
                    data = su.outputs(0)
                    break
            if data is None:
                raise RuntimeError("dataloader not found in SUGraph")
            # assume batch_size is always first dimension
            batch_size = torch.tensor([data.shape[0]], dtype=torch.int).cuda()

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
