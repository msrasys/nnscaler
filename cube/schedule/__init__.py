from typing import Callable, Optional
import torch

from cube.schedule.pool import SchedulePool
from cube.schedule.translator import IRDataLoader, LogicTranslator
from cube.schedule.sugraph import SUGraph
from cube.codegen.codegen import SScheduleCodeGen, TScheduleCodeGen


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


def schedule(model: SemanticModel, dataloader, policy_fn: Optional[Callable] = None):
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

        if myrank == 0:
            SchedulePool().clear()

            # logic translator
            fn(ir_graph, ir_dataloader)
            sus = SchedulePool().sus()

            # adapter
            sus_with_adapter = LogicTranslator.gen_adapter(sus)

            # policy
            su_graph = SUGraph(sus_with_adapter)
            if policy_fn:
                # TODO: add resource
                su_graph = policy_fn(su_graph, None)

            # check assignment and order
            for su in su_graph.sus():
                if len(su.device) == 0:
                    raise RuntimeError(f"SU {su} device is not set")
            if not SUGraph.is_topo_order(su_graph.sus()):
                raise RuntimeError(f"SUGraph order is not topological order")

            if torch.distributed.is_initialized():
                world_size = torch.distributed.get_world_size()
            else:
                world_size = 1

            # code generation
            tgener = TScheduleCodeGen(su_graph)
            sgener = SScheduleCodeGen(su_graph)
            for rank in range(world_size):
                fname = filename.format(rank)
                # generate spatial module code
                sgener.gen(rank, outfile=fname, attach=True)
                # generate temporal schedule code
                tgener.gen(
                    device = rank,
                    outfile = fname,
                    attach=True
                )
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        # load module
        model.load_module(filename.format(myrank))
        # load temporal
        return _load_tschedule_fn(filename.format(myrank))
    
    return decorator
