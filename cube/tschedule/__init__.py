from typing import Callable, Optional
import torch

from cube.tschedule.pool import TSchedulePool
from cube.graph.ir_cten import IRTensor
from cube.tschedule.suseq import SUSequence
from cube.codegen.codegen import TScheduleCodeGen


class IRTesnorDataLoader:

    def __init__(self, dataloader):
        self.dataloader = dataloader

    def __iter__(self):
        return self

    def __next__(self):
        datas = next(self.dataloader)
        ir_datas = list()
        for data in datas:
            if torch.is_tensor(data):
                tensor = IRTensor(shape=list(data.size()), name='input')
            else:
                tensor = data
            ir_datas.append(tensor)
        return tuple(ir_datas)


def schedule(model, dataloader, policy_fn: Optional[Callable] = None):
    """
    AI Scientist calls like:

        @cube.tschedule.schedule
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
    ir_graph = model.get_graph()
    ir_dataloader = IRTesnorDataLoader(dataloader)
    myrank = torch.distributed.get_rank()

    def _load_tschedule_fn(filename) -> Callable:
        print(f'> [{myrank}] loading generated schedule from {filename} ...')
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_train_step", filename
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module._train_step

    def decorator(fn: Callable) -> Callable:
        filename = 'gencode{}.py'
        if myrank == 0:
            TSchedulePool().clear()
            # collect trace
            fn(ir_graph, ir_dataloader)
            sus = TSchedulePool().sus()
            seq = SUSequence(sus)
            # policy
            if policy_fn:
                seq = policy_fn(seq)

            world_size = torch.distributed.get_world_size()
            tgener = TScheduleCodeGen(seq)
            for rank in range(world_size):
                fname = filename.format(rank)
                # generate spatial module code
                model.gen_module(seq, rank, fname, attach=False)
                # generate temporal schedule code
                tgener.gen(
                    device = rank,
                    outfile = fname,
                    attach=True
                )
        torch.distributed.barrier()
        # load module
        model.load_module(filename.format(myrank))
        # load temporal
        return _load_tschedule_fn(filename.format(myrank))
    
    return decorator
