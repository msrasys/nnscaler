import torch
from cube.profiler.timer import print_each_rank

def memory_summary():
    rank = torch.distributed.get_rank()
    # memory measurement
    mem = torch.cuda.max_memory_allocated()
    # mem = torch.cuda.max_memory_reserved()
    print_each_rank(
        '{:.2f}GB memory consumption'.format(mem / 1024 / 1024 / 1024),
    )
