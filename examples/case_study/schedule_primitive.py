import torch

from functools import partial

## Primitive ##

def select(tensor, indices, val_map_op=None):
    pass

def execute(action, *args, **kwargs):
    return action(*args, **kwargs)

def add_flow(*actions): pass

def run(schedule):
    """
    Take a list of actions and execute in list order
    """
    for action in schedule:
        outs = execute(action)
    return outs

class Action: pass


# ===================== Basic steps ================== #
def general_action(flow_in, *args, **kwargs):
    """
    flow_in: the output from previous actions
    """
    pass

def forward(flow_in, model, data): pass

def backward(flow_in): pass

def update(flow_in, optimizer): pass
# ===================== Basic steps ================== #


def naive_schedule(model, data, optimizer):

    f = Action(partial(forward, model=model, data=data))
    b = Action(partial(backward))
    u = Action(partial(update, optimizer=optimizer))
    
    add_flow(f, b ,u)

    schedules = [f, b, u]
    
    return schedules


def pipeline_schedule(model, data, optimizer, num_microbatches=4):

    # forward, backward, update function
    f = partial(forward, model=model)
    b = partial(backward)
    u = partial(update, optimizer=optimizer)

    # suppose we have 4 devices using 1f1b with num micro-batches=4
    chunk_size = data.size(0) / 4
    data = [
        select(data, slice(chunk_size * 0, chunk_size * 1)),
        select(data, slice(chunk_size * 1, chunk_size * 2)),
        select(data, slice(chunk_size * 2, chunk_size * 3)),
        select(data, slice(chunk_size * 3, chunk_size * 4))
    ]

    f0 = Action(partial(f, data=data[0]))
    f1 = Action(partial(f, data=data[1]))
    f2 = Action(partial(f, data=data[2]))
    f3 = Action(partial(f, data=data[3]))

    b0 = Action(b)
    b1 = Action(b)
    b2 = Action(b)
    b3 = Action(b)

    u = Action(u)

    # add data flow f0 -> b0 -> u
    add_flow(f0, b0, u)
    add_flow(f1, b1, u)
    add_flow(f2, b2, u)
    add_flow(f3, b3, u)

    
    global_schedule = [
        [f0, f1, f2, f3, b0, b1, b2, b3, u],  # rank 0
        [f0, f1, f2, b0, f3, b1, b2, b3, u],  # rank 1
        [f0, f1, b0, b2, f1, f3, b2, b3, u],  # rank 2
        [f0, b0, f1, b1, f2, b2, f3, b3, u],  # rank 3
    ]

    # schedules will be in dead lock
    [
        [f0, b0, f1, b1],
        [f0, f1, b0, b1],
    ]

    return global_schedule[torch.distributed.get_rank()]