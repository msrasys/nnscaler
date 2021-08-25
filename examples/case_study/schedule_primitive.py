import torch

from functools import partial

## Primitive ##

def select(tensor, indices, val_map_op=None, shape=None):
    pass

def execute(action, **kwargs):
    # action instance will automatically take flow-in results
    # and select the chunked kwargs
    return action(**kwargs)

def add_flow(*actions):
    # this will set all input actions with same flow-id
    pass

def run(schedule, num_microbs, *args):
    """
    Take a list of actions and execute in list order
    """
    chunked_args = list()
    for arg in args:
        if torch.is_tensor(arg):
            chunk_size = data.size(0) / num_microbs
            arg = [
                select(arg, slice(chunk_size * 0, chunk_size * 1)),
                select(arg, slice(chunk_size * 1, chunk_size * 2)),
                select(arg, slice(chunk_size * 2, chunk_size * 3)),
                select(arg, slice(chunk_size * 3, chunk_size * 4))
            ]
        chunked_args.append(arg)
    for action in schedule:
        outs = execute(action, *tuple(args))
    return outs

class Action: pass


# ===================== Basic steps ================== #
def general_action(flow_in, *args, **kwargs):
    """
    flow_in: the output from previous actions
    """
    pass

def forward(flow_in, model, data):
    loss = model(data)
    return loss

def backward(flow_in):
    flow_in.backwrd()
    return flow_in

# ===================== Basic steps ================== #


def naive_schedule(f, b):
    f = Action(f)
    b = Action(b)
    add_flow(f, b)
    schedules = [f, b]
    return partial(run, schedules, num_microbs=1)


def grad_accumulate(f, b, accum_times=4):
    forwards = [Action(f, fid=fid) for fid in range(accum_times)]
    backwards = [Action(b, fid=fid) for fid in range(accum_times)]
    schedules = list()
    for f, b in zip(forwards, backwards):
        add_flow(f, b)
        schedules += [f, b]
    return partial(run, schedules, num_microbs=accum_times)


def pipeline_schedule(f, b, num_microbs=4):
    """
    f: forward function
    b: backward function
    """
    # suppose we have 4 devices using 1f1b with num micro-batches=4

    f0 = Action(partial(f), fid=0)
    f1 = Action(partial(f), fid=1)
    f2 = Action(partial(f), fid=2)
    f3 = Action(partial(f), fid=3)

    b0 = Action(b)
    b1 = Action(b)
    b2 = Action(b)
    b3 = Action(b)

    # add data flow f0 -> b0
    add_flow(f0, b0)
    add_flow(f1, b1)
    add_flow(f2, b2)
    add_flow(f3, b3)

    
    global_schedule = [
        [f0, f1, f2, f3, b0, b1, b2, b3],  # rank 0
        [f0, f1, f2, b0, f3, b1, b2, b3],  # rank 1
        [f0, f1, b0, b2, f1, f3, b2, b3],  # rank 2
        [f0, b0, f1, b1, f2, b2, f3, b3],  # rank 3
    ]

    myschedule = global_schedule[torch.distributed.get_rank()]

    # schedules will be in dead lock
    # [
    #     [f0, b0, f1, b1],
    #     [f0, f1, b0, b1],
    # ]

    return partial(run, myschedule, num_microbs=num_microbs)


def dist_policy(DAG, resources):
    """
    Policy decided the parallelisms and op-placement
    """    
    return DAG


def set_schedule_policy(model, specific_schedule, bs):
    """
    forward_fn: forward function
    backward_fn: backward_function
    bs: global batch size
    """
    num_microbs = 4 if bs >= 4 else bs
    schedule = pipeline_schedule(model.forward, backward, num_microbs)
    model.set_schedule(schedule)


if __name__ == '__main__':

    # define logical model / optimizer / data loader
    class LogicalModel: pass
    class Optimizer: pass
    class DataLoader: pass


    model = LogicalModel()
    optimizer = Optimizer(model.parameters())
    dataloader = DataLoader(bs=1024)

    # def forward_step(flow_in, data, label, **kwargs):
    #     # this requires loss computation needs to be in the model
    #     # output = model(data, label)
    #     output = model(data, label)
    #     # function wrapper
    #     loss = compute_loss(output)
    #     return output
    # 
    # def backward_step(output, **kwargs):
    #     output.backward()
    #     return output

    # policy for placement and parallelisms -- will be hidden
    model = dist_policy(get_dag(model, loss_compute, input_shapes), resources)
    # data flow scheduling policy -- will be hidden
    set_schedule_policy(model, pipeline_schedule, bs=1024)

    for epoch in range(100):
        for step, (data, label) in enumerate(dataloader):
            # enqueue forward specfied by schedule and execute the first one
            output = model(data)
            # accessing partial output data without generation will rase warning
            # pop forward until to generate the backward tensor
            loss = compute_loss(output, label)
            loss.backward()

            # loss = schedule(data=data)
            optimizer.step()
            # lr_scheduler.step()
            optimizer.zero_grad()
            print(loss)

        if (epoch + 1) % 4 == 0:
            model.eval()
            # evaluation


# class Model:
# 
#     def forward(self, data):
#         # non-torch op wrapper
# 
#         if data[0] > 1:
#             self.net1(data)
#         else:
#             self.net2(data)
# 
#     def forward_(self, data)
# 
#         self.q = [(self.forward, data[0]), (xxx)]