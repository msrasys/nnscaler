from typing import Sequence
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
    myrank = torch.distributed.get_rank()
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
        if action.device == myrank:
            # wait for cross-device dependency (if have)
            action.wait()
            # execute
            outs = execute(action, *tuple(args))
    return outs

class Action: pass

def check_consistency(sequence, actions, relations): pass


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


def pipeline_1f1b_schedules(actions, relations):
    """
    Pipeline 1f1b policy description

    Actions: a list of actions

    relations: list[(Action1, Action2)]: a list of tuples indicate partial order
    """

    # suppose input actions are forward and backward of grad accumulation
    # suppose in forward -> ... -> forward -> backward -> ... -> backward
    num_stage = torch.distributed.get_world_size()
    num_micro_batch = len(actions) / 2 / num_stage

    f = lambda stage, micro_batch_id: actions[2 * micro_batch_id * num_stage + stage]
    b = lambda stage, micro_batch_id: actions[(2 * micro_batch_id + 1) * num_stage + stage]

    sequence = list()

    # warmup:
    for stage in range(num_stage):
        for mid in range(stage):
            sequence.append(f(stage, mid))
    
    # steady + cooldown:
    for mid in range(num_micro_batch):
        # enqueue backward
        for stage in range(num_stage-1, -1, -1):
            sequence.append(b(stage, mid))
        # enqueue forward
        for stage in range(num_stage):
            f_mid = mid + 1 + num_stage - stage
            if f_mid >= num_micro_batch:
                continue
            sequence.append(f(stage, f_mid))
    assert check_consistency(sequence, actions, relations)
    return sequence


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


def train_iter_grad_accumulate(model, datas, stage=2, micro_bs=4):

    out_s0_d0 = forward(model[0], datas[0])
    out_s1_d0 = forward(model[1], out_s0_d0)
    grad_s1_d0 = backward(out_s1_d0)
    grad_s0_d0 = backward(out_s0_d0, grad=grad_s1_d0)

    out_s0_d1 = forward(model[0], datas[1])
    out_s1_d1 = forward(model[1], out_s0_d1)
    grad_s1_d1 = backward(out_s1_d1)
    grad_s0_d1 = backward(out_s0_d0, grad=grad_s1_d1)

    out_s0_d2 = forward(model[0], datas[2])
    out_s1_d2 = forward(model[1], out_s0_d2)
    grad_s1_d2 = backward(out_s1_d2)
    grad_s0_d2 = backward(out_s0_d0, grad=grad_s1_d2)

    out_s0_d3 = forward(model[0], datas[3])
    out_s1_d3 = forward(model[1], out_s0_d3)
    grad_s1_d3 = backward(out_s1_d3)
    grad_s0_d3 = backward(out_s0_d0, grad=grad_s1_d3)

    update_gradient(model[0], model[0].weights.grad)
    update_gradient(model[1], model[1].weights.grad)


def train_iter_1f1b(model, datas, stage=2, micro_bs=4):

    out_s0_d0 = forward(model[0], datas[0])
    out_s1_d0 = forward(model[1], out_s0_d0)
    grad_s1_d0 = backward(out_s1_d0)

    out_s0_d1 = forward(model[0], datas[1])
    grad_s0_d0 = backward(out_s0_d0, grads=grad_s1_d0)
    out_s1_d1 = forward(model[1], out_s0_d1)
    grad_s1_d1 = backward(out_s1_d1)

    out_s0_d2 = forward(model[0], datas[2])
    grad_s0_d1 = backward(out_s0_d0, grad=grad_s1_d1)
    out_s1_d2 = forward(model[1], out_s0_d2)
    grad_s1_d2 = backward(out_s1_d2)

    out_s0_d3 = forward(model[0], datas[3])
    grad_s0_d2 = backward(out_s0_d0, grad=grad_s1_d2)
    out_s1_d3 = forward(model[1], out_s0_d3)
    grad_s1_d3 = backward(out_s1_d3)
    update_gradient(model[1], model[1].weights.grad)

    grad_s0_d3 = backward(out_s0_d0, grad=grad_s1_d3)
    update_gradient(model[0], model[0].weights.grad)


def train_iter_gpipe(model, datas, stage=2, micro_bs=4):

    out_s0_d0 = forward(model[0], datas[0])
    out_s1_d0 = forward(model[1], out_s0_d0)
    out_s0_d1 = forward(model[0], datas[1])
    out_s1_d1 = forward(model[1], out_s0_d1)
    out_s0_d2 = forward(model[0], datas[2])
    out_s1_d2 = forward(model[1], out_s0_d2)
    out_s0_d3 = forward(model[0], datas[3])
    out_s1_d3 = forward(model[1], out_s0_d3)

    grad_s1_d0 = backward(out_s1_d0)
    grad_s0_d0 = backward(out_s0_d0, grad=grad_s1_d0)
    grad_s1_d1 = backward(out_s1_d1)
    grad_s0_d1 = backward(out_s0_d0, grad=grad_s1_d1)
    grad_s1_d2 = backward(out_s1_d2)
    grad_s0_d2 = backward(out_s0_d0, grad=grad_s1_d2)
    grad_s1_d3 = backward(out_s1_d3)
    update_gradient(model[1], model[1].weights.grad)
    grad_s0_d3 = backward(out_s0_d0, grad=grad_s1_d3)
    update_gradient(model[0], model[0].weights.grad)
