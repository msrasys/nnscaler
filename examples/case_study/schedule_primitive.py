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


def pipeline_1f1b_schedule(forward, backward, update, num_stages=2, num_microbs=4):
    """
    f: forward function
    b: backward function

    Suppose model is partitioned to `num_stages` with input `num_microbs` micro-batches
    """
    # suppose we have 2 stages using 1f1b with num micro-batches=4

    # f[stage_id, data_id]
    partial_sequences = []
    for data_id in range(num_microbs):
        one_mbatch = PartialSequence()
        for stage_id in range(num_stages):
            one_mbatch.append(Action(forward))
        for stage_id in range(num_stages):
            one_mbatch.append(Action(backward))
            if data_id == num_microbs - 1:
                one_mbatch.append(Action(update))
        partial_sequences.append(one_mbatch)
    for S in range(num_stages):
        seq = PartialSequence([partial_sequences[-1-S][-num_stages-1]], Action(update))
        partial_sequences.append(seq)


    # Action f[stage, micro-batch]
    # f[S, D]: forward on stage S for micro-batch id D
    f = [partial_sequences[i][:num_stages] for i in range(num_microbs)]

    # Action b[stage, micro-batch]
    # b[S, D]: backward on stage S for micro-batch id D
    b = [partial_sequences[i][num_stages:] for i in range(num_microbs)]

    # Action u[stage, micro-batch]
    # u[S, D]: update weight on stage S
    u = [partial_sequences[i+num_microbs][1] for i in range(num_stages)]


    # =========================
    # !@#$#%$&^$# -- policy generated a legal global execution order
    global_schedule = [
        f[0,0], f[1,0], b[1,0],
        f[0,1], b[0,0], f[1,1], b[1,1],
        f[0,2], b[0,1], f[1,2], b[0,2],
        f[0,3], b[0,2], f[1,3], b[1,3], u[1],
        u[0]
    ]
    # =========================

    return global_schedule


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
