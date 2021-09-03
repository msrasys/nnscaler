import torch

from functools import partial


def select(tensor, indices, val_map_op=None, shape=None):
    pass

## Abstractions and Primitivse ##

class Action: pass

def execute(action, **kwargs):
    # action instance will automatically take flow-in results
    # and select the chunked kwargs
    return action(**kwargs)

def add_flow(*actions):
    # this will set all input actions with same flow-id
    pass


## System Runtime units ##

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

def check_consistency(sequence, actions, relations): pass


# Schedule example

def naive_schedule(actions: list(Action), relations: set((Action, Action))) -> list(Action):
    """
    Args:
        actions: order specified by AI scientist (the reference semantic)
        relations: set of action dependencies (action1, action2): action1 -> action2

    Returns:
        a execution sequence following the abstraction
    """
    # placement
    for action in actions:
        action.device = 0
    # execution sequence
    sequence = actions
    return sequence


def pipeline_1f1b_schedule(actions, relations):
    """
    Pipeline 1f1b policy description -- generate a sequence

    Actions: a list of actions

    relations: list[(Action1, Action2)]: a list of tuples indicate partial order
    """

    # suppose input actions are forward and backward of grad accumulation
    # suppose in forward -> ... -> forward -> backward -> ... -> backward
    num_stage = torch.distributed.get_world_size()
    num_microbatch = len(actions) / 2 / num_stage

    f = lambda stage, micro_batch_id: actions[2 * micro_batch_id * num_stage + stage]
    b = lambda stage, micro_batch_id: actions[(2 * micro_batch_id + 1) * num_stage + num_stage - 1 - stage]

    # action placement
    for stage in range(num_stage):
        for mid in range(num_microbatch):
            f(stage, mid).device = torch.device.cuda(stage)
            b(stage, mid).device = torch.device.cuda(stage)

    sequence = list()

    # warmup:
    for stage in range(num_stage):
        for mid in range(stage):
            sequence.append(f(stage, mid))
    
    # steady + cooldown:
    for mid in range(num_microbatch):
        # enqueue backward
        for stage in range(num_stage-1, -1, -1):
            sequence.append(b(stage, mid))
        # enqueue forward
        for stage in range(num_stage):
            f_mid = mid + 1 + num_stage - stage
            if f_mid >= num_microbatch:
                continue
            sequence.append(f(stage, f_mid))
    assert check_consistency(sequence, actions, relations)
    return sequence


def pipeline_1f1b_schedule(actions, relations):
    """
    Pipeline 1f1b policy description -- each device order

    Actions: a list of actions

    relations: list[(Action1, Action2)]: a list of tuples indicate partial order
    """
    num_stage = torch.distributed.get_world_size()
    num_microbatch = len(actions) / 2 / num_stage

    f = lambda stage, micro_batch_id: actions[2 * micro_batch_id * num_stage + stage]
    b = lambda stage, micro_batch_id: actions[(2 * micro_batch_id + 1) * num_stage + num_stage - 1 - stage]

    # action placement
    for stage in range(num_stage):
        for mid in range(num_microbatch):
            f(stage, mid).device = torch.device.cuda(stage)
            b(stage, mid).device = torch.device.cuda(stage)

    # action in-device order
    stage_order = list()

    for stage in range(num_stage):
        order = list()
        num_warmup_microbatch = num_stage - stage - 1
        num_warmup_microbatch = min(num_warmup_microbatch, num_microbatch)
        num_microbatch_remain = num_microbatch - num_warmup_microbatch

        # warmup
        for mid in range(num_warmup_microbatch):
            order.append(f(stage, mid))
        
        # steady
        for i in range(num_microbatch_remain):
            f_mid = num_warmup_microbatch + i
            b_mid = i
            order.append(f(stage, f_mid))
            order.append(b(stage, b_mid))
        
        # cooldown
        for i in range(num_warmup_microbatch):
            b_mid = num_microbatch_remain + i
            order.append(b(stage, b_mid))
        
        stage_order.append(order)

    assert check_consistency(stage_order, actions, relations)
    return stage_order



if __name__ == '__main__':

    # define logical model / optimizer / data loader
    class LogicalModel: pass
    class Optimizer: pass
    class DataLoader: pass
    compute_loss = lambda output, label : output


    model = LogicalModel()
    optimizer = Optimizer(model.parameters())
    dataloader = DataLoader(bs=1024)

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




# ======== example sequences for all kinds of configuration =============

forward = lambda model, data: model(data)
backward = lambda grad, output: output.backward(grad)
update_gradient = lambda model, grad: model.update(grad)


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
