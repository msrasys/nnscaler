from typing import Sequence
from cube.schedule.action import Action, add_flow
from cube.schedule.iterator import sequence_space
from cube.schedule.plan import ExecutionPlan
from cube.schedule.checker import correct_check

import argparse
import re

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def get_semantic(forward_fn, backward_fn, num_stage, num_microbatch):
    forward_time = 1
    backward_time = 2

    actions = list()
    relations = list()
    for mid in range(num_microbatch):
        # forward
        for stage in range(num_stage):
            action = Action(forward_fn)
            action.est_latency = forward_time
            action.tag('fS{}D{}'.format(stage, mid))
            if stage != 0:
                relation = (actions[-1], action)
                add_flow(actions[-1], action)
                relations.append(relation)
            else:
                action.fid = mid
            actions.append(action)
        # backward
        for stage in range(num_stage):
            action = Action(backward_fn)
            action.est_latency = backward_time
            action.tag('bS{}D{}'.format(num_stage - 1 - stage, mid))
            # relation
            relation = (actions[-1], action)
            add_flow(actions[-1], action)
            # append to relation sets
            relations.append(relation)
            actions.append(action)
    return actions, relations


def get_stage_and_mid(action):
    ids = re.findall(r"S(\d+)D(\d+)", action.name)
    stage, mid = int(ids[0][0]), int(ids[0][1])
    return stage, mid


def device_search(sequence):
    pass


def print_all_legal_sequence(actions, relations):
    for cnt, seq in enumerate(sequence_space(actions, relations)):
        print(seq)
    print('total found {} legal sequences'.format(cnt + 1))


def fixed_placement_sequence(actions, relations, ndevice, max_time):
    for cnt, seq in enumerate(sequence_space(actions, relations)):
        # assign device
        for action in seq:
            stage, mid = get_stage_and_mid(action)
            action.device = stage % ndevice
        execplan = ExecutionPlan(seq, ndevice)
        execplan.gen()
        iter_time = execplan.get_time()
        print(f'found seq > {seq} \t time {iter_time}')
        if iter_time > max_time:
            continue
        execplan.draw(outfile='tmp.png')
        input('>>> ')
    print('total found {} legal sequences'.format(cnt + 1))


def pipe_1f1b(actions, relations, nstage, ndevice, nmb):
    num_stage = nstage
    num_microbatch = nmb

    f = lambda stage, micro_batch_id: actions[2 * micro_batch_id * num_stage + stage]
    b = lambda stage, micro_batch_id: actions[(2 * micro_batch_id + 1) * num_stage + num_stage - 1 - stage]

    # action placement
    for stage in range(num_stage):
        for mid in range(num_microbatch):
            f(stage, mid).device = stage % ndevice
            print(f(stage, mid), f'stage={stage}, mid={mid}, device={stage % ndevice}')
            b(stage, mid).device = stage % ndevice
            print(b(stage, mid), f'stage={stage}, mid={mid}')

    sequence = list()

    # warmup:
    for stage in range(num_stage):
        for mid in range(num_stage-stage):
            sequence.append(f(stage, mid))

    # steady + cooldown:
    for mid in range(num_microbatch):
        # enqueue backward
        for stage in range(num_stage-1, -1, -1):
            sequence.append(b(stage, mid))
        # enqueue forward
        for stage in range(num_stage):
            f_mid = mid + num_stage - stage
            if f_mid >= num_microbatch:
                continue
            sequence.append(f(stage, f_mid))
    print(sequence)
    assert correct_check(sequence, actions, relations)
    execplan = ExecutionPlan(sequence, ndevice)
    execplan.draw(outfile='./pipeline-1f1b.png')


def gpipe(actions, relations, nstage, ndevice, nmb):
    num_stage = nstage
    num_microbatch = nmb

    f = lambda stage, micro_batch_id: actions[2 * micro_batch_id * num_stage + stage]
    b = lambda stage, micro_batch_id: actions[(2 * micro_batch_id + 1) * num_stage + num_stage - 1 - stage]

    # action placement
    for stage in range(num_stage):
        for mid in range(num_microbatch):
            f(stage, mid).device = stage % ndevice
            print(f(stage, mid), f'stage={stage}, mid={mid}, device={stage % ndevice}')
            b(stage, mid).device = stage % ndevice
            print(b(stage, mid), f'stage={stage}, mid={mid}')

    sequence = list()

    # warmup:
    for stage in range(num_stage):
        for mid in range(num_microbatch):
            sequence.append(f(stage, mid))
    
    # backward
    for stage in range(num_stage):
        for mid in range(num_microbatch):
            sequence.append(b(num_stage - 1 - stage, mid))

    print(sequence)
    # assert correct_check(sequence, actions, relations)
    execplan = ExecutionPlan(sequence, ndevice)
    execplan.draw(outfile='./gpipe.png')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--nstage', type=int, default=4,
                        help='number of stages')
    parser.add_argument('--nmb', type=int, default=4,
                        help='number of micro-batch')
    parser.add_argument('--ndev', type=int, default=4,
                        help='number of devices')
    parser.add_argument('--max-time', type=int, default=100,
                        help='maximal time. Will filter out plans that have larger time than this')
    args = parser.parse_args()
    
    forward = lambda data: data
    backward = lambda grad: grad

    actions, relations = get_semantic(forward, backward, args.nstage, args.nmb)

    pipe_1f1b(actions, relations, args.nstage, args.ndev, args.nmb)
    gpipe(actions, relations, args.nstage, args.ndev, args.nmb)

    fixed_placement_sequence(actions, relations, args.ndev, args.max_time)
