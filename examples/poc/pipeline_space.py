from typing import Sequence
from cube.schedule.action import Action, add_flow
from cube.schedule.iterator import sequence_space
from cube.schedule.plan import ExecutionPlan

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


def fixed_placement_sequence(actions, relations, ndevice, forward, backward):
    for cnt, seq in enumerate(sequence_space(actions, relations)):
        # assign device
        for action in seq:
            stage, mid = get_stage_and_mid(action)
            action.device = stage % ndevice
        execplan = ExecutionPlan(seq, ndevice)
        execplan.gen()
        iter_time = execplan.get_time()
        print(f'found seq > {seq} \t time {iter_time}')
        # if iter_time > 28:
        #     continue
        execplan.draw(outfile='tmp.png')
        input('>>> ')
    print('total found {} legal sequences'.format(cnt + 1))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--nstage', type=int, default=4,
                        help='number of stages')
    parser.add_argument('--nmb', type=int, default=4,
                        help='number of micro-batch')
    parser.add_argument('--ndev', type=int, default=4,
                        help='number of devices')
    args = parser.parse_args()
    
    forward = lambda data: data
    backward = lambda grad: grad

    actions, relations = get_semantic(forward, backward, args.nstage, args.nmb)

    #print_all_legal_sequence(actions, relations)
    fixed_placement_sequence(actions, relations, args.ndev, forward, backward)
