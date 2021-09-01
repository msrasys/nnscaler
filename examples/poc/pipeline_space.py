from typing import Sequence
from cube.schedule.action import Action
from cube.schedule.iterator import legal_sequence

import argparse
import re

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def get_semantic(forward_fn, backward_fn, num_stage, num_microbatch):
    actions = list()
    relations = list()
    for mid in range(num_microbatch):
        # forward
        for stage in range(num_stage):
            action = Action(forward_fn)
            action.tag('f(S{},D{})'.format(stage, mid))
            if stage != 0:
                relation = (actions[-1], action)
                action.depends_on(actions[-1])
                relations.append(relation)
            actions.append(action)
        # backward
        for stage in range(num_stage):
            action = Action(backward_fn)
            action.tag('b(S{},D{})'.format(num_stage - 1 - stage, mid))
            # relation
            relation = (actions[-1], action)
            action.depends_on(actions[-1])
            # append to relation sets
            relations.append(relation)
            actions.append(action)
    return actions, relations


def get_stage_and_mid(action):
    ids = re.findall(r"S(\d+),D(\d+)", action.name)
    stage, mid = int(ids[0][0]), int(ids[0][1])
    return stage, mid


def device_search(sequence):
    pass


def draw_execution_plan(seq, forward_fn, backward_fn, ndevice):
    forward_time = 1
    backward_time = 2
    # record each action end time
    current_time = [[1] for _ in range(ndevice)]
    device_actions = [list() for _ in range(ndevice)]

    recs = dict()

    for action in seq:
        if action.device == -1 or action.device >= ndevice:
            raise RuntimeError("action {} device not assigned or out of boundary".format(action))
        start_time = current_time[action.device][-1]
        for dev_id, (end_times, dev_actions) in enumerate(zip(current_time, device_actions)):
            if dev_id == action.device:
                continue
            # go through to check if the action has dependencies
            for end_time, dev_action in zip(end_times, dev_actions):
                print(dev_action)
                if action.depends_on(dev_action):
                    print('find dependency {} -> {}, end time: {}'.format(action, dev_action, end_time))
                    start_time = max(start_time, end_time)
                elif dev_action.depends_on(action):
                    raise RuntimeError("Action happened before")
        # draw regtangular
        if action._fn[0] == forward_fn:
            span_time = forward_time
            color = 'blue'
        elif action._fn[0] == backward_fn:
            span_time = backward_time
            color = 'orange'
        # stage, mid = get_stage_and_mid(action)
        recs[action.name] = Rectangle((start_time, action.device), span_time, 1,
                                      color=color, ec='black', lw=1.5)
        # update timeline
        current_time[action.device].append(start_time + span_time)
        device_actions[action.device].append(action)
    
    fig, ax = plt.subplots()
    for r in recs:
        ax.add_artist(recs[r])
        rx, ry = recs[r].get_xy()
        cx = rx + recs[r].get_width() / 2.0
        cy = ry + recs[r].get_height() / 2.0
        ax.annotate(r, (cx, cy), color='w', weight='bold',
                    fontsize=8, ha='center', va='center')
    
    ax.set_xlim((1, len(seq) + 1))
    ax.set_ylim((0, ndevice))
    ax.set_aspect('equal')
    plt.savefig('./tmp.png')


def print_all_legal_sequence(actions, relations):
    for cnt, seq in enumerate(legal_sequence(actions, relations)):
        print(seq)
    print('total found {} legal sequences'.format(cnt + 1))


def fixed_placement_sequence(actions, relations, ndevices, forward, backward):
    for cnt, seq in enumerate(legal_sequence(actions, relations)):
        # assign device
        for action in seq:
            stage, mid = get_stage_and_mid(action)
            action.device = stage % ndevices
        draw_execution_plan(seq, forward, backward, ndevices)
        break
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

    fixed_placement_sequence(actions, relations, args.ndev, forward, backward)
