from cube.schedule.action import Action
from cube.schedule.iterator import legal_sequence

import argparse

from examples.case_study.schedule_primitive import grad_accumulate


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
                relations.append(relation)
            actions.append(action)
        # backward
        for stage in range(num_stage):
            action = Action(backward_fn)
            action.tag('b(S{},D{})'.format(num_stage - 1 - stage, mid))
            relation = (actions[-1], action)
            relations.append(relation)
            actions.append(action)
    return actions, relations


def print_all_legal_sequence(actions, relations):
    for cnt, seq in enumerate(legal_sequence(actions, relations)):
        print(seq)
    print('total found {} legal sequences'.format(cnt + 1))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--nstage', type=int, default=4,
                        help='number of stages')
    parser.add_argument('--nmb', type=int, default=4,
                        help='number of micro-batch')
    args = parser.parse_args()
    
    forward = lambda data: data
    backward = lambda grad: grad

    actions, relations = get_semantic(forward, backward, args.nstage, args.nmb)

    print_all_legal_sequence(actions, relations)
