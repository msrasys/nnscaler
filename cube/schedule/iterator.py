from cube.schedule.action import Action
from cube.schedule.checker import correct_check

import itertools


def legal_sequence(actions, relations):
    """
    Yield all possible legal sequence given the list of actions

    Args:
        actions (list[Actions])
    
    Yield:
        sequence (list[Actions])
    """
    if not all([isinstance(action, Action) for action in actions]):
        raise TypeError("Expected the sequence to be list[Action]")

    for seq in itertools.permutations(actions):
        seq = list(seq)
        if correct_check(seq, actions, relations):
            yield seq


def ready_action_set(actions, relations, flip=False):
    """
    Return a list of actions can be executed now
    """
    flip = -1 if flip else 1
    ready_actions = list()
    for action in actions[::flip]:
        satisfy = True
        for (_, succ) in relations:
            if succ == action:
                satisfy = False
                break
        if satisfy:
            ready_actions.append(action)
    return ready_actions


def remove_dependency(action, relations):
    new_relations = list()
    for (pre, succ) in relations:
        # remove dependency
        if pre == action:
            continue
        new_relations.append((pre, succ))
    return new_relations


def sequence_space(actions, relations, seq=list()):
    if len(actions) == 0:
        yield seq
    # inital entry
    entry_actions = ready_action_set(actions, relations, flip=len(actions) % 2 == 0)
    for action in entry_actions:
        seq = seq + [action]
        action_idx = actions.index(action)
        sub_actions = actions[:action_idx] + actions[action_idx+1:]
        sub_relations = remove_dependency(action, relations)
        for res in sequence_space(sub_actions, sub_relations, seq):
            yield res
        seq = seq[:-1]
