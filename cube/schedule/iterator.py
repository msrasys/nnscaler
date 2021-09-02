from cube.schedule.action import Action
from cube.schedule.checker import correct_check

import itertools
import numpy as np


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


def ready_action_set(actions, relations):
    """
    Return a list of actions can be executed now
    """
    ready_actions = list()
    for action in actions:
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


def sequence_space(actions, relations, path_shuffle=True, seq=list()):
    if len(actions) == 0:
        yield seq
    # inital entry
    entry_actions = ready_action_set(actions, relations)
    entry_actions = np.array(entry_actions)
    if path_shuffle:
        np.random.shuffle(entry_actions)
    for aid, action in enumerate(entry_actions):
        if len(seq) == 0:
            print(f'> search progress: [{aid}/{len(entry_actions)}]...')
        seq = seq + [action]
        action_idx = actions.index(action)
        sub_actions = actions[:action_idx] + actions[action_idx+1:]
        sub_relations = remove_dependency(action, relations)
        for res in sequence_space(sub_actions, sub_relations, path_shuffle, seq):
            yield res
        seq = seq[:-1]


def sequence_space_batched(actions, relations, bs):
    """
    bs: tuple (num_workers, seq_per_worker)
    """
    seqs = list()
    for seq in sequence_space(actions, relations):
        seqs.append(seq)
        if len(seqs) % (bs[0] * bs[1]) == 0:
            seqs = [seqs[wid*bs[1]:(wid+1)*bs[1]] for wid in range(bs[0])]
            yield seqs
            seqs = list()
    # tail
    if len(seqs) != 0:
        seqs = [seqs[wid*bs[1]:(wid+1)*bs[1]] for wid in range(bs[0])]
        yield seqs


def placement_space(actions, ndevice, fb_same=True, path_shuffle=True, assigned=0):
    if assigned == len(actions):
        yield actions
        return

    action = actions[assigned]
    device_choice = np.array(list(range(ndevice)), dtype=np.int)
    if path_shuffle:
        np.random.shuffle(device_choice)

    if fb_same:
        for assigned_action in actions[:assigned]:
            # assume action name likes 'fS0D1'
            if action.name[1:] == assigned_action.name[1:]:
                device_choice = [assigned_action.device]
                break
    for device in device_choice:
        action.device = device
        for res in placement_space(actions, ndevice, fb_same, path_shuffle, assigned+1):
            yield res
