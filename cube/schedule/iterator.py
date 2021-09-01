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
