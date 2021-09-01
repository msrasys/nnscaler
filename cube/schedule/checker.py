from cube.schedule.action import Action


def correct_check(sequence, actions, relations):
    """
    Check if sequence satisfies the sequential consistency model
    Args:
        sequence (list[Actions]): action sequence
        actions (list[Action]): action lists
        relations (list(tuple(Action, Action))):
            contains happened before tuple list
    Returns:
        Boolean: whether satisfies the partial order specified in relations
    """
    if not all([isinstance(action, Action) for action in sequence]):
        raise TypeError("Expected the sequence to be list[Action]")
    if not all([isinstance(action, Action) for action in actions]):
        raise TypeError("Expected the actions to be list[Action]")
    
    # check if all Actions in `actions` are used by sequence
    if set(sequence) != set(actions):
        return False
    
    # check partial order
    for (action1, action2) in relations:
        act1_idx = sequence.index(action1)
        act2_idx = sequence.index(action2)
        if act1_idx >= act2_idx:
            return False

    # check passed
    return True
