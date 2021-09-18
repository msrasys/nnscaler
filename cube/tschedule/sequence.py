from typing import List, Tuple, NewType
import numpy as np

from cube.tschedule.action import Action


class ASequence:

    def __init__(self, actions: List[Action]):
        
        if not all([isinstance(action, Action) for action in actions]):
            raise TypeError("Expected a list of Actions")

        self.sequence = actions

    def __iter__(self):
        """
        Iterate on the actions
        """
        return self.sequence

    def __len__(self) -> int:
        """
        Get number of action in the sequence
        """
        return len(self.sequence)

    def append(self, action: Action):
        if not isinstance(action, Action):
            raise TypeError("Expected an action")
        self.sequence.append(action)

    def pop(self) -> Action:
        """
        Pop the last action and return
        """
        if len(self.sequence) == 0:
            return None
        return self.sequence.pop()

    def is_correct(self):
        """
        Check whether sequence 
        satisfies the sequential consistency model
        """
        for index, action in enumerate(self.sequence):
            for pre_action in action.pre_actions():
                # find the pre-action not appear in sequence
                if not pre_action in self.sequence:
                    return False
                pre_idx = self.sequence.index(pre_action)
                # violate happened before
                if pre_idx >= index:
                    return False
        return True


# ======= Blow should be moved from this module ======== #

Relation = NewType('Relation', List[Tuple[Action, Action]])


class ScheduleSpace:

    @staticmethod
    def tspace(remain_actions: List[Action],
              path_shuffle=True, 
              relations=None,
              seq: ASequence = ASequence(list())):
        """
        Iterate on the legal sequence space
        """
        if len(remain_actions) == 0:
            yield seq
        # inital entry
        if relations is None:
            relations = ScheduleSpace._get_relations(remain_actions)
        entry_actions = ScheduleSpace._ready_actions(remain_actions, relations)
        entry_actions = np.array(entry_actions)

        # recursive search
        if path_shuffle:
            np.random.shuffle(entry_actions)
        for aid, action in enumerate(entry_actions):
            if len(seq) == 0:
                print(f'> search progress: [{aid+1}/{len(entry_actions)}]...')
            seq.append(action)
            action_idx = remain_actions.index(action)
            sub_actions = remain_actions[:action_idx] + remain_actions[action_idx+1:]
            sub_relations = ScheduleSpace._remove_action(action, relations)
            for res in ScheduleSpace.space(sub_actions, path_shuffle, sub_relations, seq):
                yield res
            seq.pop()


    @staticmethod
    def sspace(actions: List[Action], ndevice: int, path_shuffle=True, depth=0):
        """
        Iterate on the possible action space
        """
        if depth == len(actions):
            yield actions
            return
        action = actions[depth]
        device_choice = np.array(list(range(ndevice)), dtype=np.int)
        if path_shuffle:
            np.random.shuffle(device_choice)
        for device in device_choice:
            action.device = device
            for res in ScheduleSpace.sspace(actions, ndevice, path_shuffle, depth+1):
                yield res


    @staticmethod
    def _ready_actions(actions: List[Action], sub_relations: Relation) -> List[Action]:
        """
        Get ready to emit actions based on sub_relations
        """
        ready_actions = list()
        for action in actions:
            satisfy = True
            for (_, succ) in sub_relations:
                if succ == action:
                    satisfy = False
                    break
            if satisfy:
                ready_actions.append(action)
        return ready_actions


    @staticmethod
    def _get_relations(actions: List[Action]) -> Relation:
        """
        Get relation tuples (Action1 -> Action2)
        """
        relations = list()
        for action in actions:
            relation = [(pre_action, action) for pre_action in action.pre_actions()]
            if len(relation) != 0:
                relations += relation
        return relations


    @staticmethod
    def _remove_action(target: Action, relations: Relation) -> Relation:
        """
        Remove the target action from relation set
        """
        sub_relations = list()
        for (pre, succ) in relations:
            if pre == target or succ == target:
                continue
            sub_relations.append((pre, succ))
        return sub_relations
