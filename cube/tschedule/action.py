from typing import List


class Action:
    """
    Action represents a (sub-)graph which contains operators on the
    same device
    """
    def __init__(self, ir_graph, device: int):

        if not isinstance(device, int):
            raise TypeError("Require device to be int")
        # set up attributes
        self.graph = ir_graph
        self.device: int = device
        self.name: str = None
        # dependencies
        self._pre_actions: List[Action] = list()
        self._post_actions: List[Action] = list()

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device):
        for op in self.graph.nodes():
            op.deivce = device
        self._device = device

    def tag(self, name: str):
        """
        Tag a string to indicate this action (as name)
        """
        self.name = name

    def happen_before(self, action):
        """
        Check if the self -> (happened before) action
        """
        if not isinstance(action, Action):
            raise TypeError("Expected action to be an Action")
        return action in self._post_actions

    def post_actions(self):
        """
        Get post action list
        """
        return self._post_actions

    def happen_after(self, action):
        """
        Check if the action -> (happened before) self

        Note: this may return false negative as it will only check
        1-hop dependency
        """
        if not isinstance(action, Action):
            raise TypeError("Expected action to be an Action")
        return action in self._pre_actions

    def pre_actions(self):
        """
        Get pre action list

        Note: this may return false negative as it will only check
        1-hop dependency
        """
        return self._pre_actions

    def add_flow(self, action):
        """
        Make this action (self) -> (happened before) action
        """
        if not isinstance(action, Action):
            raise TypeError("Expected action to be Action")
        self._post_actions.append(action)
        action._add_pre_action(self)

    def _add_pre_action(self, action):
        """
        Add successor that requries this action happened first
        """
        if not isinstance(action, Action):
            raise TypeError("Expected action to be Action")
        self._successors.append(action)

    def __repr__(self):
        dscp = f'Action({self.name}):\n\t{self.graph.outputs()} <- {self.graph.inputs()}'
        return dscp
    