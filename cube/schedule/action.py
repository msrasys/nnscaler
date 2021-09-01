

class Action:

    def __init__(self, fn):
        """
        fn: a function call to perform a set of operators
        """
        self._fn = [fn,]
        self.pre_actions = list()
        self.outputs = None
        self.name = 'NotSet'
    
    def __call__(self, *args, **kwargs):
        """
        Execute the action
        """
        outputs = self.get_input()
        outputs = self._fn[0](outputs, *args, **kwargs)
        self.outputs = outputs

    def get_input(self):
        """
        Get input for the flow-ins from pre_actions
        """
        raise NotImplementedError

    def add_pre_action(self, action):
        self.pre_actions.append(action)

    def depends_on(self, action):
        """
        check if the self -> action

        Note: this may return false negative as it will only check
        1-hop dependency
        """
        if not isinstance(action, Action):
            raise TypeError("Expected action to be an Action")
        return action in self.pre_actions

    def tag(self, name):
        """
        tag a string to indicate this action (as name)
        """
        self.name = name
    
    def __repr__(self):
        return self.name


def add_flow(action1, action2):
    """
    Add happened before dependency action1 -> action2

    Args:
        action1 (Action)
        action2 (Action)
    """
    if not isinstance(action1, Action):
        raise TypeError("Expected action1 to be an Action")
    if not isinstance(action2, Action):
        raise TypeError("Expected action2 to be an Anction")
    if not action1.depends_on(action2):
        action1.add_pre_action(action2)
