
class ConditionContainer:

    def __init__(self, satisfy_fn):
        if not callable(satisfy_fn):
            raise TypeError("Expected function")
        self._satisfy_fn = satisfy_fn
        self._val = None
        self._choices = None
        self._lock = False

    def get(self):
        """
        Get the current set value (default None if not set)
        """
        return self._val

    def set(self, val):
        """
        Set the value, will raise ValueError if not satisfy
        """
        if self._lock:
            raise RuntimeError("Try to set a locked config")
        if self._choices is not None:
            if not self.satisfy(val, self._choices):
                raise ValueError("Fail to set config")
        self._val = val

    def lock(self):
        """
        Lock the value, will not allow change
        """
        self._lock = True

    def satisfy(self, val):
        """
        Check whether the value satisfy the choices

        Returns:
            True if satisfy, False not
        """
        return self._satisfy_fn(val, self._choices)

    def choices(self):
        """
        Return choices.

        Use list(container.choices) to see all the choices
        """
        return self._choices

    def reset(self, choices):
        """
        Reset choices
        """
        self._val = None
        self._choices = choices


class ChoiceContainer(ConditionContainer):

    def __init__(self, choices):
        """
        Create a choice container, the value can only be
        the item in the choices.

        choices (iterable):
            list or range
        """
        def satisfy_fn(val, choices):
            return val in choices
        super().__init__(satisfy_fn)
        self._choices = choices


class TypeContainer(ConditionContainer):

    def __init__(self, type_choices):
        """
        Create a type container, the value can only be
        the instance of the type in the choices.

        type_choices (iterable):
            usually a list[type]
        """
        def satisfy_fn(val, choices):
            for t in choices:
                if isinstance(val, t):
                    return True
            return False
        super().__init__(satisfy_fn)
        self._choices = type_choices
