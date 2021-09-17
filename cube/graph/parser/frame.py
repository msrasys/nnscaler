from collections import OrderedDict
from typing import List, Any


class Frame:
    """
    Frame to save call stack and variable
    """
    def __init__(self):

        # var name -> value (IRTesnor, deterministic)
        self._vars: List[dict[str, Any]] = list()
        self._var_stack: List[str] = list()

    def push(self):
        """
        This should only be called when step in a module
        """
        self._vars.append(OrderedDict())

    def pop(self):
        """
        This should only be called step out a module
        """
        if len(self._vars) == 0:
            raise RuntimeError("Try to pop stack with 0 depth")
        self._vars.pop()

    def add_var(self, var_name: str, val: Any, graph_arg: int = -1):
        """
        Add variable to the current frame

        Args:
            var_name (str): variable name (unique)
            val: variable content
            graph_arg (int):
                indicate whether it is an argument of the graph.
                If is 0, is not a graph arg.
                If > 0, is a graph arg, will try to find 
                val from previous frame
        """
        if not isinstance(var_name, str):
            raise RuntimeError("Expected var_name is str")
        if var_name in self._vars[-1]:
            raise KeyError("Try to insert an already existed variable")
        # not a function parameter, no need for mapping
        if graph_arg == -1:
            self._vars[-1][var_name] = val
        # a function parameter, may need for mapping
        elif graph_arg >= 0:
            # root graph entry
            if self.depth() == 1:
                self._vars[-1][var_name] = val
            # fucnton call
            else:
                prev_frame = self._vars[-2]
                param_name = self._var_stack[-1-graph_arg]
                val = prev_frame[param_name]
                self._vars[-1][var_name] = val
        else:
            raise ValueError("graph_arg (int) must be >= 0")

    def get_var(self, var_name: str) -> Any:
        """
        Get variable value according to var_name

        Special mapping between frames (function calls):

            input.x will be mapped to output.k at the about 1-hop frame

        Returns:
            val (Any)
        """
        # first check whether we have variable in this frame
        if var_name in self._vars[-1]:
            return self._vars[-1][var_name]
        raise KeyError(f"Cannot find var name {var_name}")

    def push_param(self, var_name):
        """
        push var name to the method stack

        Args:
            var_name (str): variable name
        """
        if var_name not in self._vars[-1]:
            raise KeyError(f"push {var_name} not declared")
        self._var_stack.append(var_name)

    def pop_param(self, times=1):
        """
        pop var name from the method stack
        """
        for _ in range(times):
            self._var_stack.pop()

    def depth(self):
        return len(self._vars)

    def __repr__(self):
        dscp = f'frame: depth: {self.depth()}\n  var table:'
        for var_name in self._vars[-1].keys():
            dscp += f'\n    {var_name} : {self._vars[-1][var_name]}'
        dscp += f'\n  var stack:'
        for var_name in self._var_stack:
            dscp += f'\n    {var_name}'
        return dscp
