from collections import OrderedDict
from typing import List, Any, Dict
import torch


class Frame:
    """
    Frame to save call stack and variable
    """
    def __init__(self):

        # var name -> value (IRTesnor, deterministic)
        self._vars: List[dict[str, Any]] = list()
        self._var_stack: List[str] = list()
        # module attributes
        self._attributes: List[dict[str, Any]] = list()
        self._attr_vals: Dict[int, Any] = dict()  # tensor tid to real value mapping

    def push_var(self, inherit_from_top=False):
        """
        Push a new variable frame as current variable frame.
        This should only be called when stepping in a module or method.

        Args:
            inherit_from_top (bool): 
                whether to make all already defined variables in the top frame 
                accessible to the evaluation procedure
                (e.g. references to such variables won't cause VarNotFound exception).
        """
        if inherit_from_top:
            assert len(self._vars) > 0
            self._vars.append(self._vars[-1].copy())
        else:
            self._vars.append(OrderedDict())

    def pop_var(self):
        """
        Pop the current variable frame.
        This should only be called when steping out a module or method.
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
                indicate whether it is an argument of the graph. -1 indicates not an argument.
                If >= 0, is a graph arg, will try to find val from variable stack,
                and link the name of the argument name from the callee function
                to the names of the argument passed-in.
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
        
    def set_var(self, var_name: str, val: Any):
        """
        Reset a variable with arbitrary value.
        If `var_name` doesn't exist, will create a new one
        
        @param var_name str: variable name
        @param val Any
        """
        self._vars[-1][var_name] = val

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
        raise KeyError(f"Cannot find var name {var_name} in {self._vars}")

    def push_attr(self):
        """
        Push a new module attribut frame as current frame.
        This should only be called when stepping in the graph.
        """
        self._attributes.append(OrderedDict())

    def pop_attr(self):
        """
        Pop the current module attribute frame.
        This should only be called when stepping out the graph.
        """
        self._attributes.pop()

    def add_attr(self, name: str, val: Any):
        """
        Add module attribute <name: val>
        """
        if name in self._attributes[-1]:
            raise KeyError("Try to add an already existed attributed")
        self._attributes[-1][name] = val

    def get_attr(self, name: str) -> Any:
        """
        Get module attribute by name
        """
        if name not in self._attributes[-1]:
            raise KeyError(f"Cannot find var name {name}")
        return self._attributes[-1][name]

    def has_attr(self, name: str) -> bool:
        """
        Return if `name` exists in current attributes
        """
        return name in self._attributes[-1]

    def add_attr_content(self, tid: int, val: torch.Tensor):
        """
        Add module attribute content
        """
        if torch.is_tensor(val):
            val = val.cpu()
        self._attr_vals[tid] = val

    def save_attr_content(self, save_file: str = 'fullmodel.pt'):
        """
        Save attribute content into file.
        """
        torch.save(self._attr_vals, save_file)

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
