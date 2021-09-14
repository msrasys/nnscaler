

class SymbolTable:
    """
    Symbolic table for saving declared variables.

    Assume the program will first declare all possible used
    variables before entering any of its sub (children) scopes.

    Attributes:
        name (str): name of this scope
        _varlist (dict{str: DType}): declared variable dict
            var_name -> type_of_var
    """

    def __init__(self):
        self._varlist = list()

    def create(self, var_name):
        """
        Create a variable with a type.
        
        If var_name is already declared:
            if the declared type matches with var_name, return False.
            else raise Error
        
        If var name is not declared, decalre the var and return True.

        Args:
            var_name (str): variable name
            var_type (Dtype): variable type
        
        Returns:
            True if declared, False if the var already exists.
        """
        assert isinstance(var_name, str)
        if var_name in self._varlist:
            return False
        else:
            self._varlist.append(var_name)
    
    def exist(self, var_name):
        """
        Check whether a variable exists
        """
        assert isinstance(var_name, str)
        if var_name in self._varlist:
            return True
        else:
            return False
