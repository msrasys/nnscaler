import copy
import z3

def choices(solver, attributes):
    """
    Iterate each the config space

    Args:
        solver (z3.z3.Solver)
        attributes (list[z3.z3.xx])
    
    Yield:
        config (z3.z3.ModelRef)
    """
    if not isinstance(solver, z3.z3.Solver):
        raise TypeError("Expected solver to be an z3 solver")
    solver = copy.deepcopy(solver)
    while solver.check() == z3.sat:
        config = solver.model()
        solver.add(
            z3.Or([z3.Not(attr == config[attr]) for attr in attributes])
        )
        yield config
        if len(attributes) == 0:
            break
