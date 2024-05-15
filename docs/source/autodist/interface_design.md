# Interface Design

Similar to current user experiences in cube, the entrance to *AutoDist* is a function that accept a data flow graph and a resource descriptor as input. The function returns a rewritten graph. The core modules including
1. *profile*: build cost models to provide the underlying solver with operator and communication information
2. *dp_solver*: encapsulate existing dynamic programming logic

```python
from cube.graph import IRGraph

def annotate_graph(graph: IRGraph) -> AnnotatedIRGraph:
    # TODO
    pass

def profile(anno_graph, resource):
    # use_case:
    # t = comm_cost_model.estimate_cost(primitive='allreduce', size=1024)
    # TODO: multiple dim partition?
    # t, m = comp_cost_model.estimate_cost(op_name='bmm0', idx=0, dim=0, num=4, recompute=False, chunk=False)
    comm_cost_model = build_comm_cost_model(resource)
    comp_cost_model = build_comp_cost_model(anno_graph, resource)
    return (comm_cost_model, comp_cost_model)

def dp_solver(anno_graph: AnnotatedIRGraph, cost_model) -> DistPolicy:
    # TODO: solve the optimization problem
    pass

def rewrite_graph(graph: IRGraph, dist_policy) -> IRGraph:
    # transform the initial dataflow graph according to generated distributed policy
    pass

def autodist(graph: IRGraph, resource: Resource) -> IRGraph:
    anno_graph = annotate_graph(graph)
    cost_model = profile(anno_graph, resource)
    dist_policy = dp_solver(anno_graph, cost_model)
    return rewrite_graph(graph, dist_policy)
```
