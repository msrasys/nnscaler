import os
plan_ngpus = int(os.environ['PLAN_NGPUS'])
runtime_ngpus = int(os.environ['CUBE_SCALING_FACTOR']) * plan_ngpus

# 1. load graph
from cube.graph import IRGraph
graph = IRGraph.load('megatron_gpt2.cube')

# 2. register customized op
from gpt_model import GeLUFunction
from cube.graph.parser.register import register
register('* h, h -> * h')(GeLUFunction.apply)

# 3. parallel model
from fairseq.cube.pas_policies import PASData, PASRandomSPMD
graph = PASData(graph, plan_ngpus)

for node in graph.nodes(flatten=True):
    from cube.graph.function.anchor import IRGraphAnchor
    from cube.graph.function.pyfunc import IRPyFunc
    # skip graph anchor and multiref: they will be removed or replaced by system
    if isinstance(node, IRGraphAnchor) or node.name == 'multiref':
        graph.assign(node, 0)
    if isinstance(node, IRPyFunc):
        graph.assign(node, 0)
    if len(node.device) == 0:
        raise RuntimeError(f"Node {node} device is not set")
from cube.graph.gener.gen import IRAdapterGener
graph = IRAdapterGener.gen(graph, cost_fn=None)
if graph.sched is not None:
    graph.sched.apply()
    print(graph.sched)

from cube.graph.schedule.schedplan import SchedulePlan
from cube.execplan import ExecutionPlan
if isinstance(graph.sched, SchedulePlan):
    execplan = ExecutionPlan.from_schedplan(graph.sched)
else:
    execplan = ExecutionPlan.from_graph(graph)
# execplan.visualize('plan.png')
from cube.execplan.planpass.fusion import DiffFusion
execplan = DiffFusion.apply(execplan)
# plan pass for computation grouping
from cube.execplan.planpass.grouping import Grouping
if not graph.sched:
    execplan = Grouping.apply(execplan)

# 4. generate code
from cube.codegen import ModuleCodeGen, ScheduleCodeGen
filename = 'gencode{}.py'
_runtime_ngpus = None if plan_ngpus == runtime_ngpus else runtime_ngpus
assert len(execplan.graph.device) == plan_ngpus, f"{execplan.graph.device}"
mgener = ModuleCodeGen(execplan, scale_ndevs=_runtime_ngpus)
sgener = ScheduleCodeGen(execplan, scale_ndevs=_runtime_ngpus)
for rank in range(runtime_ngpus):
    fname = filename.format(rank)
    # generate spatial module code
    mgener.gen(rank, outfile=fname, attach=False)
    # generate temporal schedule code
    sgener.gen(
        device = rank,
        outfile = fname,
        attach=True
    )
