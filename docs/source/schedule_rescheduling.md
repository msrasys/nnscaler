# Operator schedule rescheduling

nnScaler can dump an execution schedule, visualize its dependencies, apply a preferred operator order, and regenerate code while preserving data dependencies.

## Dump a baseline schedule

Set the compile flags before calling `nnscaler.parallelize`:

```python
from nnscaler.flags import CompileFlag

CompileFlag.dump_op_schedule = "/tmp/schedule.yaml"
CompileFlag.dump_op_schedule_graph = "/tmp/schedule.dot"
```

The YAML or JSON file contains complete operator descriptors. The DOT file contains the same operators arranged by segment, dependency edges, and communication-order edges.

## Edit the DOT in the browser

The standalone viewer is packaged with nnScaler:

```python
from nnscaler.execplan.planpass.reschedule import schedule_viewer_path

print(schedule_viewer_path())
```

Open that HTML file in a browser, load one or two local DOT files, and use the **DOT Order Editor** to drag operators within a segment. The editor reports dependency violations, highlights invalid edges, distinguishes communication operators in yellow, and saves the result as a new DOT file.

## Convert the edited DOT

Convert the edited DOT back into a reschedule config using the original baseline config:

```python
from nnscaler.execplan.planpass.reschedule import convert_manual_dot_to_config

convert_manual_dot_to_config(
    "/tmp/schedule.edited.dot",
    "/tmp/schedule.yaml",
    "/tmp/schedule.edited.yaml",
)
```

Operators omitted from the DOT retain their original relative order. Unknown CIDs are ignored. Each DOT cluster is mapped to the baseline segment at the same index.

## Generate code with the edited order

```python
from nnscaler.flags import CompileFlag

CompileFlag.enable_op_reschedule = True
CompileFlag.op_reschedule_scope = "segment"
CompileFlag.op_reschedule_config = "/tmp/schedule.edited.yaml"
```

Run `nnscaler.parallelize` again with the same model and parallel plan. The edited order is treated as a preference: nnScaler's dependency graph corrects illegal preferences to a legal topological order before code generation.
