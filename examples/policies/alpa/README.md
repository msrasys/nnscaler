
# Alpa Implementation

## Prerequisite

```sh
pip install pulp
```

## Implementation Notes

* The implementation doesn't support auto_layer construction, and relies on the `cube.runtime.function.anchor` as stage division candidates.

* The implementation doesn't support `follow`, which relies on the user customized operator to achieve manual fusion.

* For computation cost:

  * we assume the full efficiency, which is calculated by `cost/tp/dp`

  * Similar with Alpa, we force computation-intensive operators to be partitioned, and allow computation-light operators to be replicated. The computation-intensive operators are defined as operators that require weight for input (usually are customized operators).

* For communication cost:

  * Similar with Alpa, we calculate the cost of communication by `bytes / bandwidth`.


