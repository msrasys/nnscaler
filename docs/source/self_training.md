# self.training support

To parallelize the training process, we firstly need to trace the module and get a static computational graph.

A common problem with static graph is that it is impossible to handle control flow.

But on the other hand, `self.training` is very common used in module forward method.
So we add a very limited support for `self.training` in tracing.

Please note that user code is flattened and transformed into a single `ParallelModule` at runtime, so `training` is a global module state, and we don't support the case that user want to set a sub-module's training to True but remaining modules to False.

## `if` statement

We don't support any control flow, so For the following code, we only put the `if` branch that is executed during tracing into the graph.

```python
if self.training:
    ...
else
    ...
```
The consequence is that model training/validation will use exactly the same code path.

## `if` expression

Some torch operations use `if` expression to select different parameters, for example

```python
torch.nn.functional.scaled_dot_product_attention(
    q, k, v, attn_mask=None,
    dropout_p=self.dropout if self.training else 0,
    is_causal=self.is_causal
)
```
To support that, we provide a limited `if` expression support,
by converting `if` expression to a function call.

For example:

We will convert

```python
x = a if self.training else b
```
to
```python
x = nnscaler.runtime.function.ifexpr(self.training, a, b)
```

This trick is not free. It will introduce two side effects:
1. Short-circuit evaluation is not supported.
Both branches will be evaluated, so you must make sure that both branches are valid, and have no side effect.
To reduce the side effect, we will check true expr/false expr, and requires both don't contain function calls.
so the following code will not be converted:
    ```python
    x = f(a) if self.training else b
    ```
2. We will convert `if` expression only if the condition is `self.training`.
So if a non-module class has a `training` attribute, the `if` expression in its member functions will also be converted if its condition is `self.training`.

Please note you can always use `register_op` to define a custom op to handle the `if` expression.
For example, you can convert the above code to:
```python
import nnscaler
import torch


@nnscaler.register_op('?, ? -> ?')
def get_dropout(training, dropout):
    return dropout if training else 0

torch.nn.functional.scaled_dot_product_attention(
    q, k, v, attn_mask=None,
    dropout_p=get_dropout(self, self.dropout),
    is_causal=self.is_causal
)
``

## self.training as a parameter

If you use `self.training` as a parameter, it is well supported.

For example:
```python
torch.nn.functional.dropout(x, 0.1, self.training)
# the generated code will be exactly the same as the original code:
# torch.nn.functional.dropout(x, 0.1, self.training)
```

But be careful, if you use `self.training` in a boolean operation,
the generated code may be not as you expected, because
1. We don't trace bool operations.
2. Boolean operations are short-circuit evaluated, so only one expression will be kept in generated code.

For example:
```python
torch.nn.functional.dropout(x, 0.1, global_setting.enable_dropout or self.training)
# if global_setting.enable_dropout is True, the generated code will be
# torch.nn.functional.dropout(x, 0.1, True)
# if global_setting.enable_dropout is False, the generated code will be
# torch.nn.functional.dropout(x, 0.1, self.training)
```
