# einops Support in NnScaler
=================================

Tracing einops Functions are challenging due to their dynamic nature and heavy reliance on string-based patterns and runtime shape manipulations. It is challenging to statically analyze and trace these operations accurately, because tracing doesn't work well with complex python logic (e.g. string parsing, dynamic shape computations, loops, etc) involved in einops functions.

To make things easier, we skip tracing the internal logic of einops functions and directly use the resolved transformation recipes.

This is done by skipping tracing internal einops function: `_prepare_transformation_recipe`. In future, if einops changes their internal implementation, we may need to update our patching logic accordingly.

For nnscaler, we may skip more functions in the future if needed. For exmaple, `_reconstruct_from_shape_uncached` and `_reconstruct_from_shape` are also candidates for skipping tracing, but currently we haven't found issues without skipping them. Once we find issues related to them, we will skip tracing them as well.

As a result, when you use einops functions in your model, we can't guarantee that the traced recipe will be valid when their parameters are changed (e.g. input shapes or pattern strings. `compute_config.constant_folding=False` doesn't help here).

Currently we haven't encountered problems in our tests, but it's still possible in some corner cases. If you encounter any problems, please report an issue to us.

Here is an example of using einops in a model with NnScaler:

```python
import torch
import torch.nn as nn
import einops
from nnscaler import nnscaler, ComputeConfig

class EinopsModel(nn.Module):
    def __init__(self):
        ...

    def forward(self, x):
        # this is good, because the pattern and the input shape is static (h/w/c are fixed)
        x = einops.rearrange(x, 'b (h w c) -> b c h w', h=4, w=4, c=1)
        ...
        y = ...
        # this depends on y
        # although dependence maintains properly if you set `compute_config.constant_folding=False`,
        # This can be changed in future. So be cautious when using such patterns.
        x = einops.rearrange(x, 'b c h w -> b (h w c)', b=y)
        ...
```
