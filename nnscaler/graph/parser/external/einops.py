#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import logging

import torch


_logger = logging.getLogger(__name__)

try:
    import einops
    # trigger einops initialization
    einops.rearrange(torch.arange(1), '(a b c) -> a b c', a=1, b=1, c=1)

    from nnscaler.graph.tracer.wrap_utils import default_never_wrap_function, LeafWrapInfo, Location

    default_never_wrap_function[einops.einops._prepare_transformation_recipe] = \
        LeafWrapInfo([Location(einops.einops, '_prepare_transformation_recipe')], False, None)

    # we comment out these two functions
    # because it looks not necessary for now.
    # and they also introduce some problems,
    # i.e. dynamic shape will be lost even with `compute_config.constant_folding=False`

    # default_never_wrap_function[einops.einops._reconstruct_from_shape_uncached] = \
    #     LeafWrapInfo([Location(einops.einops, '_reconstruct_from_shape_uncached')], False, None)
    # default_never_wrap_function[einops.einops._reconstruct_from_shape] = \
    #     LeafWrapInfo([Location(einops.einops, '_reconstruct_from_shape')], False, None)

except ImportError as e:
    _logger.debug("Einops is not installed")
    pass
