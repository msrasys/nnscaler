# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch

from examples.glm5.compile import (
    GLM5TrainingModel,
    install_glm5_nnscaler_adapters,
    reduced_glm5_config,
)


def test_reduced_glm5_preserves_dsa_and_sparse_moe():
    config = reduced_glm5_config()

    assert config.model_type == "glm_moe_dsa"
    assert config.index_topk > 0
    assert config.index_head_dim > config.qk_rope_head_dim
    assert config.mlp_layer_types == ["sparse"]
    assert config.n_routed_experts > 0
    assert config._experts_implementation == "batched_mm"


def test_reduced_glm5_forward_and_backward():
    install_glm5_nnscaler_adapters()
    config = reduced_glm5_config()
    model = GLM5TrainingModel(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))

    loss = model(input_ids)
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert model.model.model.embed_tokens.weight.grad is not None
    assert all(
        not parameter.requires_grad
        for parameter in model.model.model.layers[0].self_attn.indexer.parameters()
    )
