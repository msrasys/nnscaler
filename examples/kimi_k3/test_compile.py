# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from examples.kimi_k3.compile import build_model, reduced_kimi_k3_config


def test_reduced_kimi_preserves_kda_and_mla_layers():
    config = reduced_kimi_k3_config()
    model = build_model()

    assert config.linear_attn_config["kda_layers"] == [1]
    assert config.linear_attn_config["full_attn_layers"] == [2]
    assert model.model.model.layers[0].is_linear_attn
    assert not model.model.model.layers[1].is_linear_attn
