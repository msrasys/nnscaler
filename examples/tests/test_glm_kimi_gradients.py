# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Gradient metadata and empty-expert regressions shared by GLM and Kimi."""

import importlib
from collections import Counter
from pathlib import Path
from types import MethodType

import pytest
import torch

from examples.tests.test_training_parity import assert_train_step_interfaces, _training_batch


MODEL_CASES = [
    ("GLM", "micro-glm4", "glm4_dense_tp4", False),
    ("GLM", "micro-glm4-moe", "glm4_moe_355b_ep8_pp4", False),
    ("GLM", "micro-glm4-moe", "glm4_moe_355b_ep8_pp4", True),
    ("GLM", "micro-glm4-moe-lite", "glm5_dsa_744b_ep8_pp4", False),
    ("GLM", "micro-glm-moe-dsa", "glm5_dsa_744b_ep8_pp4", False),
    ("GLM", "micro-glm5-next", "glm5_next_hybrid_ep4", False),
    ("GLM", "micro-glm5-next", "glm5_next_hybrid_ep4", True),
    ("Kimi", "micro-kimi-linear", "kimi_linear_48b_ep8_tp8", False),
    ("Kimi", "micro-kimi-linear", "kimi_linear_expanded_etp4", True),
    ("Kimi", "micro-kimi-k25", "kimi_k25_ep8_pp4", False),
    ("Kimi", "micro-kimi-k25", "kimi_k25_ep8_pp4", True),
]


@pytest.mark.parametrize("grad_enabled", [True, False])
@pytest.mark.parametrize("family,op_name", [
    (family, op_name)
    for family in ["GLM", "Kimi"]
    for op_name in [
        "moe_route", "moe_grouped_gemm", "moe_permute", "grouped_linear",
        "moe_unpermute", "mla_concat_kv", "causal_conv1d", "chunk_kda",
        *(["dsa_index_mask"] if family == "GLM" else [
            "k25_vision_varlen_sdpa", "k25_embed_scatter", "k25_vision_pos_emb", "k25_vision_meta_indices",
        ]),
    ]
])
def test_fake_output_metadata(family, grad_enabled, op_name):
    prefix = family.lower()
    if op_name in ("mla_concat_kv", "dsa_index_mask"):
        category = "mla"
    elif op_name in ("causal_conv1d", "chunk_kda"):
        category = "kda"
    elif op_name.startswith("k25_"):
        category = "vision"
    else:
        category = "moe"
    module = importlib.import_module(f"examples.{family}.ops.{category}_ops")
    runtime_fn = getattr(module, f"{prefix}_{op_name}")
    fake_fn = getattr(module, f"{prefix}_{op_name}_fake")
    hidden = torch.randn(4, 3)
    selected = torch.arange(4).view(4, 1)
    probabilities = torch.rand(4, 1)
    cases = {
        "moe_route": ([hidden, torch.randn(4, 3), torch.randn(4)],
                      {"top_k": 2, "n_group": 2, "topk_group": 1}),
        "moe_grouped_gemm": ([hidden, selected, probabilities, torch.randn(4, 4, 3),
                              torch.randn(4, 3, 2)], {"local_expert_end": 4}),
        "moe_permute": ([hidden, selected], {"num_experts": 4}),
        "grouped_linear": ([hidden, torch.ones(4, dtype=torch.long), torch.randn(4, 2, 3)],
                           {"local_expert_end": 4}),
        "moe_unpermute": ([hidden, probabilities, torch.arange(4), torch.zeros(4, dtype=torch.long)],
                          {"num_tokens": 4}),
        "mla_concat_kv": ([torch.randn(1, 2, 3, 4), torch.randn(1, 1, 3, 2)], {}),
        "causal_conv1d": ([torch.randn(1, 2, 3, 4), torch.randn(2, 3, 2)], {}),
        "chunk_kda": ([torch.randn(1, 4, 2, 4), torch.randn(1, 4, 2, 4),
                       torch.randn(1, 4, 2, 4), -torch.rand(1, 4, 2, 4),
                       torch.rand(1, 4, 2)], {}),
        "dsa_index_mask": (
            [torch.randn(1, 2, 3), torch.randn(1, 2, 2), torch.randn(1, 2, 2), torch.randn(1, 2, 2),
             torch.ones(1, 2, 2, dtype=torch.bool), torch.randn(4, 2), torch.randn(2, 3),
             torch.randn(2), torch.randn(2), torch.randn(2, 3)],
            {"n_heads": 2, "head_dim": 2, "qk_rope_head_dim": 2, "index_topk": 1}),
        "k25_vision_varlen_sdpa": (
            [torch.randn(1, 2, 4, 3), torch.randn(1, 2, 4, 3), torch.randn(1, 2, 4, 3),
             torch.tensor([0, 2, 4], dtype=torch.int32)], {"scaling": 3 ** -0.5}),
        "k25_embed_scatter": (
            [torch.randn(1, 4, 3), torch.tensor([[0, 1, 1, 0]]), torch.randn(2, 3)], {"image_token_id": 1}),
        "k25_vision_pos_emb": (
            [hidden, torch.tensor([[1, 2, 2]]), torch.randn(2, 2, 3), torch.randn(2, 3)],
            {"num_grid_per_side": 2}),
        "k25_vision_meta_indices": ([hidden, torch.tensor([[1, 2, 2]])], {}),
    }
    args, kwargs = cases[op_name]
    for grad_index in [None, -1, *range(len(args))]:
        inputs = [tensor.detach().requires_grad_(
            tensor.is_floating_point() and (grad_index is None or grad_index == index)
        ) for index, tensor in enumerate(args)]
        with torch.set_grad_enabled(grad_enabled):
            expected = runtime_fn(*inputs, **kwargs)
        expected = expected if isinstance(expected, tuple) else (expected,)
        for device in ["cpu", "meta"]:
            fake_inputs = [tensor.detach().to(device).requires_grad_(tensor.requires_grad) for tensor in inputs]
            with torch.set_grad_enabled(grad_enabled):
                actual = fake_fn(*fake_inputs, **kwargs)
            actual = actual if isinstance(actual, tuple) else (actual,)
            assert len(actual) == len(expected)
            for output, reference in zip(actual, expected):
                assert (output.shape, output.dtype, output.requires_grad) == (
                    reference.shape, reference.dtype, reference.requires_grad,
                ), (family, op_name, grad_enabled, grad_index, device)
                assert output.device.type == device


@pytest.mark.parametrize("family", ["GLM", "Kimi"])
@pytest.mark.parametrize("empty_batch", [False, True])
@pytest.mark.parametrize("fused", [False, True])
def test_empty_experts_keep_backward(family, empty_batch, fused):
    module = importlib.import_module(f"examples.{family}.ops.moe_ops")
    prefix = family.lower()
    torch.manual_seed(19)
    tokens = 0 if empty_batch else 4
    inputs = torch.randn(tokens, 3, requires_grad=True)
    probabilities = torch.rand(tokens, 1, requires_grad=True)
    selected = torch.zeros(tokens, 1, dtype=torch.long)
    gate_up = torch.randn(4, 4, 3, requires_grad=True)
    down = torch.randn(4, 3, 2, requires_grad=True)
    structures, partials = [], []
    for expert in range(4):
        x = inputs.detach().requires_grad_()
        gu = gate_up[expert:expert + 1].detach().requires_grad_()
        dn = down[expert:expert + 1].detach().requires_grad_()
        probs = probabilities.detach().requires_grad_()
        if fused:
            output = getattr(module, f"{prefix}_moe_grouped_gemm")(
                x, selected, probs, gu, dn, local_expert_start=expert, local_expert_end=expert + 1,
            )
        else:
            counts = torch.tensor([tokens, 0, 0, 0])
            output = getattr(module, f"{prefix}_grouped_linear")(
                x, counts, gu, local_expert_start=expert, local_expert_end=expert + 1,
            )
        nodes, pending = set(), [output.grad_fn]
        while pending:
            node = pending.pop()
            if node is None or node in nodes:
                continue
            nodes.add(node)
            pending.extend(parent for parent, _ in node.next_functions)
        structures.append(Counter(type(node).__name__ for node in nodes))
        output.sum().backward()
        assert x.grad is not None and gu.grad is not None
        if fused:
            assert dn.grad is not None and probs.grad is not None
        if expert != 0 or empty_batch:
            torch.testing.assert_close(output, torch.zeros_like(output))
            for tensor in ([x, gu, dn, probs] if fused else [x, gu]):
                torch.testing.assert_close(tensor.grad, torch.zeros_like(tensor))
        partials.append(output.detach())
    assert all(structure == structures[0] for structure in structures)
    if fused:
        full = getattr(module, f"{prefix}_moe_grouped_gemm")(inputs, selected, probabilities, gate_up, down)
    else:
        full = getattr(module, f"{prefix}_grouped_linear")(inputs, torch.tensor([tokens, 0, 0, 0]), gate_up)
    torch.testing.assert_close(sum(partials), full)


def _micro_strategy(driver, family, strategy_name, expand, ranks):
    policy = importlib.import_module(f"examples.{family}.policy.json_pas")
    strategy = driver.load_strategy_config(
        Path(driver.__file__).parent / "strategies" / f"{strategy_name}.json",
    )
    strategy.mesh.plan_ngpus = strategy.mesh.runtime_ngpus = ranks
    strategy.mesh.pipeline_nstages = strategy.mesh.pipeline_size = strategy.mesh.pipeline_nmicros = 1
    strategy.stage_splits = []
    strategy.rewrites.expand_moe_ops = expand
    if expand and not any("grouped_linear" in (rule.op_regex or "") for rule in strategy.rules):
        strategy.rules.extend([
            policy.PartitionRule("expanded_down", "expert_tp_row",
                                 op_regex="grouped_linear", param_regex="down_proj"),
            policy.PartitionRule("expanded_gate_up", "expert_tp_col", op_regex="grouped_linear"),
        ])
    return strategy


def _assert_graph_gradients(graph, family):
    from nnscaler.graph.function.pyfunc import IRPyFunc
    from nnscaler.ir.operator import IRFwOperation
    from nnscaler.ir.tensor import IRTensor

    for node in graph.select(ntype=IRFwOperation):
        if isinstance(node, IRPyFunc):
            assert not any(isinstance(output, IRTensor) and output.requires_grad for output in node.outputs()), node
        if not node.signature.startswith(f"examples.{family}.ops."):
            continue
        if node.signature.endswith(("dsa_index_mask", "vision_meta_indices")):
            assert all(not output.requires_grad for output in node.outputs() if isinstance(output, IRTensor))
            continue
        for output in node.outputs():
            if isinstance(output, IRTensor) and output.dtype.is_floating_point:
                assert output.requires_grad and node.mirror is not None, node


@pytest.mark.parametrize("family,preset,strategy_name,expand", MODEL_CASES)
def test_micro_parallel_codegen(tmp_path, monkeypatch, family, preset, strategy_name, expand):
    from nnscaler.graph.gener.gen import IRAdapterGener
    from nnscaler.ir.adapter import IRAdapter
    from nnscaler.ir.operator import IRFwOperation
    from nnscaler.ir.tensor import IRTensor

    driver = importlib.import_module(f"examples.{family}.compile")
    strategy = _micro_strategy(driver, family, strategy_name, expand, ranks=4)
    original_audit = driver.audit_pre_pas

    def audit(graph):
        _assert_graph_gradients(graph, family)
        return original_audit(graph)

    monkeypatch.setattr(driver, "audit_pre_pas", audit)
    original_gen = IRAdapterGener.gen

    def generate_adapters(graph, **kwargs):
        graph = original_gen(graph, **kwargs)
        boundaries = {
            tensor.parent.tid
            for node in graph.select(ntype=IRFwOperation)
            if node.signature.startswith(f"examples.{family}.ops.")
            for tensor in node.inputs() + node.outputs()
            if isinstance(tensor, IRTensor) and tensor.requires_grad
        }
        for adapter in graph.select(ntype=IRAdapter):
            if adapter.isfw() and any(
                isinstance(tensor, IRTensor) and tensor.parent.tid in boundaries
                for tensor in adapter.inputs() + adapter.outputs()
            ):
                assert adapter.differentiable, adapter
        return graph

    monkeypatch.setattr(IRAdapterGener, "gen", staticmethod(generate_adapters))
    kwargs = {"multimodal": True} if preset == "micro-kimi-k25" else {}
    manifest = getattr(driver, f"compile_{family.lower()}")(
        preset, strategy, tmp_path, batch_size=2, seq_len=16, weightless=True, **kwargs,
    )
    assert manifest["pre_pas_summary"]["audit_passed"]
    assert manifest["codegen_validation"]["all_ranks_present"]
    files = list((tmp_path / "04_generated").rglob("gencode*.py"))
    assert len(files) == 4
    for path in files:
        assert_train_step_interfaces(path)


@pytest.mark.parametrize("family,preset,strategy_name,expand", MODEL_CASES)
def test_micro_generated_training(tmp_path, monkeypatch, family, preset, strategy_name, expand):
    from nnscaler.parallel import _load_parallel_module_class

    driver = importlib.import_module(f"examples.{family}.compile")
    strategy = _micro_strategy(driver, family, strategy_name, expand, ranks=1)
    captured = {}
    original_parallelize = driver.parallelize
    instance_name = f"{family}_{preset}_{expand}".replace("-", "_")

    def capture(model, dummy_args, *args, **kwargs):
        captured.update(model=model, samples=dummy_args["samples"], gen_savedir=kwargs["gen_savedir"])
        kwargs["instance_name"] = instance_name
        return original_parallelize(model, dummy_args, *args, **kwargs)

    monkeypatch.setattr(driver, "parallelize", capture)
    torch.manual_seed(20261127)
    kwargs = {"multimodal": True} if preset == "micro-kimi-k25" else {}
    getattr(driver, f"compile_{family.lower()}")(
        preset, strategy, tmp_path, batch_size=2, seq_len=16, weightless=False, **kwargs,
    )
    reference = captured["model"]
    generated_class = _load_parallel_module_class(
        type(reference), gen_savedir=captured["gen_savedir"], instance_name=instance_name, rank=0,
    )
    generated = torch.nn.Module()
    metadata = generated_class.get_attr_meta_map(0)
    parameters = dict(reference.named_parameters())
    for name, attr in metadata.items():
        if attr.is_param:
            parameter = reference.get_parameter(attr.orig_name)
            generated.register_parameter(
                name, torch.nn.Parameter(parameter.detach().clone(), requires_grad=parameter.requires_grad),
            )
        else:
            tensor = parameters[attr.orig_name] if attr.orig_name in parameters else reference.get_buffer(attr.orig_name)
            generated.register_buffer(name, tensor.detach().clone())
    for name, method in vars(generated_class).items():
        if name.startswith("segment"):
            setattr(generated, name, MethodType(method, generated))
    for path in (tmp_path / "04_generated").rglob("gencode*.py"):
        assert_train_step_interfaces(path)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.001)
    generated_optimizer = torch.optim.SGD(generated.parameters(), lr=0.001)
    vocab_size = reference.model.get_input_embeddings().num_embeddings
    for step in range(3):
        samples = _training_batch(captured["samples"], step, vocab_size,
                                  getattr(reference.model.config, "image_token_id", None))
        reference_optimizer.zero_grad()
        generated_optimizer.zero_grad()
        expected_loss = reference(samples)
        expected_loss.backward()
        with monkeypatch.context() as context:
            context.setattr(torch.cuda, "current_device", lambda: "cpu")
            actual_loss = generated_class._train_step(generated, iter([samples]))
        torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-4, atol=1e-5)
        represented = {attr.orig_name for attr in metadata.values() if attr.is_param}
        assert {name for name, parameter in parameters.items() if parameter.grad is not None} <= represented
        for name, attr in metadata.items():
            if not attr.is_param:
                continue
            actual = generated.get_parameter(name)
            expected = reference.get_parameter(attr.orig_name)
            assert (actual.grad is None) == (expected.grad is None), attr.orig_name
            if expected.grad is not None:
                torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-4, atol=1e-5, msg=attr.orig_name)
        generated_optimizer.step()
        reference_optimizer.step()
        for name, attr in metadata.items():
            if attr.is_param:
                torch.testing.assert_close(
                    generated.get_parameter(name), reference.get_parameter(attr.orig_name),
                    rtol=1e-4, atol=1e-5, msg=attr.orig_name,
                )
