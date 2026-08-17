#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import gc

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from nnscaler.runtime.executor import Executor
from nnscaler.runtime._patch_torch import configure_fbw_runtime
from nnscaler.runtime._patch_torch_checkpoint import ReusableGraphExecGroup
from nnscaler.flags import RuntimeFlag
from nnscaler.runtime.adapter import nn as adapter_nn
from nnscaler.runtime.function import function as runtime_function


@pytest.fixture(autouse=True)
def clear_executor():
    Executor.clear()
    yield
    Executor.clear()


def _make_linears(dtype=torch.float64):
    reference = torch.nn.Linear(8, 16, dtype=dtype)
    actual = torch.nn.Linear(8, 16, dtype=dtype)
    actual.load_state_dict(reference.state_dict())
    return reference, actual


def test_split_backward_matches_full_backward():
    torch.manual_seed(0)
    reference, actual = _make_linears()
    input_data = torch.randn(4, 8, dtype=torch.float64)
    output_grad = torch.randn(4, 16, dtype=torch.float64)

    reference_input = input_data.clone().requires_grad_()
    reference(reference_input).backward(output_grad)

    actual_input = input_data.clone().requires_grad_()
    output = Executor.fexecute('linear', actual, actual_input)
    input_grad = Executor.backward_input(
        'linear', [actual_input], [output], [output_grad], actual.parameters()
    )

    torch.testing.assert_close(input_grad, reference_input.grad)
    assert actual.weight.grad is None
    assert actual.bias.grad is None

    Executor.backward_weight('linear', actual.parameters())

    torch.testing.assert_close(actual.weight.grad, reference.weight.grad)
    torch.testing.assert_close(actual.bias.grad, reference.bias.grad)
    Executor.check_clear()


def test_selective_split_defers_only_registered_weight_work():
    class SelectiveLinearFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input_tensor, weight):
            ctx.save_for_backward(input_tensor, weight)
            return input_tensor @ weight.t()

        @staticmethod
        def backward(ctx, output_grad):
            input_tensor, weight = ctx.saved_tensors
            input_grad = output_grad @ weight
            if RuntimeFlag.fbw_phase == "input":
                weight_grad = output_grad.t() @ input_tensor

                def backward_dw(weight=weight, weight_grad=weight_grad):
                    return ((weight, weight_grad),)

                RuntimeFlag.defer_fbw_weight_task(backward_dw, (weight,))
                return input_grad, None
            return input_grad, output_grad.t() @ input_tensor

    class SelectiveLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(16, 8, dtype=torch.float64))
            self.bias = torch.nn.Parameter(torch.randn(16, dtype=torch.float64))

        def forward(self, input_tensor):
            return SelectiveLinearFunction.apply(input_tensor, self.weight) + self.bias

    torch.manual_seed(31)
    reference = SelectiveLinear()
    actual = SelectiveLinear()
    actual.load_state_dict(reference.state_dict())
    input_data = torch.randn(4, 8, dtype=torch.float64)
    output_grad = torch.randn(4, 16, dtype=torch.float64)

    reference_input = input_data.clone().requires_grad_()
    reference(reference_input).backward(output_grad)

    previous = RuntimeFlag.fbw_accumulate_undeferred_grads
    RuntimeFlag.fbw_accumulate_undeferred_grads = True
    try:
        actual_input = input_data.clone().requires_grad_()
        output = Executor.fexecute("selective_linear", actual, actual_input)
        input_grad = Executor.backward_input(
            "selective_linear",
            [actual_input],
            [output],
            [output_grad],
            actual.parameters(),
        )

        torch.testing.assert_close(input_grad, reference_input.grad)
        torch.testing.assert_close(actual.bias.grad, reference.bias.grad)
        assert actual.weight.grad is None

        Executor.backward_weight("selective_linear", actual.parameters())
        torch.testing.assert_close(actual.weight.grad, reference.weight.grad)
        torch.testing.assert_close(actual.bias.grad, reference.bias.grad)
        Executor.check_clear()
    finally:
        RuntimeFlag.fbw_accumulate_undeferred_grads = previous


def test_selective_split_accumulates_embedding_weight_during_input_backward():
    class Embedding(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.randn(16, 8, dtype=torch.float64)
            )

        def forward(self, token_ids):
            return runtime_function.embedding(
                token_ids,
                self.weight,
                padding_idx=None,
                start=0,
                stop=16,
            )

    torch.manual_seed(47)
    reference = Embedding()
    actual = Embedding()
    actual.load_state_dict(reference.state_dict())
    token_ids = torch.tensor([1, 4, 4, 7])
    output_grad = torch.randn(4, 8, dtype=torch.float64)

    reference(token_ids).backward(output_grad)

    previous = RuntimeFlag.fbw_accumulate_undeferred_grads
    RuntimeFlag.fbw_accumulate_undeferred_grads = True
    try:
        output = Executor.fexecute("selective_embedding", actual, token_ids)
        input_grads = Executor.backward_input(
            "selective_embedding",
            [token_ids],
            [output],
            [output_grad],
            actual.parameters(),
        )

        assert input_grads is None
        torch.testing.assert_close(actual.weight.grad, reference.weight.grad)

        Executor.backward_weight("selective_embedding", actual.parameters())
        torch.testing.assert_close(actual.weight.grad, reference.weight.grad)
        Executor.check_clear()
    finally:
        RuntimeFlag.fbw_accumulate_undeferred_grads = previous


def test_split_backward_groups_bridge_shared_parameter_paths():
    class BridgedParameters(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias_x = torch.nn.Parameter(torch.randn(8, dtype=torch.float64))
            self.bias_y = torch.nn.Parameter(torch.randn(8, dtype=torch.float64))
            self.weight = torch.nn.Parameter(torch.randn(8, 8, dtype=torch.float64))

        def forward(self, input_x, input_y):
            return (
                torch.addmm(self.bias_x, input_x, self.weight)
                + torch.addmm(self.bias_y, input_y, self.weight)
            )

    torch.manual_seed(29)
    reference = BridgedParameters()
    actual = BridgedParameters()
    actual.load_state_dict(reference.state_dict())
    input_x = torch.randn(4, 8, dtype=torch.float64)
    input_y = torch.randn(4, 8, dtype=torch.float64)
    output_grad = torch.randn(4, 8, dtype=torch.float64)

    reference_x = input_x.clone().requires_grad_()
    reference_y = input_y.clone().requires_grad_()
    reference(reference_x, reference_y).backward(output_grad)

    actual_x = input_x.clone().requires_grad_()
    actual_y = input_y.clone().requires_grad_()
    output = Executor.fexecute("bridged_params", actual, actual_x, actual_y)
    input_grads = Executor.backward_input(
        "bridged_params",
        [actual_x, actual_y],
        [output],
        [output_grad],
        actual.parameters(),
    )
    Executor.backward_weight("bridged_params", actual.parameters())

    torch.testing.assert_close(input_grads[0], reference_x.grad)
    torch.testing.assert_close(input_grads[1], reference_y.grad)
    for actual_param, reference_param in zip(
        actual.parameters(), reference.parameters(), strict=True
    ):
        torch.testing.assert_close(actual_param.grad, reference_param.grad)
    Executor.check_clear()


def test_split_backward_reuses_checkpoint_recompute_for_weight_phase():
    class CheckpointedMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w1 = torch.nn.Linear(8, 16, dtype=torch.float64)
            self.w2 = torch.nn.Linear(16, 8, dtype=torch.float64)
            self.region_calls = 0

        def region(self, input_tensor):
            self.region_calls += 1
            return self.w2(torch.nn.functional.gelu(self.w1(input_tensor)))

        def forward(self, input_tensor):
            return checkpoint(self.region, input_tensor, use_reentrant=False)

    torch.manual_seed(11)
    configure_fbw_runtime()
    reference = CheckpointedMLP()
    actual = CheckpointedMLP()
    actual.load_state_dict(reference.state_dict())
    input_data = torch.randn(4, 8, dtype=torch.float64)
    output_grad = torch.randn(4, 8, dtype=torch.float64)

    reference_input = input_data.clone().requires_grad_()
    reference(reference_input).backward(output_grad)

    actual_input = input_data.clone().requires_grad_()
    output = Executor.fexecute('checkpointed_mlp', actual, actual_input)
    input_grad = Executor.backward_input(
        'checkpointed_mlp',
        [actual_input],
        [output],
        [output_grad],
        actual.parameters(),
    )
    Executor.backward_weight('checkpointed_mlp', actual.parameters())

    # Both paths run the original forward once and checkpoint replay once.
    assert reference.region_calls == 2
    assert actual.region_calls == 2
    torch.testing.assert_close(input_grad, reference_input.grad)
    for actual_param, reference_param in zip(
        actual.parameters(), reference.parameters(), strict=True
    ):
        torch.testing.assert_close(actual_param.grad, reference_param.grad)
    Executor.check_clear()


def test_checkpoint_recompute_cache_is_paired_per_microbatch():
    class CheckpointedLinear(torch.nn.Linear):
        def __init__(self):
            super().__init__(8, 8, dtype=torch.float64)
            self.region_calls = 0

        def region(self, input_tensor):
            self.region_calls += 1
            return torch.nn.functional.linear(input_tensor, self.weight, self.bias)

        def forward(self, input_tensor):
            return checkpoint(self.region, input_tensor, use_reentrant=False)

    configure_fbw_runtime()
    module = CheckpointedLinear()
    inputs = [
        torch.randn(4, 8, dtype=torch.float64, requires_grad=True)
        for _ in range(2)
    ]
    output_grads = [torch.randn(4, 8, dtype=torch.float64) for _ in range(2)]
    outputs = [
        Executor.fexecute('checkpointed_linear', module, input_tensor)
        for input_tensor in inputs
    ]

    for input_tensor, output, output_grad in zip(
        inputs, outputs, output_grads, strict=True
    ):
        Executor.backward_input(
            'checkpointed_linear',
            [input_tensor],
            [output],
            [output_grad],
            module.parameters(),
        )
    assert module.region_calls == 4  # two forwards plus two I replays

    Executor.backward_weight('checkpointed_linear', module.parameters())
    Executor.backward_weight('checkpointed_linear', module.parameters())
    assert module.region_calls == 4  # both W phases reuse their matching I cache
    Executor.check_clear()


def test_checkpoint_replay_survives_all_weight_groups_then_releases():
    configure_fbw_runtime()

    @torch.compile(backend='aot_eager', fullgraph=True)
    def compiled_region(input_tensor, weight1, weight2):
        return torch.sin(input_tensor * weight1 + weight2)

    input_tensor = torch.randn(4, 8, requires_grad=True)
    weight1 = torch.randn(8, requires_grad=True)
    weight2 = torch.randn(8, requires_grad=True)
    output = checkpoint(
        compiled_region,
        input_tensor,
        weight1,
        weight2,
        use_reentrant=False,
    )
    output_grad = torch.randn_like(output)
    group = ReusableGraphExecGroup()

    try:
        RuntimeFlag.fbw_phase = 'input'
        with group:
            torch.autograd.grad(
                output,
                input_tensor,
                output_grad,
                retain_graph=True,
            )

        RuntimeFlag.fbw_phase = 'weight'
        with group:
            torch.autograd.grad(
                output,
                weight1,
                output_grad,
                retain_graph=True,
            )
            torch.autograd.grad(output, weight2, output_grad)

        assert group._checkpoint_frames
        frames = tuple(group._checkpoint_frames)
        group.release()
        assert not group._checkpoint_frames
        for frame in frames:
            assert group not in frame.recomputed
            assert group not in frame.recomp_counter
            assert group not in frame.is_recomputed
            assert all(
                holder is None or group not in holder.handles
                for holder in (weak_holder() for weak_holder in frame.weak_holders)
            )
    finally:
        RuntimeFlag.fbw_phase = None
        RuntimeFlag.fbw_deferred_tasks = None


def test_split_backward_caches_opaque_aot_weight_grads():
    configure_fbw_runtime()

    @torch.compile(backend='aot_eager', fullgraph=True)
    def compiled_region(input_tensor, weight1, weight2):
        return torch.sin(input_tensor * weight1 + weight2)

    class CheckpointedAOTModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight1 = torch.nn.Parameter(torch.randn(8))
            self.weight2 = torch.nn.Parameter(torch.randn(8))
            self.region_calls = 0

        def region(self, input_tensor):
            self.region_calls += 1
            return compiled_region(input_tensor, self.weight1, self.weight2)

        def forward(self, input_tensor):
            return checkpoint(self.region, input_tensor, use_reentrant=False)

    torch.manual_seed(17)
    reference = CheckpointedAOTModule()
    actual = CheckpointedAOTModule()
    actual.load_state_dict(reference.state_dict())
    input_data = torch.randn(4, 8)
    output_grad = torch.randn(4, 8)

    reference_input = input_data.clone().requires_grad_()
    reference(reference_input).backward(output_grad)

    actual_input = input_data.clone().requires_grad_()
    output = Executor.fexecute('checkpointed_aot', actual, actual_input)
    backward_calls = []
    hook = output.grad_fn.register_hook(
        lambda grad_inputs, grad_outputs: backward_calls.append(None)
    )
    input_grad = Executor.backward_input(
        'checkpointed_aot',
        [actual_input],
        [output],
        [output_grad],
        actual.parameters(),
    )
    assert len(backward_calls) == 1
    # Opaque AOT backward cannot prune dWeight from I. Consume the completed
    # result immediately so several pending I actions do not retain one full
    # dWeight allocation each.
    torch.testing.assert_close(actual.weight1.grad, reference.weight1.grad)
    torch.testing.assert_close(actual.weight2.grad, reference.weight2.grad)
    Executor.backward_weight('checkpointed_aot', actual.parameters())
    # W consumes the dWeights already returned by the opaque callback in I;
    # it must not execute the complete compiled backward again.
    assert len(backward_calls) == 1
    hook.remove()

    assert actual.region_calls == 2
    torch.testing.assert_close(input_grad, reference_input.grad)
    for actual_param, reference_param in zip(
        actual.parameters(), reference.parameters(), strict=True
    ):
        torch.testing.assert_close(actual_param.grad, reference_param.grad)
    Executor.check_clear()


def test_split_backward_caches_opaque_aot_flat_parameter_grads():
    configure_fbw_runtime()

    @torch.compile(backend='aot_eager', fullgraph=True)
    def compiled_region(input_tensor, weight1, weight2):
        return torch.sin(input_tensor * weight1 + weight2)

    class FlatParameterAOTModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.flat = torch.nn.Parameter(torch.randn(16))
            self.region_calls = 0

        def region(self, input_tensor):
            self.region_calls += 1
            return compiled_region(input_tensor, self.flat[:8], self.flat[8:])

        def forward(self, input_tensor):
            return checkpoint(self.region, input_tensor, use_reentrant=False)

    torch.manual_seed(19)
    reference = FlatParameterAOTModule()
    actual = FlatParameterAOTModule()
    actual.load_state_dict(reference.state_dict())
    input_data = torch.randn(4, 8)
    output_grad = torch.randn(4, 8)

    reference_input = input_data.clone().requires_grad_()
    reference(reference_input).backward(output_grad)

    actual_input = input_data.clone().requires_grad_()
    output = Executor.fexecute('flat_parameter_aot', actual, actual_input)
    backward_calls = []
    hook = output.grad_fn.register_hook(
        lambda grad_inputs, grad_outputs: backward_calls.append(None)
    )
    input_grad = Executor.backward_input(
        'flat_parameter_aot',
        [actual_input],
        [output],
        [output_grad],
        actual.parameters(),
    )
    assert len(backward_calls) == 1
    torch.testing.assert_close(actual.flat.grad, reference.flat.grad)
    Executor.backward_weight('flat_parameter_aot', actual.parameters())
    # The cached AOT output follows slice/view edges before reaching the flat
    # parameter. W must replay only those cheap adapter edges, not AOT itself.
    assert len(backward_calls) == 1
    hook.remove()

    assert actual.region_calls == 2
    torch.testing.assert_close(input_grad, reference_input.grad)
    torch.testing.assert_close(actual.flat.grad, reference.flat.grad)
    Executor.check_clear()


def test_opaque_cache_does_not_follow_compiled_dinput_into_prior_layer():
    configure_fbw_runtime()

    @torch.compile(backend='aot_eager', fullgraph=True)
    def compiled_region(input_tensor, weight):
        return torch.sin(input_tensor * weight)

    class PriorLinearAOTModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input_weight = torch.nn.Parameter(torch.randn(8, 8))
            self.output_weight = torch.nn.Parameter(torch.randn(8))

        def forward(self, input_tensor):
            hidden = input_tensor @ self.input_weight
            return compiled_region(hidden, self.output_weight)

    torch.manual_seed(23)
    reference = PriorLinearAOTModule()
    actual = PriorLinearAOTModule()
    actual.load_state_dict(reference.state_dict())
    input_data = torch.randn(4, 8)
    output_grad = torch.randn(4, 8)

    reference_input = input_data.clone().requires_grad_()
    reference(reference_input).backward(output_grad)

    actual_input = input_data.clone().requires_grad_()
    output = Executor.fexecute('prior_linear_aot', actual, actual_input)
    backward_calls = []
    hook = output.grad_fn.register_hook(
        lambda grad_inputs, grad_outputs: backward_calls.append(None)
    )
    input_grad = Executor.backward_input(
        'prior_linear_aot',
        [actual_input],
        [output],
        [output_grad],
        actual.parameters(),
    )
    assert len(backward_calls) == 1
    Executor.backward_weight('prior_linear_aot', actual.parameters())
    assert len(backward_calls) == 1
    hook.remove()

    torch.testing.assert_close(input_grad, reference_input.grad)
    for actual_param, reference_param in zip(
        actual.parameters(), reference.parameters(), strict=True
    ):
        torch.testing.assert_close(actual_param.grad, reference_param.grad)
    Executor.check_clear()


def test_split_backward_uses_fifo_for_multiple_invocations():
    torch.manual_seed(1)
    reference, actual = _make_linears()
    inputs = [
        torch.randn(4, 8, dtype=torch.float64),
        torch.randn(4, 8, dtype=torch.float64),
    ]
    output_grads = [
        torch.randn(4, 16, dtype=torch.float64),
        torch.randn(4, 16, dtype=torch.float64),
    ]

    reference_inputs = [value.clone().requires_grad_() for value in inputs]
    for input_tensor, output_grad in zip(reference_inputs, output_grads):
        reference(input_tensor).backward(output_grad)

    actual_inputs = [value.clone().requires_grad_() for value in inputs]
    outputs = [Executor.fexecute('linear', actual, value) for value in actual_inputs]
    input_grads = [
        Executor.backward_input(
            'linear', [input_tensor], [output], [output_grad], actual.parameters()
        )
        for input_tensor, output, output_grad in zip(actual_inputs, outputs, output_grads)
    ]

    assert actual.weight.grad is None
    assert actual.bias.grad is None
    Executor.backward_weight('linear', actual.parameters())
    Executor.backward_weight('linear', actual.parameters())

    for input_grad, reference_input in zip(input_grads, reference_inputs):
        torch.testing.assert_close(input_grad, reference_input.grad)
    torch.testing.assert_close(actual.weight.grad, reference.weight.grad)
    torch.testing.assert_close(actual.bias.grad, reference.bias.grad)
    Executor.check_clear()


def test_split_backward_without_input_gradient():
    torch.manual_seed(2)
    reference, actual = _make_linears()
    input_tensor = torch.randn(4, 8, dtype=torch.float64)
    output_grad = torch.randn(4, 16, dtype=torch.float64)

    reference(input_tensor).backward(output_grad)

    output = Executor.fexecute('linear', actual, input_tensor)
    input_grad = Executor.backward_input(
        'linear', [], [output], [output_grad], actual.parameters()
    )

    assert input_grad is None
    assert actual.weight.grad is None
    assert actual.bias.grad is None

    Executor.backward_weight('linear', actual.parameters())

    torch.testing.assert_close(actual.weight.grad, reference.weight.grad)
    torch.testing.assert_close(actual.bias.grad, reference.bias.grad)
    Executor.check_clear()


def test_split_backward_with_view_output():
    class ViewLinear(torch.nn.Linear):
        def forward(self, input_tensor):
            return super().forward(input_tensor).view(8, 8)

    torch.manual_seed(3)
    reference = ViewLinear(8, 16, dtype=torch.float64)
    actual = ViewLinear(8, 16, dtype=torch.float64)
    actual.load_state_dict(reference.state_dict())
    input_data = torch.randn(4, 8, dtype=torch.float64)
    output_grad = torch.randn(8, 8, dtype=torch.float64)

    reference_input = input_data.clone().requires_grad_()
    reference(reference_input).backward(output_grad)

    actual_input = input_data.clone().requires_grad_()
    output = Executor.fexecute('view_linear', actual, actual_input)
    input_grad = Executor.backward_input(
        'view_linear', [actual_input], [output], [output_grad], actual.parameters()
    )
    Executor.backward_weight('view_linear', actual.parameters())

    torch.testing.assert_close(input_grad, reference_input.grad)
    torch.testing.assert_close(actual.weight.grad, reference.weight.grad)
    torch.testing.assert_close(actual.bias.grad, reference.bias.grad)
    Executor.check_clear()


def test_split_backward_keeps_custom_autograd_node_alive_for_weight_backward():
    class CustomLinearFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input_tensor, weight):
            ctx.save_for_backward(input_tensor, weight)
            return input_tensor @ weight.t()

        @staticmethod
        def backward(ctx, output_grad):
            input_tensor, weight = ctx.saved_tensors
            input_grad = output_grad @ weight
            weight_grad = output_grad.t() @ input_tensor
            # Model a custom/AOT backward whose incoming gradient is backed by
            # invocation-local workspace and is not stable after it returns.
            output_grad.zero_()
            return input_grad, weight_grad

    class CustomLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(16, 8, dtype=torch.float64))

        def forward(self, input_tensor):
            return CustomLinearFunction.apply(input_tensor, self.weight)

    torch.manual_seed(5)
    reference = CustomLinear()
    actual = CustomLinear()
    actual.load_state_dict(reference.state_dict())
    input_data = torch.randn(4, 8, dtype=torch.float64)
    output_grad = torch.randn(4, 16, dtype=torch.float64)

    reference_input = input_data.clone().requires_grad_()
    reference(reference_input).backward(output_grad.clone())

    actual_input = input_data.clone().requires_grad_()
    output = Executor.fexecute('custom_linear', actual, actual_input)
    input_grad = Executor.backward_input(
        'custom_linear', [actual_input], [output], [output_grad.clone()], actual.parameters()
    )
    # Generated pipeline code releases segment outputs immediately after I.
    # Python custom-autograd Nodes still need their owning output graph at W.
    del output
    gc.collect()
    Executor.backward_weight('custom_linear', actual.parameters())

    torch.testing.assert_close(input_grad, reference_input.grad)
    torch.testing.assert_close(actual.weight.grad, reference.weight.grad)
    Executor.check_clear()


def test_split_backward_releases_custom_autograd_graph_without_full_gc():
    class CustomLinearFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input_tensor, weight):
            ctx.save_for_backward(input_tensor, weight)
            return input_tensor @ weight.t()

        @staticmethod
        def backward(ctx, output_grad):
            input_tensor, weight = ctx.saved_tensors
            return output_grad @ weight, output_grad.t() @ input_tensor

    class CustomLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(16, 8))

        def forward(self, input_tensor):
            return CustomLinearFunction.apply(input_tensor, self.weight)

    module = CustomLinear()
    gc.collect()
    gc.disable()
    try:
        for _ in range(10):
            input_tensor = torch.randn(4, 8, requires_grad=True)
            output = Executor.fexecute('custom_linear', module, input_tensor)
            Executor.backward_input(
                'custom_linear',
                [input_tensor],
                [output],
                [torch.randn_like(output)],
                module.parameters(),
            )
            Executor.backward_weight('custom_linear', module.parameters())
            module.weight.grad = None
        Executor.check_clear()

        # No local recursive closures should keep the reverse autograd graphs
        # alive until a generation-2 collection.
        assert gc.collect() == 0
    finally:
        gc.enable()


def test_weight_backward_triggers_accumulate_grad_hook():
    module = torch.nn.Linear(8, 16)
    input_tensor = torch.randn(4, 8, requires_grad=True)
    output_grad = torch.randn(4, 16)
    param_tmp = module.weight.expand_as(module.weight)
    grad_acc = param_tmp.grad_fn.next_functions[0][0]
    hook_calls = []
    handle = grad_acc.register_hook(lambda *args: hook_calls.append(args))

    output = Executor.fexecute('linear', module, input_tensor)
    Executor.backward_input(
        'linear', [input_tensor], [output], [output_grad], module.parameters()
    )
    assert hook_calls == []

    Executor.backward_weight('linear', module.parameters())

    assert len(hook_calls) == 1
    handle.remove()
    Executor.check_clear()


def test_split_backward_applies_backward_pre_hook_once():
    torch.manual_seed(4)
    reference, actual = _make_linears()
    input_data = torch.randn(4, 8, dtype=torch.float64)
    output_grad = torch.randn(4, 16, dtype=torch.float64)
    hook_calls = []

    reference_input = input_data.clone().requires_grad_()
    reference(reference_input).backward(output_grad * 2)

    def scale_grad(input_tensors, output_tensors, output_grads):
        hook_calls.append(None)
        return input_tensors, output_tensors, [grad * 2 for grad in output_grads]

    Executor.register_backward_pre_hook(scale_grad)
    actual_input = input_data.clone().requires_grad_()
    output = Executor.fexecute('linear', actual, actual_input)
    input_grad = Executor.backward_input(
        'linear', [actual_input], [output], [output_grad], actual.parameters()
    )
    Executor.backward_weight('linear', actual.parameters())

    assert len(hook_calls) == 1
    torch.testing.assert_close(input_grad, reference_input.grad)
    torch.testing.assert_close(actual.weight.grad, reference.weight.grad)
    torch.testing.assert_close(actual.bias.grad, reference.bias.grad)
    Executor.check_clear()


def test_backward_weight_requires_pending_input_backward():
    module = torch.nn.Linear(8, 16)
    with pytest.raises(RuntimeError, match='No pending weight backward'):
        Executor.backward_weight('linear', module.parameters())


def test_configure_fbw_runtime_disables_donated_buffers():
    import torch._functorch.config as functorch_config

    previous = functorch_config.donated_buffer
    try:
        functorch_config.donated_buffer = True
        configure_fbw_runtime()
        assert functorch_config.donated_buffer is False
    finally:
        functorch_config.donated_buffer = previous


@pytest.mark.parametrize("phase", ("weight", "native_weight"))
def test_identity_allreduce_reuses_deferred_weight_grad_storage(monkeypatch, phase):
    calls = []

    def fake_all_reduce(tensor, ranks, async_op=False, clone=True):
        calls.append((tensor, ranks, async_op, clone))
        return tensor if not clone else tensor.clone()

    monkeypatch.setattr(adapter_nn, "all_reduce", fake_all_reduce)
    weight = torch.randn(4, requires_grad=True)
    output = adapter_nn.identity_allreduce(weight, (0, 1))
    output_grad = torch.randn_like(output)
    try:
        RuntimeFlag.fbw_phase = phase
        weight_grad = torch.autograd.grad(output, weight, output_grad)[0]
    finally:
        RuntimeFlag.fbw_phase = None

    assert calls == [(output_grad, (0, 1), False, False)]
    assert weight_grad.data_ptr() == output_grad.data_ptr()


@pytest.mark.parametrize("deterministic", (False, True))
def test_embedding_native_weight_accumulates_chunked_leaf_grad(
    monkeypatch,
    deterministic,
):
    torch.manual_seed(41)
    token_ids = torch.tensor([2, 3, 5, 5, 6, 6, 10, 11])
    output_grad = torch.randn(8, 6, dtype=torch.bfloat16)
    initial_grad = torch.randn(8, 6, dtype=torch.float32)

    reference = torch.nn.Parameter(torch.randn(8, 6, dtype=torch.bfloat16))
    actual = torch.nn.Parameter(reference.detach().clone())
    reference.grad_dtype = torch.float32
    actual.grad_dtype = torch.float32
    reference.grad = initial_grad.clone()
    actual.grad = initial_grad.clone()

    reference_output = runtime_function.embedding(
        token_ids,
        reference,
        padding_idx=5,
        start=3,
        stop=11,
    )
    reference_output.backward(output_grad)

    actual_output = runtime_function.embedding(
        token_ids,
        actual,
        padding_idx=5,
        start=3,
        stop=11,
    )
    monkeypatch.setattr(
        runtime_function,
        "_embedding_dense_backward",
        lambda *args, **kwargs: pytest.fail(
            "native_weight must not allocate a vocabulary-sized dWeight"
        ),
    )
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(deterministic)
        RuntimeFlag.fbw_phase = "native_weight"
        actual_output.backward(output_grad)
    finally:
        RuntimeFlag.fbw_phase = None
        torch.use_deterministic_algorithms(previous_deterministic)

    assert actual.grad.dtype == torch.float32
    torch.testing.assert_close(actual.grad, reference.grad, rtol=0, atol=0)
