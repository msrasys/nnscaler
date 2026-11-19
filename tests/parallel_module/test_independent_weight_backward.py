import copy
import pytest
import torch
import nnscaler
from nnscaler import ComputeConfig, ModuleStreamRule
from nnscaler.parallel import _load_parallel_module_class
from nnscaler.runtime.weight_backward import WeightBackwardRegion
from .test_weight_backward import weight_anchor, combine_anchor, weight_linear, Streams
from ..utils import replace_all_device_with, new_empty


class Shared(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(4, 4))

    def forward(self, x):
        return weight_linear(x, self.weight)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = Shared()

    def forward(self, x):
        routed = combine_anchor(weight_anchor(x).sin())
        return routed + self.shared(x)


class IndependentStreams(Streams):
    def weight_region(self, key, name, parameters, *, independent=False):
        region = WeightBackwardRegion(name, parameters, independent=independent)
        self.weights.append(region)
        return region


@replace_all_device_with("cpu", force=True)
@pytest.mark.parametrize("workers", [1, 2])
def test_compiled_independent_shared_weights_preserve_math(
    tmp_path, monkeypatch, workers
):
    monkeypatch.setattr("nnscaler.runtime.stream.ModuleStreams", IndependentStreams)
    model = Model()
    reference = copy.deepcopy(model)
    rules = [
        ModuleStreamRule(
            "shared",
            stream="shared_compute",
            launch_before="*.combine_anchor",
            overlap_backward=True,
        ),
        ModuleStreamRule(
            "shared",
            stream="shared_weights",
            weight_backward_on="*.weight_anchor",
            weight_backward_independent=True,
        ),
    ]
    name = f"independent_shared_{workers}"
    nnscaler.parallelize(
        model,
        {"x": torch.randn(8, 4, requires_grad=True)},
        "dp",
        ComputeConfig(1, 1, module_streams=rules),
        gen_savedir=tmp_path,
        instance_name=name,
        load_module=False,
        codegen_workers=workers,
    )
    source = next(tmp_path.rglob("gencode0.py")).read_text()
    assert (
        ".capture_output(" in source
        and ".order_inputs(" in source
        and "independent=True" in source
    )
    cls = _load_parallel_module_class(
        Model, gen_savedir=tmp_path, instance_name=name, rank=0
    )
    assert set(cls.module_parameter_streams.values()) == {"shared_weights"}
    generated = new_empty(cls, device="cpu", init_params=True)
    for _ in range(2):
        for p in (*generated.parameters(), *reference.parameters()):
            p.grad = None
        x = torch.randn(8, 4, requires_grad=True)
        rx = x.detach().clone().requires_grad_()
        y, expected = generated(x), reference(rx)
        y.square().sum().backward()
        expected.square().sum().backward()
        torch.testing.assert_close(y, expected, rtol=0, atol=0)
        torch.testing.assert_close(x.grad, rx.grad, rtol=0, atol=0)
        for attr, param in generated.named_parameters():
            torch.testing.assert_close(
                param.grad,
                reference.get_parameter(generated.fullmap[attr].orig_name).grad,
                rtol=0,
                atol=0,
            )


@replace_all_device_with("cpu", force=True)
@pytest.mark.parametrize("native", [False, True])
def test_independent_requires_whole_eager_region(tmp_path, native):
    rules = [
        ModuleStreamRule(
            "shared",
            weight_backward_on="*.weight_anchor",
            weight_backward_independent=True,
        )
    ]
    if native:
        rules.insert(
            0,
            ModuleStreamRule(
                "shared", launch_on="*.combine_anchor", launch_phase="input_ready"
            ),
        )
    with pytest.raises(RuntimeError, match="Code generation failed") as failure:
        nnscaler.parallelize(
            Model(),
            {"x": torch.randn(8, 4, requires_grad=True)},
            "dp",
            ComputeConfig(1, 1, module_streams=rules),
            gen_savedir=tmp_path,
            load_module=False,
        )
    assert isinstance(failure.value.__cause__, ValueError)
    assert "whole eager" in str(failure.value.__cause__)


@replace_all_device_with("cpu", force=True)
def test_default_rule_still_rejects_an_independent_anchor(tmp_path):
    rule = ModuleStreamRule("shared", weight_backward_on="*.weight_anchor")
    with pytest.raises(RuntimeError, match="Code generation failed") as failure:
        nnscaler.parallelize(
            Model(),
            {"x": torch.randn(8, 4, requires_grad=True)},
            "dp",
            ComputeConfig(1, 1, module_streams=[rule]),
            gen_savedir=tmp_path,
            load_module=False,
        )
    assert isinstance(failure.value.__cause__, ValueError)
    assert "differentiable data dependency" in str(failure.value.__cause__)


def test_independent_option_requires_weight_rule():
    with pytest.raises(ValueError, match="requires weight_backward_on"):
        ModuleStreamRule("shared", weight_backward_independent=True)


@replace_all_device_with("cpu", force=True)
def test_independent_codegen_in_multiple_workers(tmp_path):
    rules = [
        ModuleStreamRule(
            "shared",
            stream="shared_compute",
            launch_before="*.combine_anchor",
            overlap_backward=True,
        ),
        ModuleStreamRule(
            "shared",
            stream="shared_weights",
            weight_backward_on="*.weight_anchor",
            weight_backward_independent=True,
        ),
    ]
    nnscaler.parallelize(
        Model(),
        {"x": torch.randn(8, 4, requires_grad=True)},
        "dp",
        ComputeConfig(1, 2, module_streams=rules),
        gen_savedir=tmp_path,
        instance_name="independent_workers",
        load_module=False,
        codegen_workers=2,
    )
    sources = list(tmp_path.rglob("gencode*.py"))
    assert len(sources) == 2
    for path in sources:
        source = path.read_text()
        assert ".capture_output(" in source and ".order_inputs(" in source
