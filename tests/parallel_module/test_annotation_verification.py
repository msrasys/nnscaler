import importlib

import pytest
import torch

import nnscaler
from nnscaler.graph import IRGraph
from nnscaler.graph.parser.register import CustomizedOps
from nnscaler.ir.tensor import IRFullTensor

from ..utils import replace_all_device_with


parallel_module = importlib.import_module("nnscaler.parallel")


class ReLUModel(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x)


@nnscaler.register_op("l n -> l n")
def verification_scale(x):
    return x * 2


class VerifiedModel(torch.nn.Module):
    def forward(self, x):
        return verification_scale(x)


@replace_all_device_with("cpu")
def test_parallelize_static_annotation_verification(tmp_path):
    nnscaler.parallelize(
        ReLUModel(),
        {"x": torch.randn(4, 8)},
        "dp",
        nnscaler.ComputeConfig(1, 1),
        gen_savedir=tmp_path,
        reuse="override",
        load_module=False,
        verify_annotations="static",
    )


def _registered_op_graph():
    tensor = IRFullTensor(
        (4, 8), dtype=torch.float32, requires_grad=True,
    ).tosub()
    signature = f"{verification_scale.__module__}.{verification_scale.__name__}"
    op = CustomizedOps.map(signature)(tensor)
    output = IRFullTensor(
        (4, 8), dtype=torch.float32, requires_grad=True,
    ).tosub()
    op.set_output(0, output)
    return IRGraph([op], [tensor], [output], "VerificationModel"), op


def test_all_verification_runs_before_policy(tmp_path, monkeypatch):
    graph, _ = _registered_op_graph()
    calls = []

    def verify(graph, outdir, partition_options=None):
        calls.append(("verify", partition_options))
        return True

    def policy(graph, compute_config):
        calls.append(("policy", None))
        return graph

    monkeypatch.setattr(parallel_module, "verify_op_partitions", verify)
    result = parallel_module._apply_policy_with_annotation_verification(
        graph,
        policy,
        nnscaler.ComputeConfig(2, 2),
        nnscaler.AnnotationVerification.ALL,
        tmp_path,
    )

    assert result is graph
    assert calls == [("verify", None), ("policy", None)]


def test_used_verification_records_policy_partition(tmp_path, monkeypatch):
    graph, op = _registered_op_graph()
    captured = []

    def verify(graph, outdir, partition_options=None):
        captured.extend(partition_options)
        return True

    def policy(graph, compute_config):
        subnodes = graph.partition(
            op,
            op.algorithm("dim"),
            idx=0,
            dim=0,
            num=2,
        )
        for rank, subnode in enumerate(subnodes):
            graph.assign(subnode, rank)
        return graph

    monkeypatch.setattr(parallel_module, "verify_op_partitions", verify)
    result = parallel_module._apply_policy_with_annotation_verification(
        graph,
        policy,
        nnscaler.ComputeConfig(2, 2),
        nnscaler.AnnotationVerification.USED,
        tmp_path,
    )

    assert result is graph
    assert captured == [(op, [{"idx": 0, "dim": 0, "num": 2}])]


def test_invalid_annotation_verification_mode():
    with pytest.raises(ValueError, match="not a valid AnnotationVerification"):
        nnscaler.parallelize(
            ReLUModel(),
            {"x": torch.randn(4, 8)},
            "dp",
            nnscaler.ComputeConfig(1, 1),
            load_module=False,
            verify_annotations="invalid",
        )


def test_dynamic_verification_rejects_initialized_distributed(monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    with pytest.raises(RuntimeError, match="before initializing"):
        nnscaler.parallelize(
            ReLUModel(),
            {"x": torch.randn(4, 8)},
            "dp",
            nnscaler.ComputeConfig(1, 1),
            load_module=False,
            verify_annotations="all",
        )


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="requires two GPUs",
)
@pytest.mark.parametrize(
    ("mode", "policy", "plan_ngpus"),
    [
        ("all", "dp", 1),
        ("used", "tp", 2),
    ],
)
def test_parallelize_dynamic_annotation_verification(
    tmp_path, mode, policy, plan_ngpus,
):
    nnscaler.parallelize(
        VerifiedModel(),
        {"x": torch.randn(4, 8, requires_grad=True)},
        policy,
        nnscaler.ComputeConfig(plan_ngpus, plan_ngpus),
        gen_savedir=tmp_path,
        instance_name=mode,
        reuse="override",
        load_module=False,
        verify_annotations=mode,
    )
