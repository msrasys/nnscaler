import sys
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
VERIFY_OPS_DIR = REPO_ROOT / "utility" / "verify_ops"
TEST_OPS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(VERIFY_OPS_DIR))
sys.path.insert(0, str(TEST_OPS_DIR))

import verification_test_ops
import verify_dimops
import verify_graph_operations
from verify_dimops import TensorInfo, VerifyConfig, verify_partition_options
from nnscaler.graph.parser.register import CustomizedOps
from nnscaler.ir.tensor import IRFullTensor


def _config(function_name, *, stateful):
    module_name = verification_test_ops.__name__
    return VerifyConfig(
        fsig=f"{module_name}.{function_name}",
        args=[TensorInfo("shape", (4, 8), torch.float32, True)],
        kwargs={},
        noutputs=1,
        parti_options=[{"idx": 0, "dim": 0}],
        import_customized_func=(
            f"import sys\n"
            f"sys.path.insert(0, {str(TEST_OPS_DIR)!r})\n"
            f"import {module_name}"
        ),
        setup_call=(
            f"{module_name}.setup_runtime_state()" if stateful else ""
        ),
        state_call=(
            f"{module_name}.snapshot_runtime_state()"
            if stateful else "None"
        ),
    )


def test_graph_verifier_loads_registered_lifecycle_callbacks():
    import_code, setup_call, state_call = (
        verify_graph_operations._verification_code(
            "verification_test_ops.stateful_identity"
        )
    )
    assert "import verification_test_ops" in import_code
    assert setup_call == "verification_test_ops.setup_runtime_state()"
    assert state_call == "verification_test_ops.snapshot_runtime_state()"


def test_graph_verifier_fails_closed(tmp_path, monkeypatch):
    tensor = IRFullTensor(
        (4, 8), dtype=torch.float32, requires_grad=True,
    ).tosub()
    op = CustomizedOps.map("verification_test_ops.stateless_scale")(tensor)
    op.set_output(
        0,
        IRFullTensor(
            (4, 8), dtype=torch.float32, requires_grad=True,
        ).tosub(),
    )

    class Graph:
        def nodes(self, flatten=True):
            return [op]

    monkeypatch.setattr(
        verify_dimops,
        "verify_partition_options",
        lambda config: False,
    )
    assert not verify_graph_operations.verify_op_partitions(Graph(), tmp_path)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="requires two GPUs",
)
def test_verifier_rejects_local_shape_normalization(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert not verify_partition_options(
        _config("locally_normalized_sum", stateful=True)
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="requires two GPUs",
)
def test_verifier_rejects_partitioned_replicated_state(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert not verify_partition_options(
        _config("stateful_identity", stateful=True)
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="requires two GPUs",
)
def test_verifier_compares_real_leaf_gradients(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert verify_partition_options(
        _config("stateless_scale", stateful=False)
    )
    result = torch.load(
        "verification_test_ops.stateless_scale_loss_single.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert result["gradients"][0] is not None
