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
import nnscaler.graph.verification as graph_verification
import nnscaler.graph.verification_runner as verification_runner
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


def test_graph_verifier_imports_autograd_function_owner():
    import tests.graph.parser.test_register as register_tests

    signature = (
        f"{register_tests.MockAGF.__module__}."
        f"{register_tests.MockAGF.__qualname__}.apply"
    )
    import_code, _, _ = verify_graph_operations._verification_code(signature)
    assert f"import {register_tests.MockAGF.__module__}" in import_code


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
        verification_runner,
        "verify_partition_options",
        lambda config: False,
    )
    assert not verify_graph_operations.verify_op_partitions(Graph(), tmp_path)


def test_graph_verifier_does_not_cache_empty_partition_set(
    tmp_path, monkeypatch,
):
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
        graph_verification,
        "get_candidate_options",
        lambda annotation, infos: [],
    )
    assert verify_graph_operations.verify_op_partitions(Graph(), tmp_path)
    assert not (tmp_path / "verified_ops.pt").exists()


def test_graph_verifier_cache_tracks_implementation(
    tmp_path, monkeypatch,
):
    tensor = IRFullTensor(
        (4, 8), dtype=torch.float32, requires_grad=True,
    ).tosub()
    signature = "verification_test_ops.stateless_scale"
    op = CustomizedOps.map(signature)(tensor)
    op.set_output(
        0,
        IRFullTensor(
            (4, 8), dtype=torch.float32, requires_grad=True,
        ).tosub(),
    )

    class Graph:
        def nodes(self, flatten=True):
            return [op]

    calls = 0

    def verify(config):
        nonlocal calls
        calls += 1
        return True

    monkeypatch.setattr(verification_runner, "verify_partition_options", verify)
    assert verify_graph_operations.verify_op_partitions(Graph(), tmp_path)
    assert verify_graph_operations.verify_op_partitions(Graph(), tmp_path)
    assert calls == 1

    old_code = CustomizedOps.kOpCodeDef[signature]
    monkeypatch.setitem(
        CustomizedOps.kOpCodeDef,
        signature,
        old_code + "\n# implementation changed",
    )
    assert verify_graph_operations.verify_op_partitions(Graph(), tmp_path)
    assert calls == 2


def test_verification_subprocess_inherits_python_import_paths(
    tmp_path, monkeypatch,
):
    module_dir = tmp_path / "module"
    module_dir.mkdir()
    (module_dir / "local_verification_module.py").write_text("VALUE = 1\n")
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.syspath_prepend(str(module_dir))

    verification_runner._run_checked(
        [
            sys.executable,
            "-c",
            "import local_verification_module; "
            "assert local_verification_module.VALUE == 1",
        ],
        cwd=workdir,
    )


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
