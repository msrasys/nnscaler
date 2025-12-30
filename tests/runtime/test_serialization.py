import torch
import pytest

from nnscaler.runtime.serialization import load, save, convert
from nnscaler.cli.serialization import convert_format

from tests.parallel_module.common import assert_equal


def test_normal(tmp_path):
    a = torch.randn((2, 2), device='cpu')
    b = torch.randn((2, 3), device='cpu')
    c = torch.randn((4, 4), device='cpu')
    tensors = {
        "embedding": a,
        "attention": b,
        "fc": a,  # shared tensor
        "bias": {'inner': b, 'outer': {'deep': c}}
    }
    save(tensors, tmp_path / "model.safetensors")
    loaded = load(tmp_path / "model.safetensors", lazy=False)
    assert_equal(tensors, loaded)
    convert(tmp_path / "model.safetensors", tmp_path / "model.pt")
    convert_format(
        src=str(tmp_path / "model.safetensors"),
        dst=str(tmp_path / "model2.ckpt"),
    )
    loaded_pt = torch.load(tmp_path / "model.pt")
    assert_equal(tensors, loaded_pt)
    loaded_pt2 = torch.load(tmp_path / "model2.ckpt")
    assert_equal(tensors, loaded_pt2)


def test_shared_params(tmp_path):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(4, 4)
            self.fc2 = torch.nn.Linear(4, 4)
            # share the same weight
            self.fc2.weight = self.fc1.weight

        def forward(self, x):
            x = self.fc1(x)
            x = self.fc2(x)
            return x

    model = Model()
    save(model.state_dict(), tmp_path / "model.safetensors")
    loaded = load(tmp_path / "model.safetensors", lazy=False)
    assert_equal(model.state_dict(), loaded)
    convert(tmp_path / "model.safetensors", tmp_path / "model.pt")
    loaded_pt = torch.load(tmp_path / "model.pt")
    assert_equal(model.state_dict(), loaded_pt)


def test_bad_shared_params(tmp_path):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(4, 4)
            self.fc2 = torch.nn.Linear(4, 4)
            # share the same weight
            # This case is not common,
            # so we don't support it currently.
            self.fc2.weight.data = self.fc1.weight.reshape(-1)

        def forward(self, x):
            x = self.fc1(x)
            x = self.fc2(x)
            return x

    model = Model()
    with pytest.raises(RuntimeError):
        save(model.state_dict(), tmp_path / "model.safetensors")
