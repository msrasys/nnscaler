import tempfile
import torch
from nnscaler.graph.parser.converter import convert_model

from ...utils import replace_all_device_with


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(10, 10)

    def forward(self, x):
        with torch.no_grad():
            y = self.fc(x)
        z = self.fc(x)
        return y + z


@replace_all_device_with('cpu')
def test_requires_grad():
    with tempfile.TemporaryDirectory() as tempdir:
        model = SimpleModel()
        dummy_input = {'x': torch.rand(10)}
        graph = convert_model(model, dummy_input, tempdir)

    node_no_grad_fc, node_fc, node_add = graph.nodes()
    # x under no grad context
    assert node_no_grad_fc.inputs()[0].parent.requires_grad is False
    # fc weight under no grad context
    assert node_no_grad_fc.inputs()[1].parent.requires_grad is True
    # fc output under no grad context
    assert node_no_grad_fc.outputs()[0].parent.requires_grad is False
    # x outside no grad context
    assert node_fc.inputs()[0].parent.requires_grad is False
    # fc weight outside no grad context
    assert node_fc.inputs()[1].parent.requires_grad is True
    # fc output outside no grad context
    assert node_fc.outputs()[0].parent.requires_grad is True
    # y
    assert node_add.inputs()[0].parent.requires_grad is False
    # z
    assert node_add.inputs()[1].parent.requires_grad is True
    # result
    assert node_add.outputs()[0].parent.requires_grad is True
