import cube
from cube.graph.parser.converter import convert_model
from cube.profiler.database import ProfileDataBase
import tempfile
import torch


def mock_add(x: torch.Tensor, y: torch.Tensor):
    return x + y

cube.graph.parser.register('*, * -> *')(mock_add)


@cube.graph.parser.register('*, * -> *')
def mock_add2(x: torch.Tensor, y: torch.Tensor):
    return x + y


class MockAGF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, y: torch.Tensor):
        return x + y

    @staticmethod
    def backward(ctx, grad):
        return grad, grad

cube.graph.parser.register('*, * -> *')(MockAGF.apply)


class TestModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(10, 10)

    def forward(self, x, y):
        x, y = self.fc(x), self.fc(y)
        return mock_add(x, y)

class TestModel2(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(10, 10)

    def forward(self, x, y):
        x, y = self.fc(x), self.fc(y)
        return mock_add2(x, y)

class TestModel3(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(10, 10)

    def forward(self, x, y):
        x, y = self.fc(x), self.fc(y)
        return MockAGF.apply(x, y)


# passed test
def test_common_register():
    model = TestModel()
    with tempfile.TemporaryDirectory() as tempdir:
        ir_graph = convert_model(model, {'x': torch.rand(10, 10), 'y': torch.rand(10, 10)}, tempdir, False)

        # test profiler.database
        for node, p_name in zip(ir_graph.nodes(), ['linear', 'linear', 'mock_add']):
            profile_name = ProfileDataBase.get_func(node)[0].__qualname__
            assert profile_name == p_name, f'{profile_name} should be {p_name}'


def test_common_register2():
    model = TestModel2()
    with tempfile.TemporaryDirectory() as tempdir:
        ir_graph = convert_model(model, {'x': torch.rand(10, 10), 'y': torch.rand(10, 10)}, tempdir, False)

        # test profiler.database
        for node, p_name in zip(ir_graph.nodes(), ['linear', 'linear', 'mock_add2']):
            profile_name = ProfileDataBase.get_func(node)[0].__qualname__
            assert profile_name == p_name, f'{profile_name} should be {p_name}'


def test_autograd_register():
    model = TestModel3()
    with tempfile.TemporaryDirectory() as tempdir:
        ir_graph = convert_model(model, {'x': torch.rand(10, 10), 'y': torch.rand(10, 10)}, tempdir, False)
        
        # test profiler.database
        for node, p_name in zip(ir_graph.nodes(), ['linear', 'linear', 'MockAGF.apply']):
            profile_name = ProfileDataBase.get_func(node)[0].__qualname__
            assert profile_name == p_name, f'{profile_name} should be {p_name}'
