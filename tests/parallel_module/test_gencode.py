import inspect
import tempfile

import torch
import pytest

import cube.graph.function.dimops
from cube.parallel import parallelize, ComputeConfig, CubeModule, _gen_graph

from .common import PASData, init_distributed, PASRandomSPMD
from ..launch_torchrun import launch_torchrun
from ..utils import replace_all_device_with

def _to_cube_model(module, compute_config, cube_savedir, load_module):
    return parallelize(
        module,
        {'x': torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])},
        PASData,
        compute_config,
        cube_savedir=cube_savedir,
        load_module=load_module
    )

class Module0(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 5)

    def forward(self, x):
        return self.linear(x)


def _gencode_worker(tempdir):
    init_distributed()
    m = Module0()
    with pytest.raises(RuntimeError):  # config mismatch
        _to_cube_model(m, ComputeConfig(1, 1), cube_savedir=tempdir, load_module=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='lack of gpu devices')
def test_codegen():
    with tempfile.TemporaryDirectory() as tempdir:
        m = Module0()
        m_new = _to_cube_model(m, ComputeConfig(2, 4), cube_savedir=tempdir, load_module=False)
        assert m_new is None
        launch_torchrun(1, _gencode_worker, tempdir)


class SliceModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x[:2]

@replace_all_device_with('cpu')
def test_codegen_slice():
    with tempfile.TemporaryDirectory() as tempdir:
        m_new = parallelize(
            SliceModule(),
            {'x': torch.tensor([1.0, 2.0, 3.0, 6.0])},
            PASData,
            ComputeConfig(1, 1),
            cube_savedir=tempdir,
            load_module=False
        )
        assert m_new is None


class ArgsModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 5)

    def forward(self, x, y, *args):
        return self.linear(x) + y


@replace_all_device_with('cpu')
def test_codegen_args():
    with tempfile.TemporaryDirectory() as tempdir:
        # *args is not supported.
        with pytest.raises(RuntimeError):
            parallelize(
                ArgsModule(),
                {
                    'x': torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
                    'y': 1.0,
                },
                PASData,
                ComputeConfig(1, 1),
                cube_savedir=tempdir,
                load_module=True
            )


class UnusedArgsModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 5)

    def forward(self, x, y, z=None, m=1, n=2, **kwargs):
        return self.linear(x) + m


def _gencode_unused_args_worker(tempdir):
    init_distributed()
    m_new = parallelize(
        UnusedArgsModule(),
        {
            'x': torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            'y': torch.tensor([1, 2, 3]),
            'z': None,
            'm': 0,
            'n': None,
         },
        PASData,
        ComputeConfig(1, 1),
        cube_savedir=tempdir,
        load_module=True
    )
    assert m_new is not None
    args = inspect.signature(m_new._forward_impl)
    assert len(args.parameters) == 6
    assert args.parameters['x'].default is inspect.Parameter.empty
    assert args.parameters['y'].default is None
    assert args.parameters['z'].default is None
    assert args.parameters['m'].default == 1
    assert args.parameters['n'].default is None
    assert args.parameters['kwargs'].default is inspect.Parameter.empty

    assert torch.equal(
        m_new(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])),
        m_new(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), m=1)
    )

    with pytest.raises(ValueError):
        # y must be None
        m_new(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='lack of gpu devices')
def test_codegen_unused_args():
    """
    Verify that unused args are supported by parallalize
    """
    with tempfile.TemporaryDirectory() as tempdir:
        launch_torchrun(1, _gencode_unused_args_worker, tempdir)


class UnusedArgs2Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 5)

    def forward(self, x, y, m):
        return self.linear(x) + m


def _gencode_unused_args_worker2(tempdir):
    init_distributed()
    m_new = parallelize(
        UnusedArgs2Module(),
        {
            'x': torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            'y': torch.tensor([1, 2, 3]),
            'm': 0
         },
        PASData,
        ComputeConfig(1, 1),
        cube_savedir=tempdir,
        load_module=True
    )
    assert m_new is not None
    args = inspect.signature(m_new._forward_impl)
    assert len(args.parameters) == 3
    assert args.parameters['x'].default is inspect.Parameter.empty
    assert args.parameters['y'].default is None
    assert args.parameters['m'].default is None

    m_new(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), m=1)
    with pytest.raises(TypeError, match='.*must be Tensor, not NoneType.*'):
        # raise by torch.add, as m is None
        m_new(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    with pytest.raises(ValueError):
        # y must be None
        m_new(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='lack of gpu devices')
def test_codegen_unused_args2():
    """
    Verify that unused args are supported by parallalize
    """
    with tempfile.TemporaryDirectory() as tempdir:
        launch_torchrun(1, _gencode_unused_args_worker2, tempdir)


class AttrModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, attr):
        return x + getattr(attr, 'a')


def _gencode_contains(cubesave_dir, module_class, index, search_re):
    from cube.parallel import _CUBE_MODULE_NAMESPACE, _get_full_qualified_name
    from pathlib import Path
    import re
    namespace = f'{_CUBE_MODULE_NAMESPACE}.{_get_full_qualified_name(module_class)}'
    outdir: Path = cubesave_dir / Path(namespace.replace('.', '/').strip('/'))
    filecontent = (outdir /f'gencode{index}.py').read_text()
    matches = re.findall(search_re, filecontent)
    return bool(matches)

class AttrHelper:
    def __init__(self) -> None:
        self.a = 2.0


@replace_all_device_with('cpu')
def test_codegen_attr():
    with tempfile.TemporaryDirectory() as tempdir:
        m_new = parallelize(
            AttrModule(),
            {'x': torch.tensor([1.0, 2.0, 3.0, 6.0]), 'attr': AttrHelper()},
            PASData,
            ComputeConfig(1, 1),
            cube_savedir=tempdir,
            load_module=False
        )
        # in old version, all 'forward' functions will patched to a function named 'new_func'
        assert not _gencode_contains(tempdir, AttrModule, 0, r'new_func')
        assert _gencode_contains(tempdir, AttrModule, 0, r'builtins.getattr\(.*, \'a\'\)')
        assert m_new is None


class GetItemModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batched_data):
        data_x = batched_data["x"]
        n_graph, n_node = data_x.size()[:2]
        padding_mask = (data_x[:, :, 0]).eq(0)  # B x T x 1
        padding_mask_cls = torch.zeros(
            n_graph, 1, device=padding_mask.device, dtype=padding_mask.dtype
        )
        padding_mask = torch.cat((padding_mask_cls, padding_mask), dim=1)
        return padding_mask


@replace_all_device_with('cpu')
def test_codegen_getitem():
    with tempfile.TemporaryDirectory() as tempdir:
        m_new = parallelize(
            GetItemModule(),
            {'batched_data': {'x': torch.tensor([[[1.0], [2.0], [3.0], [6.0]]])}},
            PASRandomSPMD,
            ComputeConfig(2, 2),
            cube_savedir=tempdir,
            load_module=False,
        )
        assert _gencode_contains(tempdir, GetItemModule, 0, r'_operator.getitem\(.*, slice\(None, 2, None\)\)')
        assert _gencode_contains(tempdir, GetItemModule, 1, r'_operator.getitem\(.*, slice\(None, 2, None\)\)')
        assert m_new is None


class TrainingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 5)

    def forward(self, x):
        if self.training:
            return self.linear(x)
        else:
            return self.linear(x) + 1


@replace_all_device_with('cpu')
def test_codegen_training_flag():
    with tempfile.TemporaryDirectory() as tempdir:
        m = TrainingModule()
        m.train()
        parallelize(
            m,
            {'x': torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])},
            PASData,
            ComputeConfig(1, 1),
            cube_savedir=tempdir,
            load_module=False
        )


# class IdentityModule(torch.nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, x):
#         return x


# def test_codegen_identity():
#     """
#     Test it can support modules without parameters
#     """
#     if not torch.cuda.is_available():
#         print('skip test_codegen_iter due to lack of cuda devices')
#         return
#     with tempfile.TemporaryDirectory() as tempdir:
#         m = IdentityModule()
#         m.train()
#         parallelize(
#             m,
#             {'x': torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])},
#             PASData,
#             ComputeConfig(1, 2),
#             cube_savedir=tempdir,
#             load_module=False
#         )
#         assert False


class IterModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 5)

    def forward(self, x):
        x = self.linear(x)
        assert list(x.shape) == [2, 5]  # will generate iter here.
        return x


@replace_all_device_with('cpu')
def test_codegen_iter():
    """
    Test it can support modules without parameters
    """
    with tempfile.TemporaryDirectory() as tempdir:
        m = IterModule()
        m.train()
        # assert no exception raised below
        parallelize(
            m,
            {'x': torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])},
            PASData,
            ComputeConfig(1, 1),
            cube_savedir=tempdir,
            load_module=False
        )


class ConstantModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 5)

    def forward(self, x):
        x = self.linear(x)
        y = int(x.shape[-1])
        x = x[:, :y]
        return x


@replace_all_device_with('cpu')
def test_codegen_const():
    """
    Test it can support modules without parameters
    """
    with tempfile.TemporaryDirectory() as tempdir:
        m = ConstantModule()
        m.train()
        parallelize(
            m,
            {'x': torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])},
            PASData,
            ComputeConfig(1, 1),
            cube_savedir=tempdir,
            load_module=False
        )
        assert not _gencode_contains(tempdir, ConstantModule, 0, r'\s+5 = builtins.int')


class TensorSliceModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 5)

    def forward(self, x):
        x = self.linear(x)
        padding = torch.count_nonzero(x)
        return x[:, :padding]


class TensorSliceFixedModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 5)

    def forward(self, x):
        x = self.linear(x)
        padding = torch.count_nonzero(x).item()
        return x[:, :padding]


@replace_all_device_with('cpu')
def test_codegen_tensor_slice():
    with tempfile.TemporaryDirectory() as tempdir:
        m = TensorSliceModule()
        m.train()
        with pytest.raises(RuntimeError, match='Tensor is not supported in slice.'):
            parallelize(
                m,
                {'x': torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])},
                PASData,
                ComputeConfig(1, 1),
                cube_savedir=tempdir,
                load_module=False,
                reuse='none',
            )
        m = TensorSliceFixedModule()
        m.train()
        parallelize(
            m,
            {'x': torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])},
            PASData,
            ComputeConfig(1, 1),
            cube_savedir=tempdir,
            load_module=False,
            reuse='none',
        )


class DictGetModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batched_data: dict):
        data_x = batched_data["x"]
        data_y = batched_data.get("y", batched_data['z'])
        return data_x + data_y


@replace_all_device_with('cpu')
def test_codegen_dictget():
    with tempfile.TemporaryDirectory() as tempdir:
        m_new = parallelize(
            DictGetModule(),
            {'batched_data': {
                'x': torch.tensor([[[1.0], [2.0], [3.0], [6.0]]]),
                'z': torch.tensor([[[1.0], [2.0], [3.0], [6.0]]])
            }},
            PASRandomSPMD,
            ComputeConfig(2, 2),
            cube_savedir=tempdir,
            load_module=False,
        )
        assert _gencode_contains(tempdir, DictGetModule, 0, r"dict.get\(\w+, 'y', \w+\)")
        assert _gencode_contains(tempdir, DictGetModule, 1, r"dict.get\(\w+, 'y', \w+\)")
        assert m_new is None


class CloneModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.clone()


@replace_all_device_with('cpu')
def test_codegen_clone():
    with tempfile.TemporaryDirectory() as tempdir:
        g, _ = _gen_graph(
            CloneModule(),
            {'x': torch.tensor([1.0, 2.0, 3.0, 6.0])},
            tempdir,
            True
        )
        assert isinstance(g.nodes()[0], cube.graph.function.dimops.IRDimops)
