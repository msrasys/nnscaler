#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from pathlib import Path
from time import sleep
import sys
import tempfile
import pickle
import random
import pytest
import torch
import shutil
from unittest.mock import patch

from nnscaler.graph.parser import FxModuleParser
from nnscaler.graph.parser.frame import Frame
from nnscaler.parallel import ReuseType, parallelize, ComputeConfig, ParamInitStrategy, _load_parallel_module_class
from nnscaler.runtime.module import ParallelModule

from ..utils import new_empty, replace_all_device_with, raises_with_cause


def _to_cube_model(model_class, compute_config, cube_savedir, reuse, instance_name, load_module=True):
    parallelize(
        model_class,
        {'x': torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])},
        'data',
        compute_config,
        reuse=reuse,
        gen_savedir=cube_savedir,
        instance_name=instance_name,
        load_module=False,
    )
    if load_module:
        module_class = _load_parallel_module_class(
            model_class,
            gen_savedir=cube_savedir,
            instance_name=instance_name,
            rank=0
        )
        m = new_empty(module_class, device='cpu', init_params=True)
        return m


class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 5)

    def forward(self, x):
        return self.linear(x)

    @staticmethod
    def __partial__init__(attr_meta_map):
        return {
            name: torch.zeros(meta.sub_shape, dtype=meta.dtype)
            for name, meta in attr_meta_map.items()
        }


class SeedBufferModule(MyModule):
    def __init__(self):
        super().__init__()
        self.register_buffer('offset', torch.rand(1), persistent=False)

    def forward(self, x):
        return self.linear(x) + self.offset


class SeedShapeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(random.choice([2, 3])))

    def forward(self, x):
        return x * self.weight


@patch('torch.cuda.is_available', lambda: False)
@pytest.mark.parametrize('reuse', [ReuseType.MOO, ReuseType.GRAPH])
@pytest.mark.parametrize('strategy', ['model', 'capture'])
def test_param_init_seed_retraces_constructor_structure(tmp_path, reuse, strategy):
    from ..utils import mock_cube_env, mock_dist

    instance_name = f'{strategy}_{reuse.value}'
    with mock_cube_env(0, 1), mock_dist(0, 1), patch('torch.distributed.barrier'), \
            patch('torch.distributed.broadcast_object_list'):
        parallelize(
            SeedShapeModule, {'x': torch.ones(1)}, 'dp',
            ComputeConfig(1, 1, param_init_strategy=strategy, param_init_seed=1, trace_strategy='cpu'),
            gen_savedir=tmp_path, instance_name=instance_name, load_module=False,
        )
        generated = parallelize(
            SeedShapeModule, {'x': torch.ones(1)}, 'dp',
            ComputeConfig(1, 1, param_init_strategy=strategy, param_init_seed=5, trace_strategy='cpu'),
            gen_savedir=tmp_path, instance_name=instance_name, reuse=reuse,
        )
        model = generated(build_buckets=False)
    assert len(model.fullmap) == 1
    for attr, meta in model.fullmap.items():
        assert meta.shape == (3,)
        assert torch.equal(getattr(model, attr), torch.ones(3))


@patch('torch.cuda.is_available', lambda: False)
@replace_all_device_with('cpu', force=True)
@pytest.mark.parametrize('reuse', [ReuseType.MOO, ReuseType.GRAPH])
@pytest.mark.parametrize('strategy', ['file', 'model', 'capture', 'custom'])
def test_param_init_seed_cache_reuse(tmp_path, reuse, strategy):
    constructors = []
    graph_mtimes = []
    for seed in (17, 18):
        with patch.object(SeedBufferModule, '__init__', autospec=True, wraps=None) as constructor:
            def construct(module):
                MyModule.__init__(module)
                module.register_buffer('offset', torch.rand(1), persistent=False)
                constructors.append(module.offset.clone())
            constructor.side_effect = construct
            _to_cube_model(
                SeedBufferModule, ComputeConfig(1, 1, param_init_strategy=strategy, param_init_seed=seed),
                tmp_path, reuse, 'seed', load_module=False,
            )
            assert constructor.call_count == 1
        assert bool(list(tmp_path.rglob(FxModuleParser.NON_PERSISTENT_BUFFER_FILE))) == (strategy == 'file')
        graph_mtimes.append(next(tmp_path.rglob('graph.ckp')).stat().st_mtime_ns)
    assert graph_mtimes[0] != graph_mtimes[1]
    assert len(constructors) == 2
    assert not torch.equal(constructors[0], constructors[1])


@patch('torch.cuda.is_available', lambda: False)
@replace_all_device_with('cpu', force=True)
def test_param_init_strategy_reuse(tmp_path):
    local_graph_mtime = None
    for strategy in ('file', 'model', 'capture', 'custom', 'file'):
        kwargs = dict(
            gen_savedir=tmp_path, instance_name='init_strategy',
            load_module=False, reuse='moo',
        )
        config = ComputeConfig(1, 1, param_init_strategy=strategy)
        if strategy != 'file':
            with patch.object(Frame, 'save_attr_content', side_effect=AssertionError('fullmodel write')), \
                    patch.object(Frame, 'save_np_buffer_content', side_effect=AssertionError('npbuffer write')):
                parallelize(MyModule, {'x': torch.ones(2, 3)}, 'dp', config, **kwargs)
        else:
            parallelize(MyModule, {'x': torch.ones(2, 3)}, 'dp', config, **kwargs)
        module_dir = next(tmp_path.rglob(ParallelModule.COMPUTE_CONFIG_FILE)).parent
        assert bool(list(module_dir.glob('fullmodel.pt*'))) == (strategy == 'file')
        assert (module_dir / FxModuleParser.NON_PERSISTENT_BUFFER_FILE).exists() == (strategy == 'file')
        assert (module_dir / FxModuleParser.ATTR_MAP_FILE).exists()
        graph_file = next(module_dir.glob('graph*'))
        graph_mtime = graph_file.stat().st_mtime_ns
        code_mtime = (module_dir / 'gencode0.py').stat().st_mtime_ns
        parallelize(MyModule, {'x': torch.ones(2, 3)}, 'dp', config, **{**kwargs, 'reuse': 'match'})
        assert graph_file.stat().st_mtime_ns == graph_mtime
        assert (module_dir / 'gencode0.py').stat().st_mtime_ns == code_mtime
        if strategy == 'model':
            local_graph_mtime = graph_mtime
        elif strategy in ('capture', 'custom'):
            assert graph_mtime == local_graph_mtime


@patch('torch.cuda.is_available', lambda: False)
@replace_all_device_with('cpu', force=True)
@pytest.mark.parametrize('strategy', [
    ParamInitStrategy.FILE, ParamInitStrategy.MODEL,
    ParamInitStrategy.CAPTURE, ParamInitStrategy.CUSTOM,
])
@pytest.mark.parametrize('reuse', [ReuseType.MOO, ReuseType.GRAPH])
def test_reuse_without_np_buffer_content(tmp_path, strategy, reuse):
    config = ComputeConfig(1, 1, param_init_strategy=strategy)
    _to_cube_model(MyModule, config, tmp_path, ReuseType.MATCH, 'npbuffer', load_module=False)
    module_dir = next(tmp_path.rglob(ParallelModule.COMPUTE_CONFIG_FILE)).parent
    buffer_file = module_dir / FxModuleParser.NON_PERSISTENT_BUFFER_FILE
    graph_file = module_dir / 'graph.ckp'
    graph_mtime = graph_file.stat().st_mtime_ns
    if strategy == ParamInitStrategy.FILE:
        buffer_file.unlink()
        with raises_with_cause(RuntimeError, match='existing files do not match'):
            _to_cube_model(MyModule, config, tmp_path, ReuseType.MATCH, 'npbuffer', load_module=False)
    else:
        assert not buffer_file.exists()
        _to_cube_model(MyModule, config, tmp_path, ReuseType.MATCH, 'npbuffer', load_module=False)
    _to_cube_model(MyModule, config, tmp_path, reuse, 'npbuffer', load_module=False)
    assert buffer_file.is_file() == (strategy == ParamInitStrategy.FILE)
    assert (graph_file.stat().st_mtime_ns != graph_mtime) == (strategy == ParamInitStrategy.FILE)


@replace_all_device_with('cpu')
@pytest.mark.parametrize('reuse', [ReuseType.MATCH, ReuseType.MOO])
def test_reuse_without_attr_content_index(tmp_path, reuse):
    _to_cube_model(
        MyModule, ComputeConfig(1, 1), tmp_path, ReuseType.MATCH,
        'legacy', load_module=False,
    )
    module_path = next(
        tmp_path.rglob(FxModuleParser.ATTR_CONTENT_INDEX_FILE)).parent
    index_file = module_path / FxModuleParser.ATTR_CONTENT_INDEX_FILE
    index_file.unlink()
    code_mtime = (module_path / 'gencode0.py').stat().st_mtime_ns

    _to_cube_model(
        MyModule, ComputeConfig(1, 1), tmp_path, reuse,
        'legacy', load_module=False,
    )

    assert not index_file.exists()
    assert (module_path / 'gencode0.py').stat().st_mtime_ns == code_mtime


@replace_all_device_with('cpu')
def test_override_clears_attr_content_index(tmp_path, monkeypatch):
    _to_cube_model(
        MyModule, ComputeConfig(1, 1), tmp_path, ReuseType.MATCH,
        'override_index', load_module=False,
    )
    module_path = next(
        tmp_path.rglob(FxModuleParser.ATTR_CONTENT_INDEX_FILE)).parent
    index_file = module_path / FxModuleParser.ATTR_CONTENT_INDEX_FILE

    save_attr_content = Frame.save_attr_content

    def check_index_was_cleared(self, save_file_stem, params_per_file=1024 * 1024 * 1024):
        assert not index_file.exists()
        return save_attr_content(self, save_file_stem, params_per_file)

    monkeypatch.setattr(Frame, 'save_attr_content', check_index_was_cleared)
    _to_cube_model(
        MyModule, ComputeConfig(1, 1), tmp_path, ReuseType.OVERRIDE,
        'override_index', load_module=False,
    )

    assert index_file.exists()


@replace_all_device_with('cpu')
def test_override():
    with tempfile.TemporaryDirectory() as tempdir:
        # MATCH   | empty | generate
        cmodule1 = _to_cube_model(MyModule, ComputeConfig(1, 1),tempdir, ReuseType.MATCH, 'mm0')
        # MATCH  | match | do nothing
        cmodule2 = _to_cube_model(MyModule, ComputeConfig(1, 1),tempdir, ReuseType.MATCH, 'mm0')
        for (n1, v1), (n2, v2) in zip(cmodule1.named_parameters(), cmodule2.named_parameters()):
            assert n1 == n2
            assert torch.equal(v1, v2)

        # MATCH  | match | do nothing
        cmodule3 = _to_cube_model(MyModule, ComputeConfig(1, 1),tempdir, ReuseType.MATCH, 'test')
        cmodule4 = _to_cube_model(MyModule, ComputeConfig(1, 1),tempdir, 'match', 'test')

        for (n1, v1), (n2, v2) in zip(cmodule3.named_parameters(), cmodule4.named_parameters()):
            assert n1 == n2
            assert torch.equal(v1, v2)

        cmodule2_p = dict(cmodule2.named_parameters())
        cmodule3_p = dict(cmodule3.named_parameters())
        keys = cmodule3_p.keys()
        assert all(torch.equal(cmodule2_p[key], cmodule3_p[key]) for key in keys)

        # MATCH  | unmatch | raise error
        _to_cube_model(MyModule, ComputeConfig(1, 1),tempdir, ReuseType.MATCH, 'm0')
        with raises_with_cause(RuntimeError, match='.*not empty.*'):
            _to_cube_model(MyModule, ComputeConfig(2, 2),tempdir, 'match', 'm0')

        # MOO   | empty | generate
        omodule1 = _to_cube_model(MyModule, ComputeConfig(1, 1),tempdir, ReuseType.MOO, 'o0')
        # MOO  | match | do nothing
        omodule2 = _to_cube_model(MyModule, ComputeConfig(1, 1),tempdir, ReuseType.MOO, 'o0')
        for (n1, v1), (n2, v2) in zip(omodule1.named_parameters(), omodule2.named_parameters()):
            assert n1 == n2
            assert torch.equal(v1, v2)

        # MOO  | unmatch | generate
        _to_cube_model(MyModule, ComputeConfig(1, 1),tempdir, ReuseType.MOO, 'o1', load_module=False)
        _to_cube_model(MyModule, ComputeConfig(2, 2, constant_folding=True),tempdir, ReuseType.MOO, 'o1')

        # MOO  | imported | raise error
        _to_cube_model(MyModule, ComputeConfig(1, 1),tempdir, ReuseType.MOO, 'o2', load_module=True)
        with raises_with_cause(RuntimeError):
            _to_cube_model(MyModule, ComputeConfig(2, 2),tempdir, ReuseType.MOO, 'o2')

        # OVERRIDE   | imported | raise error
        with raises_with_cause(RuntimeError):
            _to_cube_model(MyModule, ComputeConfig(1, 1),tempdir, ReuseType.OVERRIDE, 'mm0')

        # OVERRIDE   | imported | raise error
        with raises_with_cause(RuntimeError):
            _to_cube_model(MyModule, ComputeConfig(1, 1),tempdir, ReuseType.OVERRIDE, 'test')

        # OVERRIDE  | imported | raise error
        with raises_with_cause(RuntimeError):
            _to_cube_model(MyModule, ComputeConfig(2, 2),tempdir, ReuseType.OVERRIDE, 'test')

        # OVERRIDE   | empty | generate
        cmodule1 = _to_cube_model(MyModule, ComputeConfig(1, 1),tempdir, ReuseType.OVERRIDE, 'test2')
        module_path = Path(sys.modules[cmodule1.__module__].__file__).parent
        test3_module_path = module_path.with_name('test3')
        test3_module_path.mkdir(exist_ok=True, parents=True)
        test4_module_path = module_path.with_name('test4')
        test4_module_path.mkdir(exist_ok=True, parents=True)
        test5_module_path = module_path.with_name('test5')
        test5_module_path.mkdir(exist_ok=True, parents=True)
        for f in module_path.glob('*'):
            if f.is_file():
                shutil.copy(f, test3_module_path / f.name)
                shutil.copy(f, test4_module_path / f.name)
                shutil.copy(f, test5_module_path / f.name)
        # fake two gpus
        shutil.copy(test4_module_path / 'gencode0.py', test4_module_path / 'gencode1.py')
        shutil.copy(test5_module_path / 'gencode0.py', test5_module_path / 'gencode1.py')

        # OVERRIDE   | match | generate
        original_init = MyModule.__init__

        def changed_init(module):
            original_init(module)
            with torch.no_grad():
                module.linear.weight.add_(1)

        with patch.object(MyModule, '__init__', changed_init):
            cmodule2 = _to_cube_model(MyModule, ComputeConfig(1, 1), tempdir, ReuseType.OVERRIDE, 'test3')
        cmodule2_p = dict(cmodule2.named_parameters())
        cmodule1_p = dict(cmodule1.named_parameters())
        keys = cmodule2_p.keys()
        assert any(not torch.equal(cmodule2_p[key], cmodule1_p[key]) for key in keys)

        # OVERRIDE   | unmatch | generate
        assert (test4_module_path / 'gencode1.py').exists()
        cmodule3 = _to_cube_model(MyModule, ComputeConfig(1, 1), tempdir, 'override', 'test4')
        assert not (test4_module_path / 'gencode1.py').exists()

        # Graph | matched legacy attr metadata | generate
        assert (test5_module_path / 'gencode1.py').exists()
        attr_merged_file = test5_module_path / ParallelModule.ATTR_META_MERGED_FILE
        with open(attr_merged_file, 'rb') as f:
            attr_meta_maps = pickle.load(f)
        attr_merged_file.unlink()
        for rank, attr_meta_map in enumerate(attr_meta_maps):
            with open(test5_module_path / ParallelModule.ATTR_META_FILE_TEMPLATE.format(rank), 'wb') as f:
                pickle.dump(attr_meta_map, f)
        code_stat = (test5_module_path / 'gencode0.py').stat()
        graph_stat = (test5_module_path / 'graph.ckp').stat()
        args_stat = (test5_module_path / 'forward_args.pkl').stat()
        _to_cube_model(MyModule, ComputeConfig(1, 1), tempdir, 'graph', 'test5', False)
        assert not (test5_module_path / 'gencode1.py').exists()
        assert attr_merged_file.exists()
        assert not list(test5_module_path.glob(f'{ParallelModule.ATTR_META_FILE_PREFIX}[0-9]*.pkl'))
        _to_cube_model(MyModule, ComputeConfig(1, 1), tempdir, 'match', 'test5', False)
        assert (test5_module_path / 'gencode0.py').stat().st_mtime_ns != code_stat.st_mtime_ns
        assert (test5_module_path / 'graph.ckp').stat().st_mtime_ns == graph_stat.st_mtime_ns
        assert (test5_module_path / 'forward_args.pkl').stat().st_mtime_ns == args_stat.st_mtime_ns

        code_stat = (test5_module_path / 'gencode0.py').stat()
        graph_stat = (test5_module_path / 'graph.ckp').stat()
        (test5_module_path / 'forward_args.pkl').unlink()  # remove foward_args.pkl will force to generate new code
        _to_cube_model(MyModule, ComputeConfig(1, 1), tempdir, 'graph', 'test5', False)
        assert (test5_module_path / 'gencode0.py').stat().st_mtime_ns != code_stat.st_mtime_ns
        assert (test5_module_path / 'graph.ckp').stat().st_mtime_ns != graph_stat.st_mtime_ns
        assert (test5_module_path / 'forward_args.pkl').exists()

        code_stat = (test5_module_path / 'gencode0.py').stat()
        graph_stat = (test5_module_path / 'graph.ckp').stat()
        attrmap_stat = (test5_module_path / FxModuleParser.ATTR_MAP_FILE).stat()
        (test5_module_path / FxModuleParser.ATTR_CONTENT_FILE_0).unlink()  # remove fullmodel.pt.0 will force to generate new code
        _to_cube_model(MyModule, ComputeConfig(1, 1), tempdir, 'graph', 'test5', False)
        assert (test5_module_path / 'gencode0.py').stat().st_mtime_ns != code_stat.st_mtime_ns
        assert (test5_module_path / 'graph.ckp').stat().st_mtime_ns != graph_stat.st_mtime_ns
        assert (test5_module_path / FxModuleParser.ATTR_MAP_FILE).stat().st_mtime_ns != attrmap_stat.st_mtime_ns
        assert (test5_module_path / 'forward_args.pkl').exists()

        # Graph | empty | generate
        g6_module = _to_cube_model(MyModule, ComputeConfig(1, 1), tempdir, 'graph', 'g6')

        # Graph | imported | raise error
        with raises_with_cause(RuntimeError):
            _to_cube_model(MyModule, ComputeConfig(1, 1), tempdir, 'graph', 'g6')

        # Graph | unmatch | generate
        _to_cube_model(MyModule, ComputeConfig(1, 1), tempdir, 'graph', 'g7', False)
        g7_module_path = module_path.with_name('g7')
        graph_stat = (g7_module_path / 'graph.ckp').stat()
        args_stat = (g7_module_path / 'forward_args.pkl').stat()
        _to_cube_model(MyModule, ComputeConfig(2, 2, constant_folding=True), tempdir, 'graph', 'g7', False)
        assert (g7_module_path / 'graph.ckp').stat().st_mtime_ns != graph_stat.st_mtime_ns
        assert (g7_module_path / 'forward_args.pkl').stat().st_mtime_ns != args_stat.st_mtime_ns

        # Graph | graph match | generate
        _to_cube_model(MyModule, ComputeConfig(1, 1), tempdir, 'graph', 'g8', False)
        g8_module_path = module_path.with_name('g8')
        assert ComputeConfig.safe_load_from_file(g8_module_path / ParallelModule.COMPUTE_CONFIG_FILE) == ComputeConfig(1, 1)
        graph_stat = (g8_module_path / 'graph.ckp').stat()
        args_stat = (g8_module_path / 'forward_args.pkl').stat()
        _to_cube_model(MyModule, ComputeConfig(2, 2), tempdir, 'graph', 'g8', False)
        assert (g8_module_path / 'graph.ckp').stat().st_mtime_ns == graph_stat.st_mtime_ns
        assert (g8_module_path / 'forward_args.pkl').stat().st_mtime_ns == args_stat.st_mtime_ns
        assert ComputeConfig.safe_load_from_file(g8_module_path / ParallelModule.COMPUTE_CONFIG_FILE) == ComputeConfig(2, 2)

        # MOO | graph match | generate code only
        _to_cube_model(MyModule, ComputeConfig(1, 1), tempdir, 'moo', 'g9', False)
        g9_module_path = module_path.with_name('g9')
        assert ComputeConfig.safe_load_from_file(g9_module_path / ParallelModule.COMPUTE_CONFIG_FILE) == ComputeConfig(1, 1)
        graph_stat = (g9_module_path / 'graph.ckp').stat()
        args_stat = (g9_module_path / 'forward_args.pkl').stat()
        _to_cube_model(MyModule, ComputeConfig(2, 2), tempdir, 'moo', 'g9', False)
        assert (g9_module_path / 'graph.ckp').stat().st_mtime_ns == graph_stat.st_mtime_ns
        assert (g9_module_path / 'forward_args.pkl').stat().st_mtime_ns == args_stat.st_mtime_ns
        assert ComputeConfig.safe_load_from_file(g9_module_path / ParallelModule.COMPUTE_CONFIG_FILE) == ComputeConfig(2, 2)
