import pytest
import torch
from nnscaler.parallel import (
    ComputeConfig, ReuseType, _prepare_and_check_reusable,
    _GRAPH_DUMP_FILE, _FORWARD_ARGS_DUMP_FILE,
)
from nnscaler.graph.parser import FxModuleParser
from nnscaler.runtime.module import ParallelModule

def make_package(tmp_path):
    cfg=ComputeConfig(1,1)
    folder,_=_prepare_and_check_reusable(str(tmp_path),torch.nn.Linear,cfg)
    ComputeConfig.safe_dump_to_file(cfg,folder/ParallelModule.COMPUTE_CONFIG_FILE)
    for name in ['gencode0.py',_GRAPH_DUMP_FILE,_FORWARD_ARGS_DUMP_FILE,
                 ParallelModule.ORIGIN_MODULE_METADATA_FILE,ParallelModule.ATTR_META_FILE,
                 FxModuleParser.ATTR_CONTENT_FILE_0,FxModuleParser.ATTR_CONTENT_INDEX_FILE,
                 FxModuleParser.ATTR_MAP_FILE,FxModuleParser.NON_PERSISTENT_BUFFER_FILE]:
        (folder/name).touch()
    return cfg,folder

def test_resume_only_package_is_not_valid_for_fresh_training(tmp_path):
    cfg,folder=make_package(tmp_path)
    assert _prepare_and_check_reusable(str(tmp_path),torch.nn.Linear,cfg)[1]
    (folder/FxModuleParser.ATTR_CONTENT_FILE_0).unlink()
    assert _prepare_and_check_reusable(str(tmp_path),torch.nn.Linear,cfg,allow_missing_init_weights=True)[1]
    with pytest.raises(RuntimeError,match='do not match'):
        _prepare_and_check_reusable(str(tmp_path),torch.nn.Linear,cfg)

@pytest.mark.parametrize('missing',['npbuffer.pt','gencode0.py','fullmodel.pt.index'])
def test_resume_still_requires_code_buffers_and_metadata(tmp_path,missing):
    cfg,folder=make_package(tmp_path)
    (folder/FxModuleParser.ATTR_CONTENT_FILE_0).unlink()
    (folder/missing).unlink()
    # Keep another Python file to avoid the intentional failed-generation retry path.
    if missing=='gencode0.py':(folder/'unrecognized.py').touch()
    with pytest.raises(RuntimeError,match='do not match'):
        _prepare_and_check_reusable(str(tmp_path),torch.nn.Linear,cfg,allow_missing_init_weights=True)
