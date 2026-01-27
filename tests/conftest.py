#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import pytest
from pathlib import Path

from nnscaler.graph.parser import FxModuleParser

try:
    import nnscaler.autodist.dp_solver
except ImportError:
    from pathlib import Path
    from cppimport import build_filepath
    import nnscaler.autodist
    # lazy build the cpp file if it is not built yet
    build_filepath(Path(nnscaler.autodist.__file__).with_name("dp_solver.cpp"), fullname="nnscaler.autodist.dp_solver")


@pytest.fixture(autouse=True)
def clean_generated_files():
    print('hello')
    yield
    # try to clean generated files after each test run.
    basedir = Path('./').resolve()
    generated_files = [FxModuleParser.ATTR_CONTENT_FILE_0, FxModuleParser.ATTR_MAP_FILE]
    for f in generated_files:
        f = basedir / f
        if f.exists():
            f.unlink()
    for f in basedir.glob('gencode*.py'):
        f.unlink()


def pytest_collection_modifyitems(session, config, items):
    def _f(item):
        # if item.path.name == 'test_gencode_ctx_manager.py':
        #     return 0
        # if item.path.name == 'test_pipeline_nstages.py':
        #     return 1
        if 'runtime' in str(item.path):  # very fast and easy to fail
            return 10
        if 'cli' in str(item.path):  # very important and easy to fail
            return 11
        if 'parallel_module' in str(item.path):  # very important and easy to fail but a bit slow
            return 12
        if 'customized_ops' in str(item.path):  # too slow and very stable.
            return 100
        return 50
    items.sort(key=_f)
