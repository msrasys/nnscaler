#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import os
import subprocess
import sys
import textwrap


def test_module_import_is_safe_without_dion():
    code = textwrap.dedent(
        """
        import builtins
        import torch

        original_import = builtins.__import__

        def block_dion(name, *args, **kwargs):
            if name == 'dion' or name.startswith('dion.'):
                raise ImportError('Dion intentionally hidden by test')
            return original_import(name, *args, **kwargs)

        builtins.__import__ = block_dion
        from nnscaler.runtime.dion_muon_optimizer import DionMuon

        param = torch.nn.Parameter(torch.zeros(2, 2))
        try:
            DionMuon([param])
        except ImportError as error:
            assert 'Install Dion' in str(error)
        else:
            raise AssertionError('DionMuon construction should require Dion')
        """
    )
    subprocess.run(
        [sys.executable, '-c', code],
        check=True,
        env=os.environ.copy(),
    )
