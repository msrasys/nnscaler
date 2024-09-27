#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import pytest

from nnscaler.utils import select_many


def test_select_many():
    assert list(select_many([1, 2], lambda k: [])) == []
    assert list(select_many([1, [2, 3]], lambda k: k if isinstance(k, list) else [k])) == [1, 2, 3]
    with pytest.raises(TypeError):
        list(select_many([1, [2, 3]], lambda k: k))
