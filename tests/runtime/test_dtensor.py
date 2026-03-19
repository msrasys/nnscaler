from nnscaler.runtime.dtensor import DTensor


def test_pp_groups():
    class FakeDTensor:
        def __init__(self, attr_metas):
            self.attr_metas = attr_metas
    # case 1: pure pp (plan_gnpus=2, runtime_gnpus=2)
    DTensor._get_pp_groups(FakeDTensor(['x', None])) == [[0, 1]]
    DTensor._get_pp_groups(FakeDTensor([None, 'x'])) == [[0, 1]]

    # case 2: pure pp (plan_gnpus=4, runtime_gnpus=4)
    DTensor._get_pp_groups(FakeDTensor(['x', None, None, None])) == [[0, 1, 2, 3]]
    DTensor._get_pp_groups(FakeDTensor([None, 'x', None, None])) == [[0, 1, 2, 3]]
    DTensor._get_pp_groups(FakeDTensor([None, None, 'x', None])) == [[0, 1, 2, 3]]
    DTensor._get_pp_groups(FakeDTensor([None, None, None, 'x'])) == [[0, 1, 2, 3]]

    # case 3: pp + tp (plan_gnpus=4, runtime_gnpus=4)
    DTensor._get_pp_groups(FakeDTensor(['x', 'x', None, None])) == [[0, 2], [1, 3]]
    DTensor._get_pp_groups(FakeDTensor([None, None, 'x', 'x'])) == [[0, 2], [1, 3]]

    # case 4: pp + dp (plan_gnpus=2, runtime_gnpus=4)
    DTensor._get_pp_groups(FakeDTensor(['x', None, 'x', None])) == [[0, 1], [2, 3]]
    DTensor._get_pp_groups(FakeDTensor([None, 'x', None, 'x'])) == [[0, 1], [2, 3]]

    # case 5: pp + tp + dp (plan_gnpus=4, runtime_gnpus=8)
    DTensor._get_pp_groups(FakeDTensor(['x', 'x', None, None, 'x', 'x', None, None])) == [[0, 2], [1, 3], [4, 6], [5, 7]]
    DTensor._get_pp_groups(FakeDTensor([None, None, 'x', 'x', None, None, 'x', 'x'])) == [[0, 2], [1, 3], [4, 6], [5, 7]]

    # case 6: bigger pp + tp + dp (plan_gnpus=8, runtime_gnpus=16)
    DTensor._get_pp_groups(FakeDTensor(['x', 'x', None, None, None, None, None, None, 'x', 'x', None, None, None, None, None, None])) == [[0, 2, 4, 6], [1, 3, 5, 7], [8, 10, 12, 14], [9, 11, 13, 15]]
    DTensor._get_pp_groups(FakeDTensor([None, None, None, None, 'x', 'x', None, None, None, None, None, None, 'x', 'x', None, None])) == [[0, 2, 4, 6], [1, 3, 5, 7], [8, 10, 12, 14], [9, 11, 13, 15]]
