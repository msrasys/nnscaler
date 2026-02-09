from nnscaler.runtime.utils import split_array_min_max


def test_split_array_min_max():
    nums = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    g = 3
    groups, group_idx = split_array_min_max(nums, g, keep_order=True)
    assert groups == [[1, 2, 3, 4, 5], [6, 7], [8, 9]]
    assert group_idx == [[0, 1, 2, 3, 4], [5, 6], [7, 8]]

    groups, group_idx = split_array_min_max(nums, g, keep_order=False)
    assert groups == [[9, 4, 3], [8, 5, 2], [7, 6, 1]]
    assert group_idx == [[8, 3, 2], [7, 4, 1], [6, 5, 0]]

    nums = [10, 10, 10, 10, 10, 10]
    g = 3
    groups, group_idx = split_array_min_max(nums, g, keep_order=True)
    assert groups == [[10, 10], [10, 10], [10, 10]]
    assert group_idx == [[0, 1], [2, 3], [4, 5]]

    groups, group_idx = split_array_min_max(nums, g, keep_order=False)
    assert groups == [[10, 10], [10, 10], [10, 10]]
    assert group_idx == [[5, 2], [4, 1], [3, 0]]

    nums = [
        6553600, 1310720, 1310720, 6553600, 2621440, 2621440, 2621440,
        6553600, 1310720, 1310720, 6553600, 2621440, 2621440, 2621440,
        6553600, 1310720, 1310720, 6553600, 2621440, 2621440, 2621440,
        6553600, 1310720, 1310720, 6553600, 2621440, 2621440, 2621440,
        6553600, 1310720, 1310720, 6553600, 2621440, 2621440, 2621440,
        6553600, 1310720, 1310720, 6553600, 2621440, 2621440, 2621440,
        6553600, 1310720, 1310720, 6553600, 2621440, 2621440, 2621440,
        6553600, 1310720, 1310720, 6553600, 2621440, 2621440, 2621440,
        6553600, 1310720, 1310720, 6553600, 2621440, 2621440, 2621440,
        6553600, 1310720, 1310720, 6553600, 2621440, 2621440, 2621440,
        6553600, 6553600, 2621440, 2621440, 2621440,
        6553600, 6553600, 2621440, 2621440, 2621440,
        6553600, 6553600, 2621440, 2621440, 2621440,
        6553600, 6553600, 2621440, 2621440, 2621440,
        6553600, 6553600, 2621440, 2621440, 2621440,
        6553600, 6553600, 2621440, 2621440, 2621440,
        6553600, 6553600, 2621440, 2621440, 2621440,
        6553600, 6553600, 2621440, 2621440, 2621440,
        6553600, 6553600, 2621440, 2621440, 2621440,
        6553600, 6553600, 2621440, 2621440, 2621440,
        1310720, 1310720
    ]
    g = 8
    best_sum = sum(nums) // g

    groups, group_idx = split_array_min_max(nums, g, keep_order=True)
    max_sum = max(sum(group) for group in groups)
    assert len(groups) == 8
    assert list(j for k in group_idx for j in k) == list(range(len(nums)))

    groups, group_idx = split_array_min_max(nums, g, keep_order=False)
    assert len(groups) == 8
    max_sum2 = max(sum(group) for group in groups)
    assert list(j for k in group_idx for j in k) != list(range(len(nums)))

    assert best_sum< max_sum2 < max_sum
    print(f'best_sum: {best_sum}, keep_order: {max_sum}, not keep_order: {max_sum2}')
