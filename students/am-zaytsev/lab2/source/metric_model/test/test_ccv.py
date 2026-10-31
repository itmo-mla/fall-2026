import numpy as np
import pytest
from metric_model.train.ccv import (
    closest_idx,
    get_2nn_idx_list,
    get_m_closest_class,
    get_sorted_m,
    remove_row_set_compact_profile,
)


@pytest.mark.parametrize("n", [(2), (10), (1000)])
def test_get_sorted_m(n):
    arr = np.random.uniform(-10, 10, n)
    sorted_arr = np.sort(arr)
    for i in range(arr.shape[0]):
        sorted_arr_i = get_sorted_m(arr, i)
        assert sorted_arr_i == sorted_arr[i]


def slow_get_m_closest_class(item, X, y, m):
    dist_list = np.linalg.norm(X - item, axis=1)
    sorted_idx = np.argsort(dist_list)
    m_close_idx = sorted_idx[m]
    return y[m_close_idx], m_close_idx


def test_get_m_closest_class():
    X = np.random.random([100, 4])
    Y = np.int32(np.random.uniform(-10, 10, [100, 1]) > 0)

    for m in range(X.shape[0]):
        for x in X:
            true = slow_get_m_closest_class(x, X, Y, m)
            pred = get_m_closest_class(x, X, Y, m, return_idx=True)
            assert true == pred
            true = slow_get_m_closest_class(x, X, Y, m)[0]
            pred = get_m_closest_class(x, X, Y, m, return_idx=False)
            assert true == pred


# def test_dp_dx():
#     n = 100
#     x = np.random.random([n, 5])
#     y = np.int32(np.random.uniform(-10, 10, [n, 1]) > 0)

#     p_old = compact_profile(x, y, 1)

#     for del_i in range(x.shape[0]):
#         p_new_pred = remove_row_compact_profile(x, y, del_i, p_old)

#         p_new_true = compact_profile(remove_row(x, del_i), remove_row(y, del_i), 1)
#         assert (p_new_pred - p_new_true) / p_new_true < 0.001



# ----------------------------------------------------------------------
# Tests for closest_idx
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "x, xi, n, removed_idx_set, expected",
    [
        (
            np.array([[0], [1], [2], [3], [4]]),
            np.array([2]),
            3,
            set(),
            [2, 1, 3],
        ),
        (
            np.array([[0], [1], [2], [3], [4]]),
            np.array([2]),
            2,
            {2},
            [1, 3],
        ),
        (
            np.array([[0], [1], [2], [3], [4]]),
            np.array([2]),
            4,
            {0, 4},
            [2, 1, 3],
        ),
    ],
)
def test_closest_idx(x, xi, n, removed_idx_set, expected):
    result = closest_idx(x, xi, n, removed_idx_set)
    # convert numpy ints to Python ints for a clean comparison
    result = [int(i) for i in result]
    assert result == expected


# ----------------------------------------------------------------------
# Tests for get_2nn_idx_list
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "x, removed_idx_set, expected_k1, expected_k2",
    [
        (
            np.array([[0], [2], [5], [9], [14]]),
            set(),
            [1, 0, 1, 2, 3],
            [2, 2, 3, 4, 2],
        ),
        (
            np.array([[0], [2], [5], [9], [14]]),
            {1},
            [2, 0, 3, 2, 3],
            [3, 2, 0, 4, 2],
        ),
        (
            np.array([[0], [2], [5], [9], [14]]),
            {0, 4},
            [1, 2, 1, 2, 3],
            [2, 3, 3, 1, 2],
        ),
    ],
)
def test_get_2nn_idx_list(x, removed_idx_set, expected_k1, expected_k2):
    k1, k2 = get_2nn_idx_list(x, removed_idx_set)
    assert k1 == expected_k1
    assert k2 == expected_k2


# ----------------------------------------------------------------------
# Tests for remove_row_set_compact_profile
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "x, y, row_i, old_p, removed_idx_set, expected",
    [
        (
            np.array([[0], [2], [5], [9], [14]]),
            np.array([0, 0, 1, 1, 0]),
            2,
            0.4,
            set(),
            0.6,
        ),
        (
            np.array([[0], [2], [5], [9], [14]]),
            np.array([0, 0, 1, 1, 0]),
            2,
            0.4,
            {1},
            0.6,
        ),
        (
            np.array([[0], [2], [5], [9], [14]]),
            np.array([0, 0, 1, 1, 0]),
            0,
            0.4,
            set(),
            0.6,
        ),
        (
            np.array([[0], [2], [5], [9], [14]]),
            np.array([0, 0, 1, 1, 0]),
            3,
            0.4,
            set(),
            0.4,
        ),
    ],
)
def test_remove_row_set_compact_profile(x, y, row_i, old_p, removed_idx_set, expected):
    result = remove_row_set_compact_profile(x, y, row_i, old_p, removed_idx_set)
    assert np.isclose(result, expected)