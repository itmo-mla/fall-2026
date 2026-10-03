import numpy as np
import pytest
from metric_model.dataset.utils import remove_row
from metric_model.train.ccv import (
    compact_profile,
    get_m_closest_class,
    get_sorted_m,
    remove_row_compact_profile,
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


def test_dp_dx():
    n = 100
    x = np.random.random([n, 5])
    y = np.int32(np.random.uniform(-10, 10, [n, 1]) > 0)

    p_old = compact_profile(x, y, 1)

    for del_i in range(x.shape[0]):
        p_new_pred = remove_row_compact_profile(x, y, del_i, p_old)

        p_new_true = compact_profile(remove_row(x, del_i), remove_row(y, del_i), 1)
        assert (p_new_pred - p_new_true) / p_new_true < 0.001
