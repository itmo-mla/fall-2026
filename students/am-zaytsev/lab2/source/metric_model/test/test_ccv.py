import numpy as np
import pytest
from metric_model.train.ccv import get_m_closest_class, get_sorted_m


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
    X = np.random.uniform(0, 1, [100, 4])
    Y = np.int32(np.random.uniform(-10, 10, [100, 1]) > 0)

    for m in range(X.shape[0]):
        for x in X:
            true = slow_get_m_closest_class(x, X, Y, m)
            pred = get_m_closest_class(x, X, Y, m, return_idx=True)
            assert true == pred
            true = slow_get_m_closest_class(x, X, Y, m)[0]
            pred = get_m_closest_class(x, X, Y, m, return_idx=False)
            assert true == pred
