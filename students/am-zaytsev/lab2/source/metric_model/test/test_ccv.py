import numpy as np
import pytest
from metric_model.train.ccv import get_sorted_m


@pytest.mark.parametrize("n", [(2), (10), (1000)])
def test_get_sorted_m(n):
    arr = np.random.uniform(-10, 10, n)
    sorted_arr = np.sort(arr)
    for i in range(arr.shape[0]):
        sorted_arr_i = get_sorted_m(arr, i)
        assert sorted_arr_i == sorted_arr[i]
