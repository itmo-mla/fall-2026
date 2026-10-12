import numpy as np
import pytest
from lab4.PCA import PCA

np.random.default_rng(12345)


@pytest.mark.parametrize(
    "n, m", [(1, 1), (2, 2), (2, 1), (1, 2), (4, 4), (50, 4), (4, 50)]
)
def test_reconstruct(n, m):
    arr = np.random.randint(0, 100, (n, m))
    pca = PCA(arr)
    assert np.allclose(arr, pca.reconstruct())
