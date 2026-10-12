import numpy as np
import pytest
from lab4.PCA import PCA
from sklearn.decomposition import PCA as skPCA

np.random.default_rng(12345)


@pytest.mark.parametrize(
    "n, m", [(1, 1), (2, 2), (2, 1), (1, 2), (4, 4), (50, 4), (4, 50)]
)
def test_reconstruct(n, m):
    arr = np.random.randint(0, 100, (n, m))
    pca = PCA(arr)
    assert np.allclose(arr, pca.reconstruct())


@pytest.mark.parametrize("n, m", [(4, 4), (50, 4), (40, 40)])
def test_reduce_components(n, m):
    arr = np.random.randint(0, 100, (n, m))
    for reduce_cnt in range(m):
        pca = PCA(arr)
        pca.reduce_component(reduce_cnt)

        sse = ((pca.reconstruct() - arr) ** 2).sum()
        if reduce_cnt == 0:
            reduced_singular_values = 0
        else:
            reduced_singular_values = pca.s[-reduce_cnt:]
        singular_error = np.sum(reduced_singular_values**2)

        assert abs(sse - singular_error) < 1e-4, f"{reduce_cnt=}"


@pytest.mark.parametrize("components_k", list(range(10)))
def test_compare_pca(components_k):
    arr = np.random.randint(0, 100, (100, 10))

    sk_pca = skPCA(n_components=components_k)
    sk_x = sk_pca.fit_transform(arr)

    pca = PCA(arr)
    pca.reduce_component(arr.shape[1] - components_k)
    assert np.allclose(np.abs(pca.g), np.abs(sk_x), atol=0.001)
