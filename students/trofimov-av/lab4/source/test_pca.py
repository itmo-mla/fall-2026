import unittest

import numpy as np
from sklearn.decomposition import PCA as SklearnPCA

from data import prepare
from pca import PCA, effective_dimension


class PCATest(unittest.TestCase):
    def test_reference_on_unseen_data(self):
        rng = np.random.default_rng(42)
        train = rng.normal(size=(40, 4)) @ np.diag([4, 3, 2, 1]) + 10
        test = rng.normal(size=(8, 4)) + 12
        for k in range(1, 5):
            with self.subTest(k=k):
                model = PCA(k).fit(train)
                reference = SklearnPCA(k, svd_solver="full").fit(train)
                np.testing.assert_allclose(model.explained_variance_ratio_, reference.explained_variance_ratio_)
                np.testing.assert_allclose(
                    model.inverse_transform(model.transform(test)),
                    reference.inverse_transform(reference.transform(test)), atol=1e-12,
                )
                np.testing.assert_allclose(model.components_ @ model.components_.T, np.eye(k), atol=1e-12)

    def test_reconstruction_and_singular_values(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(20, 4))
        full = PCA(4).fit(x)
        errors = []
        for k in range(1, 5):
            model = PCA(k).fit(x)
            restored = model.inverse_transform(model.transform(x))
            error = np.mean((x - restored)**2)
            expected = np.sum(full.singular_values_[k:]**2) / x.size
            self.assertAlmostEqual(error, expected)
            errors.append(error)
        self.assertTrue(np.all(np.diff(errors) <= 1e-12))
        np.testing.assert_allclose(restored, x, atol=1e-12)

    def test_rank_deficient_and_constant_data(self):
        t = np.arange(10, dtype=float)
        x = np.column_stack((t, 2 * t, np.ones(10)))
        model = PCA(3).fit(x)
        self.assertEqual(effective_dimension(model.explained_variance_ratio_, 0.99), 1)
        np.testing.assert_allclose(model.inverse_transform(model.transform(x)), x, atol=1e-12)
        constant = PCA(2).fit(np.ones((10, 3)))
        np.testing.assert_array_equal(constant.explained_variance_ratio_, [0, 0])
        with self.assertRaises(ValueError):
            effective_dimension(constant.explained_variance_ratio_)

    def test_thresholds_and_train_preprocessing(self):
        self.assertEqual(effective_dimension([0.8, 0.15, 0.05], 0.95), 2)
        self.assertEqual(effective_dimension([0.8, 0.15, 0.05], 1), 3)
        train = np.array([[0., 4.], [2., 4.]])
        test = np.array([[11., 8.]])
        prepared_train, prepared_test, mean, scale = prepare(train, test, True)
        np.testing.assert_allclose(mean, [1, 4])
        np.testing.assert_allclose(scale, [1, 1])
        np.testing.assert_allclose(prepared_train, [[-1, 0], [1, 0]])
        np.testing.assert_allclose(prepared_test, [[10, 4]])
        with self.assertRaises(ValueError):
            PCA(5).fit(train)


if __name__ == "__main__":
    unittest.main()
