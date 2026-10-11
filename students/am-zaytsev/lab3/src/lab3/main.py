import numpy as np

from lab3.datasets import pima as dataset
from lab3.model import SVM

RANDOM_SEED = 42
TEST_SIZE = 0.2


def train_test_split(x, y, test_size=TEST_SIZE, seed=RANDOM_SEED):
    n = x.shape[0]
    idx = np.random.default_rng(seed).permutation(n)
    n_test = int(n * test_size)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return x[train_idx], y[train_idx], x[test_idx], y[test_idx]


def fit_normalizer(x_train):
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return mean, std


def apply_normalizer(x, mean, std):
    return (x - mean) / std


def main():
    df = dataset.download_dataset()
    x, y = dataset.preprocess_data(df)

    x_train, y_train, x_test, y_test = train_test_split(x, y)

    mean, std = fit_normalizer(x_train)
    x_train = apply_normalizer(x_train, mean, std)
    x_test = apply_normalizer(x_test, mean, std)

    model = SVM(1)
    model.train(x_train, y_train)

    train_acc = (model.predict(x_train) == y_train).mean()
    test_acc = (model.predict(x_test) == y_test).mean()
    print(f"train accuracy: {train_acc:.4f}")
    print(f"test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
