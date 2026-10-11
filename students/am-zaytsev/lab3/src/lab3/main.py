from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from lab3.datasets import pima as dataset
from lab3.model import SVM

RANDOM_SEED = 42
TEST_SIZE = 0.2


def main():
    df = dataset.download_dataset()
    x, y = dataset.preprocess_data(df)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    model = SVM(1)
    model.train(x_train, y_train)

    train_acc = (model.predict(x_train) == y_train).mean()
    test_acc = (model.predict(x_test) == y_test).mean()
    print(f"train accuracy: {train_acc:.4f}")
    print(f"test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
