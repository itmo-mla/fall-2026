from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from lab3.datasets import pima as dataset
from lab3.model import SVM, kernels
from lab3.visual.utils import draw_dataset

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

    model = SVM(1, kernels.rbf)
    model.train(x_train, y_train)

    train_acc = (model.predict(x_train) == y_train).mean()
    test_acc = (model.predict(x_test) == y_test).mean()
    draw_dataset(
        x_train, y_train, y_pred=model.predict(x_train), mask=model.lam == model.c
    )
    plt.savefig("./images/train_data.png", dpi=300, bbox_inches="tight")

    draw_dataset(x_test, y_test, y_pred=model.predict(x_test))
    plt.savefig("./images/test_data.png", dpi=300, bbox_inches="tight")
    print(f"train accuracy: {train_acc:.4f}")
    print(f"test accuracy: {test_acc:.4f}")

    model = SVC(kernel="rbf", C=1.0)
    model.fit(x_train, y_train)
    train_acc = (model.predict(x_train) == y_train).mean()
    test_acc = (model.predict(x_test) == y_test).mean()

    print(f"sklearn: train accuracy: {train_acc:.4f}")
    print(f"sklearn: test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
