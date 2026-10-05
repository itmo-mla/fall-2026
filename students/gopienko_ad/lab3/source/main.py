from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC

from students.gopienko_ad.lab3.source.dataset import StandardScaler, impute_zero_values, calculate_impute_medians, \
    train_val_test_split, analyze_dataset, load_dataset
from students.gopienko_ad.lab3.source.metrics import confusion_matrix, calculate_metrics
from students.gopienko_ad.lab3.source.plots import plot_decision_boundary
from students.gopienko_ad.lab3.source.svm import SVMClassifier

RESULT_DIR = Path("result")
PLOTS_DIR = RESULT_DIR / "plots"


def print_metrics(
    name: str,
    model,
    x: np.ndarray,
    y: np.ndarray,
) -> None:
    metrics = calculate_metrics(model, x, y)
    prediction = model.predict(x)

    print(f"\n{name}")
    print("-" * len(name))

    for metric_name, value in metrics.items():
        print(
            f"{metric_name:10s}: "
            f"{value:.4f}"
        )

    print("confusion matrix:")
    print(confusion_matrix(y, prediction))
    print("support vectors:", model.n_support_,)


def compare_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    models = {
        "Custom Linear SVM": SVMClassifier(
            C=1.0,
            kernel="linear",
        ),

        "sklearn Linear SVC": SVC(
            C=1.0,
            kernel="linear",
        ),

        "Custom Poly SVM": SVMClassifier(
            C=1.0,
            kernel="poly",
            degree=3,
            gamma=0.1,
            coef0=1.0,
        ),

        "sklearn Poly SVC": SVC(
            C=1.0,
            kernel="poly",
            degree=3,
            gamma=0.1,
            coef0=1.0,
        ),

        "Custom RBF SVM": SVMClassifier(
            C=1.0,
            kernel="rbf",
            gamma=0.1,
        ),

        "sklearn RBF SVC": SVC(
            C=1.0,
            kernel="rbf",
            gamma=0.1,
        ),
    }

    for name, model in models.items():
        model.fit(X_train, y_train)

        print_metrics(
            name,
            model,
            X_test,
            y_test,
        )


def visualize(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> None:

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_train)

    models = {
        "custom_linear": SVMClassifier(
            C=1.0,
            kernel="linear",
        ),

        "custom_poly": SVMClassifier(
            C=1.0,
            kernel="poly",
            degree=3,
            gamma=0.5,
            coef0=1.0,
        ),

        "custom_rbf": SVMClassifier(
            C=1.0,
            kernel="rbf",
            gamma=0.5,
        ),

        "sklearn_linear": SVC(
            C=1.0,
            kernel="linear",
        ),

        "sklearn_poly": SVC(
            C=1.0,
            kernel="poly",
            degree=3,
            gamma=0.5,
            coef0=1.0,
        ),

        "sklearn_rbf": SVC(
            C=1.0,
            kernel="rbf",
            gamma=0.5,
        ),
    }

    for name, model in models.items():
        model.fit(X_2d, y_train)

        plot_decision_boundary(
            model,
            X_2d,
            y_train,
            title=name.replace(
                "_",
                " ",
            ).title(),
            output_path=(PLOTS_DIR / f"{name}.png")
        )


def main() -> None:
    RESULT_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset(output_dir=RESULT_DIR / "data")
    analyze_dataset(df)

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = train_val_test_split(
        df,
        target_column="class",
        val_size=0.15,
        test_size=0.15,
        stratify=True,
        random_state=42,
    )

    medians = calculate_impute_medians(X_train)

    X_train = impute_zero_values(X_train, medians)
    X_val = impute_zero_values(X_val, medians)
    X_test = impute_zero_values(X_test, medians)

    X_train = X_train.to_numpy(dtype=float)
    X_val = X_val.to_numpy(dtype=float)
    X_test = X_test.to_numpy(dtype=float)

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    print(
        "\nShapes:",
        X_train.shape,
        X_val.shape,
        X_test.shape,
    )

    compare_models(
        X_train,
        y_train,
        X_test,
        y_test,
    )

    visualize(X_train, y_train)

    print(
        f"\nPlots saved to: "
        f"{PLOTS_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()
