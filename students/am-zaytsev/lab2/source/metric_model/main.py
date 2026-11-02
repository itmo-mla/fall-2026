from itertools import product

import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from metric_model.dataset import pima as dataset_module
from metric_model.model import AdaptiveParzenKNN, SklearnKNN
from metric_model.train import get_ref_mask, loo
from metric_model.visual.report_generatop import generate_report
from metric_model.visual.utils import draw_dataset, draw_loo_res


def main():
    # Load dataset
    df = dataset_module.download_dataset()

    dataset_module.draw_data(df)
    # plt.savefig("./images/dataset_plot.png", dpi=300, bbox_inches="tight")

    # Preprocess data
    X, y = dataset_module.preprocess_data(df)

    # Hold-out test: fixed before any parameter search
    idx = np.arange(X.shape[0])
    train_idx, test_idx = train_test_split(
        idx, test_size=0.3, random_state=42, stratify=y.argmax(axis=1)
    )
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # K Search
    h_list = np.arange(1, 50, 2, dtype=np.int32)
    kernel_list = [
        "gauss",
        "rect",
        "tri",
        "quartic",
    ]

    search_params_list = list(product(h_list, kernel_list))
    res = loo(
        X_train,
        y_train,
        AdaptiveParzenKNN,
        model_params_list=search_params_list,
        progress_bar=True,
    )
    (best_k, best_kernel), best_loo = min(res, key=lambda x: x[1])
    my_best_k, my_best_kernel, my_best_loo = best_k, best_kernel, best_loo
    print(
        f"Full model (train LOO): k={best_k}, kernel={best_kernel}, "
        f"LOO={best_loo:.4f}"
    )
    model = AdaptiveParzenKNN(best_k, best_kernel)
    model.train(X_train, y_train)
    acc_full = model.acc(X_test, y_test)
    print("Full model test acc:", acc_full)
    fig = draw_loo_res(res, {best_kernel})
    # fig.suptitle('Loss/k plot')
    fig.savefig("./images/choose_k")

    # SKlearn

    full_res = res.copy()
    weights_values = ["sklearn distance", "sklearn uniform"]
    sk_res = loo(
        X_train,
        y_train,
        SklearnKNN,
        model_params_list=list(product(h_list, weights_values)),
        progress_bar=True,
    )
    full_res += sk_res
    (sk_best_k, sk_best_kernel), sk_best_loo = min(sk_res, key=lambda x: x[1])
    print(
        f"sklearn model (train LOO): k={sk_best_k}, weights={sk_best_kernel}, "
        f"LOO={sk_best_loo:.4f}"
    )

    model = SklearnKNN(sk_best_k, sk_best_kernel)
    model.train(X_train, y_train)
    acc_sk = model.acc(X_test, y_test)
    print("sklearn test acc:", acc_sk)
    fig = draw_loo_res(full_res, {sk_best_kernel} | set(weights_values))
    # fig.suptitle('Loss/k plot')
    fig.savefig("./images/compare_sklearn")

    # Ref selection
    ref_mask = get_ref_mask(X_train, y_train[:, 1][:, np.newaxis])
    X_ref, y_ref = X_train[ref_mask], y_train[ref_mask]

    res_ref = loo(
        X_train,
        y_train,
        AdaptiveParzenKNN,
        model_params_list=search_params_list,
        mask=ref_mask,
        progress_bar=True,
    )
    # fig = draw_loo_res(res_ref, {best_kernel_ref})

    (best_k_ref, best_kernel_ref), best_loo_ref = min(res_ref, key=lambda x: x[1])
    print(
        f"Ref model (train LOO, mask=ref_mask): k={best_k_ref}, "
        f"kernel={best_kernel_ref}, LOO={best_loo_ref:.4f}"
    )

    model = AdaptiveParzenKNN(best_k_ref, best_kernel_ref)
    model.train(X_ref, y_ref)
    acc_ref = model.acc(X_test, y_test)
    print("Ref model test acc:", acc_ref)

    print("Parameter search (train only):")
    print(
        f"  full:    k={int(my_best_k)}, kernel={my_best_kernel}, "
        f"LOO={my_best_loo:.4f}"
    )
    print(
        f"  ref:     k={int(best_k_ref)}, kernel={best_kernel_ref}, "
        f"LOO={best_loo_ref:.4f} (n_ref={ref_mask.sum()})"
    )
    print(
        f"  sklearn: k={int(sk_best_k)}, weights={sk_best_kernel}, "
        f"LOO={sk_best_loo:.4f}"
    )

    print(f"Common test (untouched, n={X_test.shape[0]}):")
    print(f"  full:    acc={acc_full:.4f}")
    print(f"  ref:     acc={acc_ref:.4f}")
    print(f"  sklearn: acc={acc_sk:.4f}")

    fig = draw_dataset(X_train, 1 - y_train, ref_mask)
    plt.savefig("./images/ref_dataset_plot.png", dpi=300, bbox_inches="tight")

    plt.show()

    generate_report(
        {
            "dataset_desc": dataset_module.DATASET_DESC,
            "h_best_my": int(my_best_k),
            "kernel_best_my": my_best_kernel,
            "llo_best_my": round(float(my_best_loo), 4),
            "h_best_sk": int(sk_best_k),
            "kernel_best_sk": sk_best_kernel,
            "llo_best_sk": round(float(sk_best_loo), 4),
            "acc_best_my": round(float(acc_full), 4),
            "acc_best_sk": round(float(acc_sk), 4),
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
            "ref_size": int(ref_mask.sum()),
            "ref_size_percentage": round(float(ref_mask.mean()) * 100, 1),
            "ref_loo": round(float(best_loo_ref), 4),
            "acc_ref": round(float(acc_ref), 4),
        }
    )


if __name__ == "__main__":
    main()
