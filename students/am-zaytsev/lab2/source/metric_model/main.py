from itertools import product

import numpy as np
from matplotlib import pyplot as plt

from metric_model.dataset import blood as dataset_module
from metric_model.model import AdaptiveParzenKNN, SklearnKNN
from metric_model.train import get_ref_mask, loo
from metric_model.visual.utils import draw_dataset, draw_loo_res


def main():
    # Load dataset
    df = dataset_module.download_dataset()

    dataset_module.draw_data(df)
    plt.savefig("./images/dataset_plot.png", dpi=300, bbox_inches="tight")

    # Preprocess data
    X, y = dataset_module.preprocess_data(df)

    # K Search
    h_list = np.arange(1, 30, dtype=np.int32)
    kernel_list = [
        "gauss",
        "rect",
        "tri",
        "quartic",
    ]

    search_params_list = list(product(h_list, kernel_list))
    res = loo(
        X,
        y,
        AdaptiveParzenKNN,
        model_params_list=search_params_list,
        progress_bar=True,
    )
    (best_k, best_kernel), best_loo = min(res, key=lambda x: x[1])
    print(
        f"Found best params: k: {best_k}, kernel: {best_kernel}. LOO = {round(best_loo, 2)}"
    )
    fig = draw_loo_res(res, {best_kernel})
    # fig.suptitle('Loss/k plot')
    fig.savefig("./images/choose_k")

    # SKlearn

    full_res = res.copy()
    h_list = np.arange(1, 30, dtype=np.int32)
    weights_values = ["sklearn distance", "sklearn uniform"]
    full_res += loo(
        X,
        y,
        SklearnKNN,
        model_params_list=list(product(h_list, weights_values)),
        progress_bar=True,
    )

    fig = draw_loo_res(full_res, {best_kernel} | set(weights_values))
    # fig.suptitle('Loss/k plot')
    fig.savefig("./images/compare_sklearn")

    # Ref selection
    ref_mask = get_ref_mask(X, y[:, 1][:, np.newaxis])

    res_ref = loo(X, y, AdaptiveParzenKNN, search_params_list, mask=ref_mask)
    # fig = draw_loo_res(res_ref, {best_kernel_ref})

    (best_k_ref, best_kernel_ref), best_loo_ref = min(res_ref, key=lambda x: x[1])
    print(f"{best_k_ref=} {best_kernel_ref=}")

    print("Full dataset:")
    print("   -size:", X.shape[0])
    print("   -loo:", round(best_loo, 4))

    print("Reference dataset:")
    print("   -size:", ref_mask.sum(), f"({round(ref_mask.mean() * 100, 1)}%)")
    print("   -loo:", round(best_loo_ref, 4))

    fig = draw_dataset(X, 1 - y, ref_mask)
    plt.savefig("./images/ref_dataset_plot.png", dpi=300, bbox_inches="tight")

    plt.show()
