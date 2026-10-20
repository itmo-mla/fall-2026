from pathlib import Path

import matplotlib
from sklearn.decomposition import PCA as SklearnPCA

matplotlib.use("Agg")

import data
import evaluation
import model
import reporting


def main() -> None:
    raw_data = data.load_data()
    prepared_data = data.prepare_data(raw_data)
    variance_threshold = 0.95

    own_pca = model.PCA().fit(prepared_data.train_features)
    reference = SklearnPCA(svd_solver="full").fit(prepared_data.train_features)
    results = evaluation.evaluate_results(
        prepared_data, own_pca, reference, variance_threshold
    )

    reporting.print_results(prepared_data, results)
    reporting.save_results(
        prepared_data,
        results,
        variance_threshold,
        lab_directory=Path(__file__).resolve().parents[1],
    )


if __name__ == "__main__":
    main()
