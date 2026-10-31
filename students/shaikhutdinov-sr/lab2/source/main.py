from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import data
import evaluation
import experiments
import reporting


def main() -> None:
    raw_data = data.load_data()
    x_train, x_test, y_train, y_test = data.prepare_data(raw_data)

    k_values = list(range(1, 51))
    results = experiments.run_experiments(x_train, y_train, k_values)

    metrics, predictions = evaluation.evaluate_models(results, x_test, y_test)
    print(metrics.round(4).to_string())

    lab_directory = Path(__file__).resolve().parents[1]
    reporting.save_report(
        x_train=x_train,
        y_train=y_train,
        y_test=y_test,
        results=results,
        metrics=metrics,
        predictions=predictions,
        lab_directory=lab_directory,
    )


if __name__ == "__main__":
    main()
