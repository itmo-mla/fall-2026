from pathlib import Path

import matplotlib
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")

import data
import evaluation
import experiments
import reporting


def main() -> None:
    raw_data = data.load_data()
    features, labels = data.build_dataset(raw_data)
    training_features, test_features, training_labels, test_labels = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )

    parameters, validation_results = experiments.select_parameters(
        training_features, training_labels
    )
    training_features, test_features = data.standardize_features(
        training_features, test_features
    )
    results = experiments.train_models(training_features, training_labels, parameters)
    metrics, diagnostics = evaluation.evaluate_results(
        results, training_features, training_labels, test_features, test_labels
    )
    print(metrics.round(4).to_string(index=False))
    print(diagnostics.to_string(index=False))

    boundary_models = experiments.train_boundary_models(training_features, training_labels)
    lab_directory = Path(__file__).resolve().parents[1]
    reporting.save_figures(
        results=results,
        boundary_models=boundary_models,
        training_features=training_features,
        training_labels=training_labels,
        lab_directory=lab_directory,
    )
    reporting.write_report(
        results=results,
        training_labels=training_labels,
        test_labels=test_labels,
        validation_results=validation_results,
        metrics=metrics,
        diagnostics=diagnostics,
        lab_directory=lab_directory,
    )


if __name__ == "__main__":
    main()
