from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


FEATURES = [
    ("culmen_length_mm", "Длина клюва, мм"),
    ("culmen_depth_mm", "Глубина клюва, мм"),
    ("flipper_length_mm", "Длина ласта, мм"),
    ("body_mass_g", "Масса тела, г"),
]
SPECIES_COLORS = {
    "Adelie": "#2878B5",
    "Gentoo": "#E07A2D",
}


def main():
    data_path = Path(__file__).with_name("penguins.csv")
    data = pd.read_csv(data_path)
    feature_names = [name for name, _ in FEATURES]
    data = data[data["species"].isin(SPECIES_COLORS)].dropna(
        subset=feature_names
    )

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(15, 11),
        subplot_kw={"projection": "3d"},
        constrained_layout=True,
    )

    for omitted_index, axis in enumerate(axes.flat):
        plotted_features = [
            index for index in range(len(FEATURES)) if index != omitted_index
        ]
        for species, color in SPECIES_COLORS.items():
            species_data = data[data["species"] == species]
            axis.scatter(
                *(species_data[FEATURES[index][0]] for index in plotted_features),
                label=species,
                color=color,
                alpha=0.8,
                s=28,
            )

        axis.set_xlabel(FEATURES[plotted_features[0]][1])
        axis.set_ylabel(FEATURES[plotted_features[1]][1])
        axis.set_zlabel(FEATURES[plotted_features[2]][1])
        axis.set_title(
            f"Признаки {', '.join(str(index + 1) for index in plotted_features)} "
            f"(признак {omitted_index + 1} не учитывается)"
        )
        axis.legend(title="Вид")

    figure.suptitle(f"Пингвины Adelie и Gentoo: {len(data)} записей")
    plt.show()


if __name__ == "__main__":
    main()