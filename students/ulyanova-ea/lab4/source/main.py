
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA


class PCASVD:
    def fit(self, F):
        _, sqrt_lambdas, Ut = np.linalg.svd(F, full_matrices=False)

        self.U = Ut.T
        self.lambdas = sqrt_lambdas**2
        return self

    def transform(self, F, m):
        return F @ self.U[:, :m]

    def inverse_transform(self, G):
        return G @ self.U[:, :G.shape[1]].T

    def errors(self):
        return np.array([
            self.lambdas[m:].sum() / self.lambdas.sum()
            for m in range(len(self.lambdas) + 1)
        ])

    def effective_dimension(self, epsilon=0.05):
        if not 0 <= epsilon < 1:
            raise ValueError("epsilon должен принадлежать интервалу [0, 1)")

        return int(np.where(self.errors() <= epsilon)[0][0])

    def reconstruction_error(self, F, m):
        F_reconstructed = self.inverse_transform(self.transform(F, m))
        return np.linalg.norm(F - F_reconstructed, "fro")**2 / np.linalg.norm(F, "fro")**2


def main():
    F = load_diabetes().data
    epsilon = 0.05

    model = PCASVD().fit(F)
    m = model.effective_dimension(epsilon)

    G = model.transform(F, m)
    F_reconstructed = model.inverse_transform(G)

    reference = PCA(n_components=m).fit(F)
    F_reference = reference.inverse_transform(reference.transform(F))

    np.testing.assert_allclose(F_reconstructed, F_reference, atol=1e-10)
    np.testing.assert_allclose(
        model.U[:, :m].T @ model.U[:, :m],
        np.eye(m),
        atol=1e-10
    )

    error = model.reconstruction_error(F, m)

    print(f"Размерность исходных данных: {F.shape[1]}")
    print(f"Эффективная размерность: {m}")
    print(f"Em: {error:.6f}")

    img_dir = Path(__file__).resolve().parent.parent / "img"
    img_dir.mkdir(parents=True, exist_ok=True)

    dimensions = np.arange(1, len(model.lambdas) + 1)
    errors = model.errors()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(dimensions, model.lambdas, marker="o")
    ax.set(
        xlabel="Номер главной компоненты",
        ylabel="Собственное значение λ",
        title="Собственные значения главных компонент"
    )
    ax.set_xticks(dimensions)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(img_dir / "eigenvalues.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(len(errors)), errors, marker="o")
    ax.axhline(epsilon, linestyle="--", label=f"Допустимая ошибка = {epsilon}")
    ax.set(
        xlabel="Число сохранённых компонент",
        ylabel="Em",
        title="Зависимость Em от числа компонент"
    )
    ax.set_xticks(np.arange(len(errors)))
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(img_dir / "Em.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()