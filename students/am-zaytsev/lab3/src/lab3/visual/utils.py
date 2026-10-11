import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def draw_dataset(X, y_true, y_pred=None, mask=None):
    if mask is None:
        mask = np.ones(X.shape[0], dtype=bool)
    y_true = np.float32(y_true)

    # Class-to-color mapping for points
    color_map = {"-1.0": "blue", "1.0": "red"}

    # Prepare colors for unmasked and masked points
    classes_unmasked = [str(val) for val, m in zip(y_true, mask) if not m]
    classes_masked = [str(val) for val, m in zip(y_true, mask) if m]
    colors_unmasked = [color_map.get(c, "gray") for c in classes_unmasked]
    colors_masked = [color_map.get(c, "gray") for c in classes_masked]

    # Scale features
    X_scaled = StandardScaler().fit_transform(X)

    # t-SNE embedding
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        random_state=42,
    )
    X_tsne = tsne.fit_transform(X_scaled)

    # Setup figure
    plt.figure(figsize=(8, 6))

    # ----- Background prediction using k-NN on t-SNE -----
    if y_pred is not None:
        # Create meshgrid covering the t-SNE plot area
        x_min, x_max = X_tsne[:, 0].min() - 1, X_tsne[:, 0].max() + 1
        y_min, y_max = X_tsne[:, 1].min() - 1, X_tsne[:, 1].max() + 1
        resolution = 200
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution)
        )

        # Fit k-NN on the 2D t-SNE coordinates using model predictions
        knn = KNeighborsClassifier(
            n_neighbors=1
        )  # you can increase n_neighbors for smoother boundaries
        knn.fit(X_tsne, y_pred)

        # Predict on grid
        Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Map labels (-1, 1) to (0, 1) for a clean colormap
        Z_numeric = (Z + 1) / 2  # -1 -> 0, 1 -> 1

        # Define colormap matching your point colors
        cmap_background = ListedColormap(["blue", "red"])

        # Plot background
        plt.contourf(
            xx, yy, Z_numeric, alpha=0.3, cmap=cmap_background, levels=[-0.5, 0.5, 1.5]
        )

    # ----- Scatter points -----
    # Unmasked (faded)
    plt.scatter(
        x=X_tsne[~mask, 0],
        y=X_tsne[~mask, 1],
        c=colors_unmasked,
        marker="o",
        s=10,
        alpha=0.3,
    )

    # Masked (focused)
    plt.scatter(
        x=X_tsne[mask, 0],
        y=X_tsne[mask, 1],
        c=colors_masked,
        marker="o",
        s=25,
        alpha=1.0,
    )

    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title("t-SNE with k-NN approximated decision background")


def draw_dataset_pca(X, y_true, y_pred=None, mask=None, model=None):
    if mask is None:
        mask = np.ones(X.shape[0], dtype=bool)
    y_true = np.float32(y_true)

    color_map = {"-1.0": "blue", "1.0": "red"}

    classes_unmasked = [str(val) for val, m in zip(y_true, mask) if not m]
    classes_masked = [str(val) for val, m in zip(y_true, mask) if m]
    colors_unmasked = [color_map.get(c, "gray") for c in classes_unmasked]
    colors_masked = [color_map.get(c, "gray") for c in classes_masked]

    X_scaled = StandardScaler().fit_transform(X)

    mean = X_scaled.mean(axis=0)
    u, s, vt = np.linalg.svd(X_scaled - mean, full_matrices=False)
    X_pca = u[:, :2] * s[:2]
    components_inv = np.linalg.pinv(vt[:2].T)

    plt.figure(figsize=(8, 6))

    if model is not None or y_pred is not None:
        x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
        y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
        resolution = 200
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution)
        )
        grid = np.c_[xx.ravel(), yy.ravel()]

        if model is not None:
            original_space = mean + grid @ components_inv
            Z = model.predict(original_space)
        else:
            knn = KNeighborsClassifier(n_neighbors=1)
            knn.fit(X_pca, y_pred)
            Z = knn.predict(grid)

        Z = Z.reshape(xx.shape)

        Z_numeric = (Z + 1) / 2

        cmap_background = ListedColormap(["blue", "red"])

        plt.contourf(
            xx, yy, Z_numeric, alpha=0.3, cmap=cmap_background, levels=[-0.5, 0.5, 1.5]
        )

    plt.scatter(
        x=X_pca[~mask, 0],
        y=X_pca[~mask, 1],
        c=colors_unmasked,
        marker="o",
        s=10,
        alpha=0.3,
    )

    plt.scatter(
        x=X_pca[mask, 0],
        y=X_pca[mask, 1],
        c=colors_masked,
        marker="o",
        s=25,
        alpha=1.0,
    )

    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    if model is not None:
        plt.title("PCA with model decision background")
    else:
        plt.title("PCA with k-NN approximated decision background")
